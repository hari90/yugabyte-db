// Copyright (c) YugabyteDB, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied.  See the License for the specific language governing permissions and limitations
// under the License.
//
// ScaNN adapter for VectorIndexIf.
//
// The ScaNN config is hardcoded to brute-force without fixed-point
// quantization, which allows dynamic inserts after the initial batch.
// The distance measure is derived from HNSWOptions::distance_kind so that
// ScaNN's search results are directly usable without recomputation.
//
// Persistence stores every (VectorId, Vector) pair in a single flat binary
// file.  On load the ScaNN index is rebuilt from the raw vectors so there is
// no dependency on ScaNN's own serialization format.

#include "yb/ann_methods/scann_wrapper_adapter.h"

#include <atomic>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

#include "yb/gutil/casts.h"

#include "yb/util/status.h"

#include "yb/vector_index/distance.h"
#include "yb/vector_index/index_wrapper_base.h"
#include "yb/vector_index/vector_index_if.h"

// ScaNN wrapper headers — safe to include here (no absl leakage).
#include "scann/scann_wrapper.h"

namespace yb::ann_methods {

using vector_index::DistanceKind;
using vector_index::HNSWOptions;
using vector_index::IndexableVectorType;
using vector_index::IndexWrapperBase;
using vector_index::SearchOptions;
using vector_index::ValidDistanceResultType;
using vector_index::VectorId;
using vector_index::VectorIndexIfPtr;
using vector_index::VectorWithDistance;

namespace {

// ---------------------------------------------------------------------------
// Hardcoded ScaNN parameters
// ---------------------------------------------------------------------------

// Maximum number of neighbors the ScaNN config advertises.  The actual number
// returned per query is controlled by the `final_nn` argument at search time.
constexpr int kScannMaxNeighbors = 1000;

// Number of training threads (only meaningful for tree/AH configs; brute-force
// ignores this, but ScaNN still requires a value).
constexpr int kScannTrainingThreads = 1;

// UUID size in bytes.
constexpr size_t kUuidBytes = 16;

// ---------------------------------------------------------------------------
// Map DistanceKind → ScaNN distance measure string
// ---------------------------------------------------------------------------

const std::string& ScannDistanceMeasure(DistanceKind kind) {
  static const std::string kSquaredL2 = "SquaredL2Distance";
  static const std::string kDotProduct = "DotProductDistance";
  static const std::string kCosine = "CosineDistance";
  switch (kind) {
    case DistanceKind::kL2Squared:
      return kSquaredL2;
    case DistanceKind::kInnerProduct:
      return kDotProduct;
    case DistanceKind::kCosine:
      return kCosine;
  }
  LOG(DFATAL) << "Unknown DistanceKind: " << static_cast<int>(kind);
  return kSquaredL2;
}

// ---------------------------------------------------------------------------
// File format for SaveToFile / LoadFromFile
//
//   [uint32_t] dimensions
//   [uint32_t] num_vectors
//   For each vector:
//     [16 bytes]                           VectorId UUID
//     [dimensions * sizeof(float)]         vector data (as float)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ScannVectorIterator — iterates over the stored (VectorId, Vector) entries
// ---------------------------------------------------------------------------

template <IndexableVectorType Vector, ValidDistanceResultType DistanceResult>
class ScannVectorIterator : public AbstractIterator<std::pair<VectorId, Vector>> {
 public:
  using Entry = std::pair<VectorId, Vector>;

  ScannVectorIterator(
      const std::vector<Entry>* entries,
      size_t position)
      : entries_(entries), position_(position) {}

 protected:
  Entry Dereference() const override {
    return (*entries_)[position_];
  }

  void Next() override {
    ++position_;
  }

  bool NotEquals(const AbstractIterator<Entry>& other) const override {
    const auto& o = down_cast<const ScannVectorIterator&>(other);
    return position_ != o.position_;
  }

 private:
  const std::vector<Entry>* entries_;
  size_t position_;
};

// ---------------------------------------------------------------------------
// ScannIndex — VectorIndexIf adapter around yb::scann::ScannWrapper
// ---------------------------------------------------------------------------

template<IndexableVectorType Vector, ValidDistanceResultType DistanceResult>
class ScannIndex :
    public IndexWrapperBase<ScannIndex<Vector, DistanceResult>, Vector, DistanceResult> {
 public:
  using Scalar = typename Vector::value_type;
  using Entry = std::pair<VectorId, Vector>;
  using IteratorImpl = ScannVectorIterator<Vector, DistanceResult>;

  explicit ScannIndex(const HNSWOptions& options) : options_(options) {}

  // ---------------------------------------------------------------------------
  // VectorIndexWriterIf
  // ---------------------------------------------------------------------------

  Status Reserve(size_t num_vectors, size_t, size_t) override {
    capacity_ = num_vectors;
    entries_.reserve(num_vectors);
    return Status::OK();
  }

  // Called from IndexWrapperBase::Insert (non-const).
  Status DoInsert(VectorId vector_id, const Vector& v) {
    std::vector<float> fvec(v.begin(), v.end());

    if (!initialized_.load(std::memory_order_acquire)) {
      // First insert — initialise ScaNN with this single vector as the
      // seed dataset.  Subsequent inserts use dynamic Insert().
      auto config = scann::ScannBruteForceConfig(
          kScannMaxNeighbors,
          static_cast<int>(options_.dimensions),
          /*fixed_point=*/false,
          ScannDistanceMeasure(options_.distance_kind));

      std::vector<scann::ScannVectorId> labels = {
        scann::ScannVectorId(vector_id.GetUuid())
      };

      RETURN_NOT_OK(scann_.Initialize(
          fvec,
          /*n_points=*/1,
          config,
          kScannTrainingThreads,
          labels));

      initialized_.store(true, std::memory_order_release);
    } else {
      // Dynamic insert into an already-initialized index.
      auto label = scann::ScannVectorId(vector_id.GetUuid());
      VERIFY_RESULT(scann_.Insert(fvec, vector_id.ToString(), label));
    }

    // Keep a local copy for iteration / persistence.
    entries_.emplace_back(vector_id, v);
    return Status::OK();
  }

  size_t Size() const override {
    return entries_.size();
  }

  size_t Capacity() const override {
    return capacity_;
  }

  // ---------------------------------------------------------------------------
  // VectorIndexIf
  // ---------------------------------------------------------------------------

  size_t Dimensions() const override {
    return options_.dimensions;
  }

  // ---------------------------------------------------------------------------
  // Search
  // ---------------------------------------------------------------------------

  std::vector<VectorWithDistance<DistanceResult>> DoSearch(
      const Vector& query_vector, const SearchOptions& options) const {
    if (!initialized_.load(std::memory_order_acquire)) {
      return {};
    }

    std::vector<float> query(query_vector.begin(), query_vector.end());
    int k = static_cast<int>(options.max_num_results);
    auto scann_results = scann_.Search(query, k, k, /*leaves=*/0);
    if (!scann_results.ok()) {
      LOG(WARNING) << "ScaNN search failed: " << scann_results.status();
      return {};
    }

    std::vector<VectorWithDistance<DistanceResult>> result;
    result.reserve(scann_results->size());
    for (const auto& r : *scann_results) {
      auto vid = VectorId(r.label.GetUuid());
      if (!vid.IsNil() && options.filter(vid)) {
        result.push_back({vid, static_cast<DistanceResult>(r.distance)});
      }
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Distance
  // ---------------------------------------------------------------------------

  DistanceResult Distance(const Vector& lhs, const Vector& rhs) const override {
    auto fn = vector_index::GetDistanceFunction<Vector, DistanceResult>(options_.distance_kind);
    return fn(lhs, rhs);
  }

  // ---------------------------------------------------------------------------
  // Persistence
  // ---------------------------------------------------------------------------

  Result<VectorIndexIfPtr<Vector, DistanceResult>> DoSaveToFile(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      return STATUS_FORMAT(IOError, "Cannot open $0 for writing", path);
    }

    auto dims = static_cast<uint32_t>(options_.dimensions);
    auto count = static_cast<uint32_t>(entries_.size());
    out.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& [vid, vec] : entries_) {
      // Write the 16-byte VectorId UUID.
      out.write(reinterpret_cast<const char*>(vid.data()), kUuidBytes);

      // Write the vector as floats.
      std::vector<float> fvec(vec.begin(), vec.end());
      out.write(reinterpret_cast<const char*>(fvec.data()),
                static_cast<std::streamsize>(dims * sizeof(float)));
    }

    if (!out) {
      return STATUS_FORMAT(IOError, "Error writing ScaNN index to $0", path);
    }

    return nullptr;
  }

  Status DoLoadFromFile(const std::string& path, size_t) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return STATUS_FORMAT(IOError, "Cannot open $0 for reading", path);
    }

    uint32_t dims = 0;
    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in) {
      return STATUS_FORMAT(Corruption, "Failed to read header from $0", path);
    }

    if (dims != options_.dimensions) {
      return STATUS_FORMAT(
          Corruption,
          "Dimension mismatch in $0: file has $1, expected $2",
          path, dims, options_.dimensions);
    }

    // Read all entries.
    entries_.clear();
    entries_.reserve(count);

    std::vector<float> flat_dataset;
    flat_dataset.reserve(static_cast<size_t>(count) * dims);

    std::vector<scann::ScannVectorId> labels;
    labels.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
      // Read VectorId.
      uint8_t uuid_bytes[kUuidBytes];
      in.read(reinterpret_cast<char*>(uuid_bytes), kUuidBytes);
      if (!in) {
        return STATUS_FORMAT(Corruption, "Failed to read VectorId $0 from $1", i, path);
      }
      auto vid = VectorId(Uuid::TryFullyDecode(Slice(uuid_bytes, kUuidBytes)));

      // Read vector data as floats.
      std::vector<float> fvec(dims);
      in.read(reinterpret_cast<char*>(fvec.data()),
              static_cast<std::streamsize>(dims * sizeof(float)));
      if (!in) {
        return STATUS_FORMAT(Corruption, "Failed to read vector $0 from $1", i, path);
      }

      // Build the typed vector and collect into entries_.
      Vector vec(fvec.begin(), fvec.end());
      entries_.emplace_back(vid, std::move(vec));

      flat_dataset.insert(flat_dataset.end(), fvec.begin(), fvec.end());
      labels.push_back(scann::ScannVectorId(vid.GetUuid()));
    }

    // Rebuild the ScaNN index from the raw vectors.
    auto config = scann::ScannBruteForceConfig(
        kScannMaxNeighbors, static_cast<int>(dims), /*fixed_point=*/false,
        ScannDistanceMeasure(options_.distance_kind));

    RETURN_NOT_OK(scann_.Initialize(
        flat_dataset, count, config, kScannTrainingThreads, labels));

    initialized_.store(true, std::memory_order_release);
    capacity_ = std::max(capacity_, static_cast<size_t>(count));
    return Status::OK();
  }

  // ---------------------------------------------------------------------------
  // Vector retrieval / iteration
  // ---------------------------------------------------------------------------

  Result<Vector> GetVector(VectorId) const override {
    return STATUS(NotSupported, "ScaNN adapter does not support GetVector");
  }

  std::unique_ptr<AbstractIterator<Entry>> BeginImpl() const override {
    return std::make_unique<IteratorImpl>(&entries_, 0);
  }

  std::unique_ptr<AbstractIterator<Entry>> EndImpl() const override {
    return std::make_unique<IteratorImpl>(&entries_, entries_.size());
  }

  std::string IndexStatsStr() const override {
    return Format("ScaNN brute-force index, $0 vectors, $1 dimensions",
                  entries_.size(), options_.dimensions);
  }

 private:
  HNSWOptions options_;
  size_t capacity_ = 0;

  // ScaNN is lazily initialized on first DoInsert (or DoLoadFromFile).
  std::atomic<bool> initialized_{false};
  scann::ScannWrapper scann_;

  // Local copy of all (VectorId, Vector) entries for iteration and persistence.
  std::vector<Entry> entries_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

template <IndexableVectorType Vector, ValidDistanceResultType DistanceResult>
VectorIndexIfPtr<Vector, DistanceResult> ScannIndexFactory<Vector, DistanceResult>::Create(
    vector_index::FactoryMode mode, const HNSWOptions& options) {
  return std::make_shared<ScannIndex<Vector, DistanceResult>>(options);
}

// Explicit instantiation — ScaNN only supports float vectors / float distances.
template class ScannIndexFactory<FloatVector, float>;

}  // namespace yb::ann_methods
