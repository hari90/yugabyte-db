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
// Persistence stores every (VectorId, ybctid, Vector) tuple in a single flat
// binary file.  On load the ScaNN index is rebuilt from the raw vectors so
// there is no dependency on ScaNN's own serialization format.
//
// The ScaNN label for each datapoint is the concatenation of the 16-byte
// VectorId UUID and the variable-length ybctid:
//
//   label = [16 bytes VectorId][ybctid bytes]
//
// On search, VectorId and ybctid are decoded directly from the label
// returned in ScannSearchResult, avoiding any lookup into entries_.
//
// Vector id is not stored in ScaNN as a docid — a simple sequential counter
// is used instead.  The adapter keeps a local entries_ vector for
// iteration / persistence.

#include "yb/ann_methods/scann_wrapper_adapter.h"

#include <cstring>
#include <fstream>
#include <mutex>
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

// UUID size in bytes (VectorId is a UUID).
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
// Label encoding / decoding
//
// The ScaNN label encodes both VectorId and ybctid in a single byte string:
//   [16 bytes VectorId UUID][variable-length ybctid]
// ---------------------------------------------------------------------------

// Encode VectorId + ybctid into a single label string.
std::string EncodeLabel(VectorId vector_id, Slice ybctid) {
  std::string label;
  label.resize(kUuidBytes + ybctid.size());
  std::memcpy(label.data(), vector_id.data(), kUuidBytes);
  if (!ybctid.empty()) {
    std::memcpy(label.data() + kUuidBytes, ybctid.data(), ybctid.size());
  }
  return label;
}

// Decode VectorId from a label (first 16 bytes).
VectorId DecodeVectorId(const std::string& label) {
  DCHECK_GE(label.size(), kUuidBytes);
  return VectorId(Uuid::TryFullyDecode(
      Slice(reinterpret_cast<const uint8_t*>(label.data()), kUuidBytes)));
}

// Decode ybctid from a label (everything after the first 16 bytes).
std::string DecodeYbctid(const std::string& label) {
  if (label.size() <= kUuidBytes) {
    return {};
  }
  return label.substr(kUuidBytes);
}

// ---------------------------------------------------------------------------
// File format for SaveToFile / LoadFromFile
//
//   [uint32_t] dimensions
//   [uint32_t] num_vectors
//   For each vector:
//     [16 bytes]                           VectorId UUID
//     [uint16_t]                           ybctid_length (0 if absent)
//     [ybctid_length bytes]                ybctid data
//     [dimensions * sizeof(float)]         vector data (as float)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Internal entry type — holds VectorId, Vector, and ybctid together.
// ---------------------------------------------------------------------------

template <IndexableVectorType Vector>
struct StoredEntry {
  VectorId vector_id;
  Vector vector;
  std::string ybctid;
};

// ---------------------------------------------------------------------------
// ScannVectorIterator — iterates over the stored entries, producing
// (VectorId, Vector) pairs as required by the VectorIndexIf interface.
// ---------------------------------------------------------------------------

template <IndexableVectorType Vector, ValidDistanceResultType DistanceResult>
class ScannVectorIterator : public AbstractIterator<std::pair<VectorId, Vector>> {
 public:
  using IterEntry = std::pair<VectorId, Vector>;

  ScannVectorIterator(
      const std::vector<StoredEntry<Vector>>* entries,
      size_t position)
      : entries_(entries), position_(position) {}

 protected:
  IterEntry Dereference() const override {
    const auto& e = (*entries_)[position_];
    return {e.vector_id, e.vector};
  }

  void Next() override {
    ++position_;
  }

  bool NotEquals(const AbstractIterator<IterEntry>& other) const override {
    const auto& o = down_cast<const ScannVectorIterator&>(other);
    return position_ != o.position_;
  }

 private:
  const std::vector<StoredEntry<Vector>>* entries_;
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
  // The ScaNN label stores VectorId + ybctid so that search can decode both
  // without looking up entries_.
  // Vector id is not stored in ScaNN as a docid — a simple sequential counter
  // is used instead.
  //
  // VectorLSM dispatches inserts in parallel via a thread pool.  ScannWrapper
  // protects its own internals, but we still need to serialize here so that
  // entries_ stays in sync with the ScaNN index and the initialized_ flag is
  // checked consistently with the Initialize/Insert call.
  Status DoInsert(VectorId vector_id, const Vector& v, Slice ybctid = Slice()) {
    std::vector<float> fvec(v.begin(), v.end());
    std::string label = EncodeLabel(vector_id, ybctid);

    std::lock_guard lock(mutex_);

    if (!initialized_) {
      // First insert — initialise ScaNN with this single vector as the
      // seed dataset.  Subsequent inserts use dynamic Insert().
      auto config = scann::ScannBruteForceConfig(
          kScannMaxNeighbors,
          static_cast<int>(options_.dimensions),
          /*fixed_point=*/false,
          ScannDistanceMeasure(options_.distance_kind));

      std::vector<Slice> labels;
      labels.emplace_back(label);

      RETURN_NOT_OK(scann_.Initialize(
          fvec,
          /*n_points=*/1,
          config,
          kScannTrainingThreads,
          labels));

      initialized_ = true;
    } else {
      // Dynamic insert into an already-initialized index.
      // Use a sequential docid (vector id is not used as docid in ScaNN).
      std::string docid = std::to_string(next_docid_++);
      VERIFY_RESULT(scann_.Insert(fvec, docid, Slice(label)));
    }

    // Keep a local copy for iteration / persistence.
    entries_.push_back({vector_id, v, ybctid.ToBuffer()});
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
    std::lock_guard lock(mutex_);
    if (!initialized_) {
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
      // Decode VectorId and ybctid directly from the label.
      if (r.label.size() < kUuidBytes) {
        continue;
      }
      auto vid = DecodeVectorId(r.label);
      if (!options.filter(vid)) {
        continue;
      }
      auto ybctid = DecodeYbctid(r.label);
      result.emplace_back(
          vid, static_cast<DistanceResult>(r.distance), std::move(ybctid));
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

    for (const auto& entry : entries_) {
      // Write the 16-byte VectorId UUID.
      out.write(reinterpret_cast<const char*>(entry.vector_id.data()), kUuidBytes);

      // Write ybctid: [uint16_t length][data].
      auto ybctid_len = static_cast<uint16_t>(entry.ybctid.size());
      out.write(reinterpret_cast<const char*>(&ybctid_len), sizeof(ybctid_len));
      if (ybctid_len > 0) {
        out.write(entry.ybctid.data(), ybctid_len);
      }

      // Write the vector as floats.
      std::vector<float> fvec(entry.vector.begin(), entry.vector.end());
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

    // Build encoded labels (VectorId + ybctid) for ScaNN initialization.
    std::vector<std::string> label_storage;
    label_storage.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
      // Read VectorId.
      uint8_t uuid_bytes[kUuidBytes];
      in.read(reinterpret_cast<char*>(uuid_bytes), kUuidBytes);
      if (!in) {
        return STATUS_FORMAT(Corruption, "Failed to read VectorId $0 from $1", i, path);
      }
      auto vid = VectorId(Uuid::TryFullyDecode(Slice(uuid_bytes, kUuidBytes)));

      // Read ybctid: [uint16_t length][data].
      uint16_t ybctid_len = 0;
      in.read(reinterpret_cast<char*>(&ybctid_len), sizeof(ybctid_len));
      if (!in) {
        return STATUS_FORMAT(Corruption, "Failed to read ybctid length $0 from $1", i, path);
      }
      std::string ybctid;
      if (ybctid_len > 0) {
        ybctid.resize(ybctid_len);
        in.read(ybctid.data(), ybctid_len);
        if (!in) {
          return STATUS_FORMAT(Corruption, "Failed to read ybctid data $0 from $1", i, path);
        }
      }

      // Read vector data as floats.
      std::vector<float> fvec(dims);
      in.read(reinterpret_cast<char*>(fvec.data()),
              static_cast<std::streamsize>(dims * sizeof(float)));
      if (!in) {
        return STATUS_FORMAT(Corruption, "Failed to read vector $0 from $1", i, path);
      }

      // Build encoded label (VectorId + ybctid) for ScaNN.
      label_storage.push_back(EncodeLabel(vid, Slice(ybctid)));

      // Build the typed vector and collect into entries_.
      Vector vec(fvec.begin(), fvec.end());
      entries_.push_back({vid, std::move(vec), std::move(ybctid)});

      flat_dataset.insert(flat_dataset.end(), fvec.begin(), fvec.end());
    }

    // Build Slice labels pointing into label_storage.
    std::vector<Slice> labels;
    labels.reserve(count);
    for (const auto& s : label_storage) {
      labels.emplace_back(s);
    }

    // Rebuild the ScaNN index from the raw vectors with encoded labels.
    auto config = scann::ScannBruteForceConfig(
        kScannMaxNeighbors, static_cast<int>(dims), /*fixed_point=*/false,
        ScannDistanceMeasure(options_.distance_kind));

    RETURN_NOT_OK(scann_.Initialize(
        flat_dataset, count, config, kScannTrainingThreads, labels));

    initialized_ = true;
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

  // Protects initialized_, entries_, and next_docid_ from concurrent access.
  // ScannWrapper has its own internal lock for ScaNN data structures; this
  // mutex ensures the adapter's local bookkeeping stays in sync.
  mutable std::mutex mutex_;

  // ScaNN is lazily initialized on first DoInsert (or DoLoadFromFile).
  bool initialized_ = false;
  scann::ScannWrapper scann_;

  // Sequential docid counter for ScaNN Insert (vector id is not used as docid).
  uint64_t next_docid_ = 1;

  // Local copy of all entries for iteration and persistence.
  // Each entry holds VectorId, Vector, and ybctid.
  std::vector<StoredEntry<Vector>> entries_;
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
