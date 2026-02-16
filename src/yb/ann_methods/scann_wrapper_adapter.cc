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
// Persistence is delegated to ScannWrapper::Serialize() and
// ScannWrapper::LoadFromDisk(), which store / restore the ScaNN index and its
// label map to / from a directory on disk.
//
// The ScaNN label for each datapoint is the concatenation of the 16-byte
// VectorId UUID and the variable-length ybctid:
//
//   label = [16 bytes VectorId][ybctid bytes]
//
// On search, VectorId and ybctid are decoded directly from the label
// returned in ScannSearchResult, avoiding any external lookup.
//
// Vector id is not stored in ScaNN as a docid — a simple sequential counter
// is used instead.  Iteration reads vectors and labels directly from the
// ScaNN index via ScannWrapper::GetDatapoint().

#include "yb/ann_methods/scann_wrapper_adapter.h"

#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "yb/gutil/casts.h"

#include "yb/util/env.h"
#include "yb/util/flags.h"
#include "yb/util/status.h"

#include "yb/vector_index/distance.h"
#include "yb/vector_index/index_wrapper_base.h"
#include "yb/vector_index/vector_index_if.h"

// ScaNN wrapper headers — safe to include here (no absl leakage).
#include "scann/scann_wrapper.h"

// ---------------------------------------------------------------------------
// gFlags for ScaNN tree index configuration
// ---------------------------------------------------------------------------

DEFINE_RUNTIME_int32(scann_max_num_levels, 1,
    "Maximum number of centroid levels in the ScaNN K-means clustering tree. "
    "1 = two-level tree (recommended for < 10M vectors). "
    "2 = three-level tree (recommended for > 100M vectors). "
    "For 10M-100M vectors, use 2 for faster build or 1 for better recall.");

DEFINE_RUNTIME_int32(scann_num_leaves, 0,
    "Number of partitions (leaves) for the ScaNN tree index. "
    "0 = auto-calculate based on dataset size (ROWS/100). "
    "Two-level tree: use sqrt(ROWS) as a starting point, or ROWS/100 for "
    "optimal recall. "
    "Three-level tree: use power(ROWS, 2/3) as a starting point, or ROWS/100 "
    "for optimal recall. "
    "Valid range: 1 to 1048576.");

DEFINE_RUNTIME_string(scann_quantizer, "SQ8",
    "Quantizer type for the ScaNN tree index. "
    "SQ8 = asymmetric hashing (faster search, minimal recall loss < 1-2%). "
    "FLAT = exact brute-force scoring (recall >= 99%).");

DEFINE_RUNTIME_bool(scann_enable_pca, true,
    "Enable Principal Component Analysis (PCA) for ScaNN's asymmetric hashing "
    "projection. PCA automatically reduces embedding dimensions when possible, "
    "improving quantization efficiency. Set to false if recall degrades.");

// ---------------------------------------------------------------------------
// gFlags for ScaNN query-time configuration
// ---------------------------------------------------------------------------

DEFINE_RUNTIME_int32(scann_num_leaves_to_search, 0,
    "Number of tree leaves (partitions) to search per query. "
    "Higher values improve recall at the cost of QPS. "
    "0 = auto-calculate as 1% of num_leaves.");

DEFINE_RUNTIME_int32(scann_pre_reordering_num_neighbors, 0,
    "Number of candidate neighbors to consider during exact reordering after "
    "the initial AH search. Must be >= the final number of neighbors (LIMIT). "
    "Higher values improve recall at the cost of QPS. "
    "0 = disabled (no reordering) when PCA is off; 50 * LIMIT when PCA is on.");

DEFINE_RUNTIME_int32(scann_num_search_threads, 2,
    "Number of threads for parallel ScaNN ANN search. "
    "Increasing this reduces single-query latency if the server is not already "
    "CPU-bound. Default is 2.");

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

// When the index has at least this many vectors, DoSaveToFile rebuilds
// the index with a Tree-AH config before serializing.  Tree-AH is
// significantly faster for search on large datasets.
constexpr size_t kTreeAhThreshold = 100000;

// Number of training threads used when rebuilding for Tree-AH on save.
constexpr int kTreeAhTrainingThreads = 4;

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
std::string EncodeLabel(const VectorId& vector_id, Slice ybctid) {
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
// ScannVectorIterator — iterates over the ScaNN index, producing
// (VectorId, Vector) pairs by reading vectors and labels directly from the
// ScannWrapper via GetDatapoint().
// ---------------------------------------------------------------------------

template <IndexableVectorType Vector, ValidDistanceResultType DistanceResult>
class ScannVectorIterator : public AbstractIterator<std::pair<VectorId, Vector>> {
 public:
  using IterEntry = std::pair<VectorId, Vector>;

  ScannVectorIterator(
      const scann::ScannWrapper* scann,
      size_t position)
      : scann_(scann), position_(position) {}

 protected:
  IterEntry Dereference() const override {
    // Retrieve both vector and label in a single call.
    auto dp_result = scann_->GetDatapoint(static_cast<int32_t>(position_));
    if (!dp_result.ok()) {
      LOG(DFATAL) << "Failed to get datapoint " << position_
                  << " from ScaNN index: " << dp_result.status();
      return {VectorId(), Vector(scann_->dimensionality(), 0)};
    }
    auto& dp = *dp_result;

    // Decode VectorId from the label (first 16 bytes).
    auto vid = DecodeVectorId(dp.label);

    Vector vec(dp.vector.begin(), dp.vector.end());
    return {vid, std::move(vec)};
  }

  void Next() override {
    ++position_;
  }

  bool NotEquals(const AbstractIterator<IterEntry>& other) const override {
    const auto& o = down_cast<const ScannVectorIterator&>(other);
    return position_ != o.position_;
  }

 private:
  const scann::ScannWrapper* scann_;
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
    return Status::OK();
  }

  // Called from IndexWrapperBase::Insert (non-const).
  // The ScaNN label stores VectorId + ybctid so that search can decode both
  // without any external lookup.
  // Vector id is not stored in ScaNN as a docid — a simple sequential counter
  // is used instead.
  //
  // VectorLSM dispatches inserts in parallel via a thread pool.  ScannWrapper
  // protects its own internals, but we still need to serialize here so that
  // the initialized_ flag is checked consistently with the
  // Initialize/Insert call.
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

      scann_.SetNumThreads(FLAGS_scann_num_search_threads);
      initialized_ = true;
    } else {
      // Dynamic insert into an already-initialized index.
      // Use a sequential docid (vector id is not used as docid in ScaNN).
      std::string docid = std::to_string(next_docid_++);
      VERIFY_RESULT(scann_.Insert(fvec, docid, Slice(label)));
    }

    return Status::OK();
  }

  size_t Size() const override {
    return initialized_ ? scann_.n_points() : 0;
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

    int leaves = FLAGS_scann_num_leaves_to_search;
    int pre_reorder_nn = FLAGS_scann_pre_reordering_num_neighbors > 0
                             ? FLAGS_scann_pre_reordering_num_neighbors
                             : (FLAGS_scann_enable_pca ? k * 50 : k);

    auto scann_results = scann_.Search(query, k, pre_reorder_nn, leaves);
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
    const auto n = Size();
    LOG(INFO) << "ScaNN: DoSaveToFile: Count: " << n << ", path: " << path;
    RETURN_NOT_OK(Env::Default()->CreateDirs(path));

    // Hold the mutex for the entire save: Rebuild() replaces impl_ in place,
    // so concurrent searches must be blocked to avoid a use-after-free on the
    // old searcher.  This is acceptable because save is called once when the
    // mutable chunk is being finalized.
    std::lock_guard lock(mutex_);

    if (n >= kTreeAhThreshold) {
      const auto dim = static_cast<int>(options_.dimensions);
      const auto& distance = ScannDistanceMeasure(options_.distance_kind);

      // Build tree options from gFlags.
      scann::ScannTreeOptions tree_opts;
      tree_opts.num_leaves = FLAGS_scann_num_leaves > 0
                                 ? FLAGS_scann_num_leaves
                                 : static_cast<int>(n / 100);
      tree_opts.max_num_levels = FLAGS_scann_max_num_levels;
      tree_opts.enable_pca = FLAGS_scann_enable_pca;

      scann_internal::ScannConfigPtr config;
      if (FLAGS_scann_quantizer == "FLAT") {
        config = scann::ScannTreeBruteForceConfig(
            kScannMaxNeighbors, dim, distance, tree_opts);
      } else {
        // Default to SQ8 (Tree-AH).
        config = scann::ScannTreeAhConfig(
            kScannMaxNeighbors, dim, distance, tree_opts);
      }

      RETURN_NOT_OK(scann_.Rebuild(config, kTreeAhTrainingThreads));
    }

    RETURN_NOT_OK(scann_.Serialize(path));
    return nullptr;
  }

  Status DoLoadFromFile(const std::string& path, size_t) {
    // ScannWrapper::LoadFromDisk restores the ScaNN index and the label map
    // from the artifacts directory written by Serialize().
    RETURN_NOT_OK(scann_.LoadFromDisk(path));
    scann_.SetNumThreads(FLAGS_scann_num_search_threads);
    initialized_ = true;
    capacity_ = std::max(capacity_, scann_.n_points());
    return Status::OK();
  }

  // ---------------------------------------------------------------------------
  // Vector retrieval / iteration
  //
  // Iteration reads directly from the ScaNN index: labels provide VectorId
  // (and ybctid), and GetDatapoint() provides the raw float vector.
  // ---------------------------------------------------------------------------

  Result<Vector> GetVector(VectorId) const override {
    return STATUS(NotSupported, "ScaNN adapter does not support GetVector");
  }

  std::unique_ptr<AbstractIterator<Entry>> BeginImpl() const override {
    return std::make_unique<IteratorImpl>(&scann_, 0);
  }

  std::unique_ptr<AbstractIterator<Entry>> EndImpl() const override {
    return std::make_unique<IteratorImpl>(&scann_, Size());
  }

  std::string IndexStatsStr() const override {
    return Format("ScaNN brute-force index, $0 vectors, $1 dimensions",
                  Size(), options_.dimensions);
  }

 private:
  HNSWOptions options_;
  size_t capacity_ = 0;

  // Protects initialized_ and next_docid_ from concurrent access.
  // ScannWrapper has its own internal lock for ScaNN data structures; this
  // mutex ensures the adapter's local bookkeeping stays in sync.
  mutable std::mutex mutex_;

  // ScaNN is lazily initialized on first DoInsert (or DoLoadFromFile).
  bool initialized_ = false;
  scann::ScannWrapper scann_;

  // Sequential docid counter for ScaNN Insert (vector id is not used as docid).
  uint64_t next_docid_ = 1;
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
