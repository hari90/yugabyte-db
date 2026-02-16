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
// Public-facing ScannWrapper implementation.  This TU includes yb/util/status.h
// (the "real" yb::Status) and does NOT include any ScaNN or absl headers.
// All ScaNN work is delegated to scann_wrapper_impl.cc via the bridge API
// declared in scann_wrapper_impl.h.

#include "scann/scann_wrapper.h"

#include <cmath>
#include <fstream>

#include "yb/util/monotime.h"
#include "yb/util/result.h"
#include "yb/util/status.h"

namespace yb::scann {

// ---------------------------------------------------------------------------
// scann_internal::ImplStatus → yb::Status conversion
// ---------------------------------------------------------------------------

namespace {

Status::Code ImplToYbStatusCode(int code) {
  switch (code) {
    case 3:  // absl::StatusCode::kInvalidArgument
      return Status::kInvalidArgument;
    case 5:  // absl::StatusCode::kNotFound
      return Status::kNotFound;
    case 13: // absl::StatusCode::kInternal
      return Status::kInternalError;
    case 9:  // absl::StatusCode::kFailedPrecondition
      return Status::kIllegalState;
    case 6:  // absl::StatusCode::kAlreadyExists
      return Status::kAlreadyPresent;
    case 14: // absl::StatusCode::kUnavailable
      return Status::kServiceUnavailable;
    case 12: // absl::StatusCode::kUnimplemented
      return Status::kNotSupported;
    case 10: // absl::StatusCode::kAborted
      return Status::kAborted;
    default:
      return Status::kRuntimeError;
  }
}

class Timer {
 public:
  Timer(const std::string& label) : label_(label), start_(MonoTime::Now()) {}
  ~Timer() {
    auto end = MonoTime::Now();
    YB_LOG_EVERY_N_SECS(INFO, 10) << "scann: " << label_ << ": "
                                  << (end - start_).ToSeconds() * 1000 << "ms";
  }

 private:
  std::string label_;
  MonoTime start_;
};

// Compute the L2 norm of a single vector.
float ComputeL2Norm(const float* v, size_t dim) {
  float norm_sq = 0.0f;
  for (size_t d = 0; d < dim; ++d) {
    norm_sq += v[d] * v[d];
  }
  return (norm_sq > 0.0f) ? std::sqrt(norm_sq) : 0.0f;
}

// L2-normalise every vector in a flat row-major dataset in-place.
// Each vector of `dim` floats is divided by its L2 norm.
// Zero vectors are left unchanged.
// If `norms_out` is non-null, it is resized to `n` and filled with the
// original L2 norms (before normalisation).
void L2NormalizeDataset(float* data, uint32_t n, uint32_t dim,
                        std::vector<float>* norms_out = nullptr) {
  if (norms_out) {
    norms_out->resize(n);
  }
  for (uint32_t i = 0; i < n; ++i) {
    float* v = data + static_cast<size_t>(i) * dim;
    float norm = ComputeL2Norm(v, dim);
    if (norms_out) {
      (*norms_out)[i] = norm;
    }
    if (norm > 0.0f) {
      float inv_norm = 1.0f / norm;
      for (uint32_t d = 0; d < dim; ++d) {
        v[d] *= inv_norm;
      }
    }
  }
}

static const char* kNormsFileName = "scann_norms.bin";

}  // namespace

#define RETURN_IMPL_STATUS_NOT_OK(impl_status) \
  do { \
    const auto& impl_s_ = (impl_status); \
    if (!impl_s_.ok()) { \
      return ::yb::Status(ImplToYbStatusCode(impl_s_.code), __FILE__, __LINE__, impl_s_.message); \
    } \
  } while (0)

// ---------------------------------------------------------------------------
// ScannWrapper public API
// ---------------------------------------------------------------------------

ScannWrapper::ScannWrapper() : impl_(scann_internal::CreateScannImpl()) {}
ScannWrapper::~ScannWrapper() = default;
ScannWrapper::ScannWrapper(ScannWrapper&&) noexcept = default;
ScannWrapper& ScannWrapper::operator=(ScannWrapper&&) noexcept = default;

bool ScannWrapper::NeedsNormalization() const {
  return normalized_;
}

// -- Index construction -------------------------------------------------------

Status ScannWrapper::Initialize(const std::vector<float>& dataset,
                                uint32_t n_points,
                                const scann_internal::ScannConfigPtr& config,
                                int training_threads,
                                const std::vector<Slice>& labels) {
  if (!labels.empty()) {
    SCHECK_EQ(labels.size(), n_points, InvalidArgument, "labels size must equal n_points");
  }

  const auto distance_measure = scann_internal::ImplGetDistanceMeasure(*config);

  // ScaNN normalises vectors during Initialize only for CosineDistance.
  // Compute and store original L2 norms so GetDatapoint() can denormalise.
  if (scann_internal::ScannNormalizesVectors(distance_measure) && n_points > 0) {
    const auto dim = dataset.size() / n_points;
    norms_.resize(n_points);
    for (uint32_t i = 0; i < n_points; ++i) {
      norms_[i] = ComputeL2Norm(dataset.data() + static_cast<size_t>(i) * dim, dim);
    }
    normalized_ = true;
  }

  Timer timer("Initialize");
  RETURN_IMPL_STATUS_NOT_OK(
      scann_internal::ImplInitialize(
          impl_.get(), dataset.data(), dataset.size(), n_points, *config, training_threads));

  labels_.Reset(labels);
  return Status();
}

Status ScannWrapper::LoadFromDisk(const std::string& artifacts_dir,
                                  const std::string& scann_assets_pbtxt) {
  Timer timer("LoadFromDisk");
  RETURN_IMPL_STATUS_NOT_OK(
      scann_internal::ImplLoadFromDisk(
          impl_.get(), artifacts_dir, scann_assets_pbtxt));

  RETURN_NOT_OK(labels_.Load(artifacts_dir));

  // Load norms from disk (if they exist).
  {
    std::string path = artifacts_dir + "/" + kNormsFileName;
    std::ifstream in(path, std::ios::binary);
    if (in) {
      uint32_t count = 0;
      in.read(reinterpret_cast<char*>(&count), sizeof(count));
      if (in) {
        norms_.resize(count);
        if (count > 0) {
          in.read(reinterpret_cast<char*>(norms_.data()),
                  static_cast<std::streamsize>(count * sizeof(float)));
        }
        if (!in) {
          norms_.clear();
          return Status(Status::kInternalError, __FILE__, __LINE__,
                        "Truncated norms data in " + path);
        }
      }
    } else {
      norms_.clear();
    }
  }
  normalized_ = !norms_.empty();
  return Status();
}

// -- Mutation -----------------------------------------------------------------

Result<int32_t> ScannWrapper::Insert(const std::vector<float>& datapoint,
                                     const std::string& docid,
                                     Slice label) {
  Timer timer("Insert");

  // ScaNN's mutator auto-normalises inserted vectors when the dataset has
  // the UNITL2NORM tag (set during Initialize for CosineDistance).
  // Store the original norm so GetDatapoint() can denormalise.
  float norm = 0.0f;
  if (normalized_) {
    norm = ComputeL2Norm(datapoint.data(), datapoint.size());
  }

  int32_t assigned_index;
  RETURN_IMPL_STATUS_NOT_OK(
      scann_internal::ImplInsert(
          impl_.get(), datapoint.data(), datapoint.size(), docid, &assigned_index));

  labels_.Put(assigned_index, label);

  if (normalized_) {
    auto idx = static_cast<size_t>(assigned_index);
    if (idx >= norms_.size()) {
      norms_.resize(idx + 1, 0.0f);
    }
    norms_[idx] = norm;
  }

  // RETURN_NOT_OK(RunMaintenance());
  return assigned_index;
}

Status ScannWrapper::Delete(const std::string& docid) {
  // NOTE: Delete-by-docid cannot efficiently clean up the label map because
  // the mapping from docid to index is internal to ScaNN.  The label entry
  // will become stale.  Use Delete(int32_t index) to ensure label cleanup.
  Timer timer("Delete(const std::string&)");
  RETURN_IMPL_STATUS_NOT_OK(scann_internal::ImplDelete(impl_.get(), docid));
  return RunMaintenance();
}

Status ScannWrapper::Delete(int32_t index) {
  Timer timer("Delete(int32_t)");
  RETURN_IMPL_STATUS_NOT_OK(scann_internal::ImplDelete(impl_.get(), index));
  labels_.Erase(index);
  return RunMaintenance();
}

Status ScannWrapper::RunMaintenance() {
  bool retrain_performed = false;

  {
    Timer timer("RunMaintenance");
    RETURN_IMPL_STATUS_NOT_OK(scann_internal::ImplRunMaintenance(impl_.get(), &retrain_performed));
    if (retrain_performed) {
      LOG(INFO) << "scann: RunMaintenance: retrain performed";
    }
  }

  return Status::OK();
}

Status ScannWrapper::Retrain() {
  Timer timer("Retrain");
  RETURN_IMPL_STATUS_NOT_OK(scann_internal::ImplRetrainAndReindex(impl_.get()));
  return Status::OK();
}

// -- Search -------------------------------------------------------------------

Result<std::vector<ScannSearchResult>> ScannWrapper::Search(
    const std::vector<float>& query,
    int final_nn, int pre_reorder_nn, int leaves) const {
  Timer timer("Search");
  // Route through the batched path (batch_size=1) because ScaNN's single-query
  // FindNeighbors crashes for some searcher types (e.g. plain AH without
  // reordering) while FindNeighborsBatched works universally.
  std::vector<std::vector<scann_internal::ImplSearchResult>> batch_results;
  RETURN_IMPL_STATUS_NOT_OK(
      scann_internal::ImplSearchBatched(
          impl_.get(), query.data(), query.size(), /*num_queries=*/1, final_nn, pre_reorder_nn,
          leaves, &batch_results));
  if (batch_results.empty()) {
    return std::vector<ScannSearchResult>{};
  }
  return labels_.ResolveLabels(batch_results[0]);
}

Result<std::vector<std::vector<ScannSearchResult>>> ScannWrapper::SearchBatched(
    const std::vector<float>& queries, size_t num_queries,
    int final_nn, int pre_reorder_nn, int leaves) const {
  Timer timer("SearchBatched");
  std::vector<std::vector<scann_internal::ImplSearchResult>> impl_results;
  RETURN_IMPL_STATUS_NOT_OK(
      scann_internal::ImplSearchBatched(
          impl_.get(), queries.data(), queries.size(), num_queries, final_nn, pre_reorder_nn,
          leaves, &impl_results));

  std::vector<std::vector<ScannSearchResult>> results;
  results.reserve(impl_results.size());
  for (const auto& batch : impl_results) {
    results.push_back(labels_.ResolveLabels(batch));
  }
  return results;
}

Result<std::vector<std::vector<ScannSearchResult>>>
ScannWrapper::SearchBatchedParallel(
    const std::vector<float>& queries, size_t num_queries,
    int final_nn, int pre_reorder_nn, int leaves, int batch_size) const {
  Timer timer("SearchBatchedParallel");
  std::vector<std::vector<scann_internal::ImplSearchResult>> impl_results;
  RETURN_IMPL_STATUS_NOT_OK(
      scann_internal::ImplSearchBatchedParallel(
          impl_.get(), queries.data(), queries.size(), num_queries, final_nn, pre_reorder_nn,
          leaves, batch_size, &impl_results));

  std::vector<std::vector<ScannSearchResult>> results;
  results.reserve(impl_results.size());
  for (const auto& batch : impl_results) {
    results.push_back(labels_.ResolveLabels(batch));
  }
  return results;
}

// -- Persistence --------------------------------------------------------------

Status ScannWrapper::Serialize(const std::string& path) {
  Timer timer("Serialize");
  RETURN_IMPL_STATUS_NOT_OK(scann_internal::ImplSerialize(impl_.get(), path));

  RETURN_NOT_OK(labels_.Serialize(path));

  // Write norms alongside the other artifacts.
  if (!norms_.empty()) {
    std::string norms_path = path + "/" + kNormsFileName;
    std::ofstream out(norms_path, std::ios::binary);
    if (!out) {
      return Status(Status::kInternalError, __FILE__, __LINE__,
                    "Failed to open " + norms_path + " for writing");
    }
    auto count = static_cast<uint32_t>(norms_.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(norms_.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
    if (!out) {
      return Status(Status::kInternalError, __FILE__, __LINE__,
                    "Failed writing norms data to " + norms_path);
    }
  }

  return Status();
}

Status ScannWrapper::Rebuild(
    const scann_internal::ScannConfigPtr& config, int training_threads) {
  Timer timer("Rebuild");
  const auto n = static_cast<uint32_t>(n_points());
  const auto dim = dimensionality();
  const bool normalize = scann_internal::NeedsNormalization(
      scann_internal::ImplGetDistanceMeasure(*config));

  LOG(INFO) << "scann: Rebuild [vectors: " << n << ", dimensions: " << dim
            << ", normalize: " << (normalize ? "true" : "false") << "]";

  // Collect all vectors into a flat row-major dataset.
  std::vector<float> dataset;
  dataset.reserve(static_cast<size_t>(n) * dim);
  for (uint32_t i = 0; i < n; ++i) {
    std::vector<float> vec;
    RETURN_IMPL_STATUS_NOT_OK(
        scann_internal::ImplGetDatapoint(impl_.get(), static_cast<int32_t>(i), &vec));
    dataset.insert(dataset.end(), vec.begin(), vec.end());
  }

  if (normalize) {
    if (norms_.empty()) {
      // Norms not yet stored (e.g. DotProductDistance where ScaNN did not
      // normalise on initial Initialize).  Compute and store norms from the
      // current (un-normalised) vectors before normalising in-place.
      L2NormalizeDataset(dataset.data(), n, dim, &norms_);
    } else {
      // Norms already stored (e.g. CosineDistance where ScaNN normalised
      // during Initialize).  The vectors read from the index are already
      // normalised, so a second normalisation is a no-op — but we call it
      // for consistency.
      L2NormalizeDataset(dataset.data(), n, dim);
    }
  }

  // Re-initialize this wrapper's impl_ with the new config.
  // Labels and norms are already correct and don't need updating.
  auto new_impl = scann_internal::CreateScannImpl();
  RETURN_IMPL_STATUS_NOT_OK(
      scann_internal::ImplInitialize(
          new_impl.get(), dataset.data(), dataset.size(), n, *config, training_threads));
  impl_.swap(new_impl);
  if (normalize) {
    normalized_ = true;
  }
  return Status::OK();
}

// -- Configuration ------------------------------------------------------------

Result<ScannWrapper::Datapoint> ScannWrapper::GetDatapoint(int32_t index) const {
  Datapoint dp;
  RETURN_IMPL_STATUS_NOT_OK(scann_internal::ImplGetDatapoint(impl_.get(), index, &dp.vector));
  auto label_slice = labels_.Get(index);
  dp.label.assign(label_slice.cdata(), label_slice.size());

  // Denormalize: if norms are stored, multiply the (unit) vector by the
  // original L2 norm to recover the pre-normalisation vector.
  if (!norms_.empty()) {
    auto idx = static_cast<size_t>(index);
    if (idx < norms_.size() && norms_[idx] > 0.0f) {
      float norm = norms_[idx];
      for (auto& v : dp.vector) {
        v *= norm;
      }
    }
  }

  return dp;
}

void ScannWrapper::SetNumThreads(int num_threads) {
  scann_internal::ImplSetNumThreads(impl_.get(), num_threads);
}

size_t ScannWrapper::n_points() const {
  return scann_internal::ImplNPoints(impl_.get());
}

size_t ScannWrapper::dimensionality() const {
  return scann_internal::ImplDimensionality(impl_.get());
}

// ---------------------------------------------------------------------------
// ScaNN configuration builders – delegate to proto-based implementations
// in scann_wrapper_impl.cc (the only TU that can include proto headers).
// ---------------------------------------------------------------------------

scann_internal::ScannConfigPtr ScannAhConfig(
    int num_neighbors, int dim, const std::string& distance_measure) {
  return scann_internal::ImplAhConfig(num_neighbors, dim, distance_measure);
}

scann_internal::ScannConfigPtr ScannTreeAhConfig(
    int num_neighbors, int dim, const std::string& distance_measure,
    const ScannTreeOptions& tree_opts) {
  return scann_internal::ImplTreeAhConfig(
      num_neighbors, dim, distance_measure,
      tree_opts.num_leaves, tree_opts.max_num_levels, tree_opts.enable_pca);
}

scann_internal::ScannConfigPtr ScannTreeBruteForceConfig(
    int num_neighbors, int dim, const std::string& distance_measure,
    const ScannTreeOptions& tree_opts) {
  return scann_internal::ImplTreeBruteForceConfig(
      num_neighbors, dim, distance_measure,
      tree_opts.num_leaves, tree_opts.max_num_levels);
}

scann_internal::ScannConfigPtr ScannBruteForceConfig(
    int num_neighbors, int dim, bool fixed_point,
    const std::string& distance_measure) {
  return scann_internal::ImplBruteForceConfig(num_neighbors, dim, fixed_point, distance_measure);
}

scann_internal::ScannConfigPtr ScannReorderConfig(
    int num_neighbors, int dim, const std::string& distance_measure) {
  return scann_internal::ImplReorderConfig(num_neighbors, dim, distance_measure);
}

std::string ScannConfigToString(const scann_internal::ScannConfigPtr& config) {
  return scann_internal::ImplConfigToString(*config);
}

}  // namespace yb::scann
