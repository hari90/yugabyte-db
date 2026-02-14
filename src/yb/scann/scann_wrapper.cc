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

#include "yb/util/result.h"
#include "yb/util/status.h"

namespace yb::scann {

// ---------------------------------------------------------------------------
// scann_internal::ImplStatus → yb::Status conversion
// ---------------------------------------------------------------------------

namespace {

Status ImplToYbStatus(const scann_internal::ImplStatus& s) {
  if (s.ok()) {
    return Status();
  }

  Status::Code code;
  switch (s.code) {
    case 3:  // absl::StatusCode::kInvalidArgument
      code = Status::kInvalidArgument; break;
    case 5:  // absl::StatusCode::kNotFound
      code = Status::kNotFound; break;
    case 13: // absl::StatusCode::kInternal
      code = Status::kInternalError; break;
    case 9:  // absl::StatusCode::kFailedPrecondition
      code = Status::kIllegalState; break;
    case 6:  // absl::StatusCode::kAlreadyExists
      code = Status::kAlreadyPresent; break;
    case 14: // absl::StatusCode::kUnavailable
      code = Status::kServiceUnavailable; break;
    case 12: // absl::StatusCode::kUnimplemented
      code = Status::kNotSupported; break;
    case 10: // absl::StatusCode::kAborted
      code = Status::kAborted; break;
    default:
      code = Status::kRuntimeError; break;
  }
  return Status(code, __FILE__, __LINE__, s.message);
}

}  // namespace

// ---------------------------------------------------------------------------
// ScannWrapper public API
// ---------------------------------------------------------------------------

ScannWrapper::ScannWrapper() : impl_(scann_internal::CreateScannImpl()) {}
ScannWrapper::~ScannWrapper() = default;
ScannWrapper::ScannWrapper(ScannWrapper&&) noexcept = default;
ScannWrapper& ScannWrapper::operator=(ScannWrapper&&) noexcept = default;

// -- Index construction -------------------------------------------------------

Status ScannWrapper::Initialize(const std::vector<float>& dataset,
                                uint32_t n_points,
                                const scann_internal::ScannConfigPtr& config,
                                int training_threads,
                                const std::vector<ScannVectorId>& labels) {
  if (!labels.empty() && labels.size() != n_points) {
    return Status(Status::kInvalidArgument, __FILE__, __LINE__,
                  "labels size must equal n_points");
  }

  auto impl_status = scann_internal::ImplInitialize(
      impl_.get(), dataset.data(), dataset.size(), n_points, *config,
      training_threads);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

  labels_.Reset(labels);
  return Status();
}

Status ScannWrapper::LoadFromDisk(const std::string& artifacts_dir,
                                  const std::string& scann_assets_pbtxt) {
  auto impl_status = scann_internal::ImplLoadFromDisk(
      impl_.get(), artifacts_dir, scann_assets_pbtxt);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

  return labels_.Load(artifacts_dir);
}

// -- Mutation -----------------------------------------------------------------

Result<int32_t> ScannWrapper::Insert(const std::vector<float>& datapoint,
                                     const std::string& docid,
                                     const ScannVectorId& label) {
  int32_t assigned_index;
  auto impl_status = scann_internal::ImplInsert(
      impl_.get(), datapoint.data(), datapoint.size(), docid, &assigned_index);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

  labels_.Put(assigned_index, label);
  return assigned_index;
}

Status ScannWrapper::Delete(const std::string& docid) {
  // NOTE: Delete-by-docid cannot efficiently clean up the label map because
  // the mapping from docid to index is internal to ScaNN.  The label entry
  // will become stale.  Use Delete(int32_t index) to ensure label cleanup.
  return ImplToYbStatus(scann_internal::ImplDelete(impl_.get(), docid));
}

Status ScannWrapper::Delete(int32_t index) {
  auto status = ImplToYbStatus(scann_internal::ImplDelete(impl_.get(), index));
  if (status.ok()) {
    labels_.Erase(index);
  }
  return status;
}

// -- Search -------------------------------------------------------------------

Result<std::vector<ScannSearchResult>> ScannWrapper::Search(
    const std::vector<float>& query,
    int final_nn, int pre_reorder_nn, int leaves) const {
  // Route through the batched path (batch_size=1) because ScaNN's single-query
  // FindNeighbors crashes for some searcher types (e.g. plain AH without
  // reordering) while FindNeighborsBatched works universally.
  std::vector<std::vector<scann_internal::ImplSearchResult>> batch_results;
  auto impl_status = scann_internal::ImplSearchBatched(
      impl_.get(), query.data(), query.size(), /*num_queries=*/1,
      final_nn, pre_reorder_nn, leaves, &batch_results);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }
  if (batch_results.empty()) {
    return std::vector<ScannSearchResult>{};
  }
  return labels_.ResolveLabels(batch_results[0]);
}

Result<std::vector<std::vector<ScannSearchResult>>> ScannWrapper::SearchBatched(
    const std::vector<float>& queries, size_t num_queries,
    int final_nn, int pre_reorder_nn, int leaves) const {
  std::vector<std::vector<scann_internal::ImplSearchResult>> impl_results;
  auto impl_status = scann_internal::ImplSearchBatched(
      impl_.get(), queries.data(), queries.size(), num_queries,
      final_nn, pre_reorder_nn, leaves, &impl_results);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

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
  std::vector<std::vector<scann_internal::ImplSearchResult>> impl_results;
  auto impl_status = scann_internal::ImplSearchBatchedParallel(
      impl_.get(), queries.data(), queries.size(), num_queries,
      final_nn, pre_reorder_nn, leaves, batch_size, &impl_results);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

  std::vector<std::vector<ScannSearchResult>> results;
  results.reserve(impl_results.size());
  for (const auto& batch : impl_results) {
    results.push_back(labels_.ResolveLabels(batch));
  }
  return results;
}

// -- Persistence --------------------------------------------------------------

Status ScannWrapper::Serialize(const std::string& path) {
  auto impl_status = scann_internal::ImplSerialize(impl_.get(), path);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

  return labels_.Serialize(path);
}

// -- Configuration ------------------------------------------------------------

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

scann_internal::ScannConfigPtr ScannAhConfig(int num_neighbors, int dim) {
  return scann_internal::ImplAhConfig(num_neighbors, dim);
}

scann_internal::ScannConfigPtr ScannTreeAhConfig(int num_neighbors, int dim) {
  return scann_internal::ImplTreeAhConfig(num_neighbors, dim);
}

scann_internal::ScannConfigPtr ScannTreeBruteForceConfig(int num_neighbors, int dim) {
  return scann_internal::ImplTreeBruteForceConfig(num_neighbors, dim);
}

scann_internal::ScannConfigPtr ScannBruteForceConfig(
    int num_neighbors, int dim, bool fixed_point) {
  return scann_internal::ImplBruteForceConfig(num_neighbors, dim, fixed_point);
}

scann_internal::ScannConfigPtr ScannReorderConfig(int num_neighbors, int dim) {
  return scann_internal::ImplReorderConfig(num_neighbors, dim);
}

std::string ScannConfigToString(const scann_internal::ScannConfigPtr& config) {
  return scann_internal::ImplConfigToString(*config);
}

}  // namespace yb::scann
