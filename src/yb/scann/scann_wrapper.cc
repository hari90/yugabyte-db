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

#include "yb/util/monotime.h"
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

class Timer {
 public:
  Timer(const std::string& label) : label_(label), start_(MonoTime::Now()) {}
  ~Timer() {
    auto end = MonoTime::Now();
    LOG(INFO) << "scann: " << label_ << ": " << (end - start_).ToSeconds() * 1000 << "ms";
  }

 private:
  std::string label_;
  MonoTime start_;
};

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
                                const std::vector<Slice>& labels) {
  if (!labels.empty()) {
    SCHECK_EQ(labels.size(), n_points, InvalidArgument, "labels size must equal n_points");
  }

  Timer timer("Initialize");
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
  Timer timer("LoadFromDisk");
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
                                     Slice label) {
  Timer timer("Insert");
  int32_t assigned_index;
  auto impl_status = scann_internal::ImplInsert(
      impl_.get(), datapoint.data(), datapoint.size(), docid, &assigned_index);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

  labels_.Put(assigned_index, label);

  RETURN_NOT_OK(RunMaintenance());
  return assigned_index;
}

Status ScannWrapper::Delete(const std::string& docid) {
  // NOTE: Delete-by-docid cannot efficiently clean up the label map because
  // the mapping from docid to index is internal to ScaNN.  The label entry
  // will become stale.  Use Delete(int32_t index) to ensure label cleanup.
  Timer timer("Delete(const std::string&)");
  RETURN_NOT_OK(ImplToYbStatus(scann_internal::ImplDelete(impl_.get(), docid)));
  return RunMaintenance();
}

Status ScannWrapper::Delete(int32_t index) {
  Timer timer("Delete(int32_t)");
  RETURN_NOT_OK(ImplToYbStatus(scann_internal::ImplDelete(impl_.get(), index)));
  labels_.Erase(index);
  return RunMaintenance();
}

Status ScannWrapper::RunMaintenance() {
  bool retrain_performed = false;

  {
    Timer timer("RunMaintenance");
    RETURN_NOT_OK(
        ImplToYbStatus(scann_internal::ImplRunMaintenance(impl_.get(), &retrain_performed)));
    if (retrain_performed) {
      LOG(INFO) << "scann: RunMaintenance: retrain performed";
    }
  }

  return Status::OK();
}

Status ScannWrapper::Retrain() {
  Timer timer("Retrain");
  return ImplToYbStatus(scann_internal::ImplRetrainAndReindex(impl_.get()));
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
  Timer timer("SearchBatched");
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
  Timer timer("SearchBatchedParallel");
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
  Timer timer("Serialize");
  auto impl_status = scann_internal::ImplSerialize(impl_.get(), path);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }

  return labels_.Serialize(path);
}

// -- Configuration ------------------------------------------------------------

Slice ScannWrapper::GetLabel(int32_t index) const {
  return labels_.Get(index);
}

Result<std::vector<float>> ScannWrapper::GetDatapoint(int32_t index) const {
  std::vector<float> result;
  auto impl_status = scann_internal::ImplGetDatapoint(impl_.get(), index, &result);
  if (!impl_status.ok()) {
    return ImplToYbStatus(impl_status);
  }
  return result;
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
    int num_neighbors, int dim, const std::string& distance_measure) {
  return scann_internal::ImplTreeAhConfig(num_neighbors, dim, distance_measure);
}

scann_internal::ScannConfigPtr ScannTreeBruteForceConfig(
    int num_neighbors, int dim, const std::string& distance_measure) {
  return scann_internal::ImplTreeBruteForceConfig(num_neighbors, dim, distance_measure);
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
