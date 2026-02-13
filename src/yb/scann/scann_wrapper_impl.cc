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
// ScaNN-facing implementation of the wrapper.  This is the ONLY translation
// unit that includes ScaNN-internal headers.  It must NOT include
// yb/util/status.h (which defines a different yb::Status class that conflicts
// with ScaNN's `using ::absl::Status` alias in namespace yb).

#include "scann/scann_wrapper_impl.h"

#include <fstream>

#include "google/protobuf/text_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace yb {

// ---------------------------------------------------------------------------
// Helpers (file-local)
// ---------------------------------------------------------------------------

namespace {

// Convert an absl::Status to our lightweight ImplStatus POD.
scann_internal::ImplStatus ToImplStatus(const absl::Status& s) {
  if (s.ok()) {
    return {0, ""};
  }
  return {static_cast<int>(s.code()), std::string(s.message())};
}

// Convert ScaNN's NNResultsVector to our public ScannSearchResult vector.
void ConvertResults(const NNResultsVector& nn_results,
                    std::vector<ScannSearchResult>* out) {
  out->resize(nn_results.size());
  for (size_t i = 0; i < nn_results.size(); ++i) {
    (*out)[i].index = static_cast<int32_t>(nn_results[i].first);
    (*out)[i].distance = nn_results[i].second;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Opaque implementation type
// ---------------------------------------------------------------------------

namespace scann_internal {

struct ScannImplOpaque {
  ScannInterface scann;
};

// Deleter & factory --------------------------------------------------------

void ScannImplDeleter::operator()(ScannImplOpaque* p) const { delete p; }

ScannImplPtr CreateScannImpl() {
  return ScannImplPtr(new ScannImplOpaque());
}

// ---------------------------------------------------------------------------
// Free functions that operate on the opaque impl
// ---------------------------------------------------------------------------

ImplStatus ImplInitialize(ScannImplOpaque* impl,
                          const float* dataset, size_t dataset_size,
                          uint32_t n_points,
                          const std::string& config, int training_threads) {
  return ToImplStatus(impl->scann.Initialize(
      ConstSpan<float>(dataset, dataset_size),
      static_cast<DatapointIndex>(n_points), config, training_threads));
}

ImplStatus ImplLoadFromDisk(ScannImplOpaque* impl,
                            const std::string& artifacts_dir,
                            const std::string& scann_assets_pbtxt) {
  auto load_or = ScannInterface::LoadArtifacts(artifacts_dir, scann_assets_pbtxt);
  if (!load_or.ok()) {
    return ToImplStatus(load_or.status());
  }
  return ToImplStatus(impl->scann.Initialize(std::move(load_or).value()));
}

ImplStatus ImplSearch(const ScannImplOpaque* impl,
                      const float* query, size_t query_size,
                      int final_nn, int pre_reorder_nn, int leaves,
                      std::vector<ScannSearchResult>* results) {
  NNResultsVector nn_results;
  auto status = impl->scann.Search(
      MakeDatapointPtr(query, static_cast<DimensionIndex>(query_size)),
      &nn_results, final_nn, pre_reorder_nn, leaves);
  if (!status.ok()) {
    return ToImplStatus(status);
  }
  ConvertResults(nn_results, results);
  return {0, ""};
}

ImplStatus ImplSearchBatched(const ScannImplOpaque* impl,
                             const float* queries, size_t queries_size,
                             size_t num_queries,
                             int final_nn, int pre_reorder_nn, int leaves,
                             std::vector<std::vector<ScannSearchResult>>* results) {
  DenseDataset<float> query_dataset(
      std::vector<float>(queries, queries + queries_size), num_queries);

  std::vector<NNResultsVector> nn_results(num_queries);
  auto status = impl->scann.SearchBatched(
      query_dataset, MakeMutableSpan(nn_results), final_nn, pre_reorder_nn,
      leaves);
  if (!status.ok()) {
    return ToImplStatus(status);
  }

  results->resize(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    ConvertResults(nn_results[i], &(*results)[i]);
  }
  return {0, ""};
}

ImplStatus ImplSearchBatchedParallel(const ScannImplOpaque* impl,
                                     const float* queries, size_t queries_size,
                                     size_t num_queries,
                                     int final_nn, int pre_reorder_nn,
                                     int leaves, int batch_size,
                                     std::vector<std::vector<ScannSearchResult>>* results) {
  DenseDataset<float> query_dataset(
      std::vector<float>(queries, queries + queries_size), num_queries);

  std::vector<NNResultsVector> nn_results(num_queries);
  auto status = impl->scann.SearchBatchedParallel(
      query_dataset, MakeMutableSpan(nn_results), final_nn, pre_reorder_nn,
      leaves, batch_size);
  if (!status.ok()) {
    return ToImplStatus(status);
  }

  results->resize(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    ConvertResults(nn_results[i], &(*results)[i]);
  }
  return {0, ""};
}

ImplStatus ImplInsert(
    ScannImplOpaque* impl, const float* datapoint, size_t datapoint_size, const std::string& docid,
    int32_t* assigned_index) {
  auto mutator_or = impl->scann.GetMutator();
  if (!mutator_or.ok()) {
    return ToImplStatus(mutator_or.status());
  }
  auto* mutator = mutator_or.value();

  auto index_or = mutator->AddDatapoint(
      MakeDatapointPtr(datapoint, static_cast<DimensionIndex>(datapoint_size)), docid);
  if (!index_or.ok()) {
    return ToImplStatus(index_or.status());
  }
  *assigned_index = static_cast<int32_t>(index_or.value());
  return {0, ""};
}

ImplStatus ImplDelete(ScannImplOpaque* impl, const std::string& docid) {
  auto mutator_or = impl->scann.GetMutator();
  if (!mutator_or.ok()) {
    return ToImplStatus(mutator_or.status());
  }
  return ToImplStatus(mutator_or.value()->RemoveDatapoint(docid));
}

ImplStatus ImplDelete(ScannImplOpaque* impl, int32_t index) {
  auto mutator_or = impl->scann.GetMutator();
  if (!mutator_or.ok()) {
    return ToImplStatus(mutator_or.status());
  }
  return ToImplStatus(mutator_or.value()->RemoveDatapoint(static_cast<DatapointIndex>(index)));
}

ImplStatus ImplSerialize(ScannImplOpaque* impl, const std::string& path) {
  auto assets_or = impl->scann.Serialize(path, /*relative_path=*/false);
  if (!assets_or.ok()) {
    return ToImplStatus(assets_or.status());
  }

  // Write the manifest proto so that LoadFromDisk can find it.
  std::string assets_pbtxt;
  if (!google::protobuf::TextFormat::PrintToString(assets_or.value(),
                                                   &assets_pbtxt)) {
    return {13, "Failed to serialize ScannAssets to text"};  // kInternal
  }
  std::ofstream out(path + "/scann_assets.pbtxt");
  if (!out) {
    return {13, "Failed to open scann_assets.pbtxt for writing"};  // kInternal
  }
  out << assets_pbtxt;
  return {0, ""};
}

void ImplSetNumThreads(ScannImplOpaque* impl, int num_threads) {
  impl->scann.SetNumThreads(num_threads);
}

size_t ImplNPoints(const ScannImplOpaque* impl) {
  return impl->scann.n_points();
}

size_t ImplDimensionality(const ScannImplOpaque* impl) {
  return static_cast<size_t>(impl->scann.dimensionality());
}

}  // namespace scann_internal
}  // namespace yb
