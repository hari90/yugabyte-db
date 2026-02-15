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

#include <algorithm>
#include <fstream>

#include "absl/base/internal/sysinfo.h"
#include "google/protobuf/text_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/threads.h"
#include "scann/utils/types.h"

namespace yb {

// ---------------------------------------------------------------------------
// Helpers (file-local)
// ---------------------------------------------------------------------------

namespace {

int GetNumCPUs() { return std::max(absl::base_internal::NumCPUs(), 1); }

static const scann_internal::ImplStatus StatusOK = {0, ""};

// Convert an absl::Status to our lightweight ImplStatus POD.
scann_internal::ImplStatus ToImplStatus(const absl::Status& s) {
  if (s.ok()) {
    return StatusOK;
  }
  return {static_cast<int>(s.code()), std::string(s.message())};
}

// Convert ScaNN's NNResultsVector to our internal ImplSearchResult vector.
void ConvertResults(const NNResultsVector& nn_results,
                    std::vector<scann_internal::ImplSearchResult>* out) {
  out->resize(nn_results.size());
  for (size_t i = 0; i < nn_results.size(); ++i) {
    (*out)[i].index = static_cast<int32_t>(nn_results[i].first);
    (*out)[i].distance = nn_results[i].second;
  }
}

// Serialize a ScannConfig proto to its text-format representation.
std::string ConfigToString(const ScannConfig& config) {
  std::string result;
  google::protobuf::TextFormat::PrintToString(config, &result);
  return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Opaque implementation type
// ---------------------------------------------------------------------------

namespace scann_internal {

struct ScannImplOpaque {
  ScannInterface scann;
};

struct ScannConfigOpaque {
  ScannConfig config;
};

// Deleters & factories ------------------------------------------------------

void ScannImplDeleter::operator()(ScannImplOpaque* p) const { delete p; }
void ScannConfigDeleter::operator()(ScannConfigOpaque* p) const { delete p; }

ScannImplPtr CreateScannImpl() {
  return ScannImplPtr(new ScannImplOpaque());
}

// ---------------------------------------------------------------------------
// Free functions that operate on the opaque impl
// ---------------------------------------------------------------------------

ImplStatus ImplInitialize(ScannImplOpaque* impl,
                          const float* dataset, size_t dataset_size,
                          uint32_t n_points,
                          const ScannConfigOpaque& config,
                          int training_threads) {
  if (training_threads < 0) {
    return {3, "training_threads must be non-negative"};  // kInvalidArgument
  }
  if (training_threads == 0) {
    training_threads = GetNumCPUs();
  }

  SingleMachineFactoryOptions opts;
  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);

  // Replicate the dataset construction that the string-based Initialize does.
  DimensionIndex n_dim = kInvalidDimension;
  if (config.config.input_output().pure_dynamic_config().has_dimensionality()) {
    n_dim = config.config.input_output().pure_dynamic_config().dimensionality();
  }

  ConstSpan<float> ds_span(dataset, dataset_size);
  std::shared_ptr<DenseDataset<float>> ds_ptr;
  if (!ds_span.empty() || n_dim != kInvalidDimension) {
    std::vector<float> ds_vec(ds_span.data(), ds_span.data() + ds_span.size());
    auto ds = std::make_unique<DenseDataset<float>>(
        std::move(ds_vec), static_cast<DatapointIndex>(n_points));
    if (n_dim != kInvalidDimension) {
      ds->set_dimensionality(n_dim);
    }

    // If the distance measure requires unit-L2-normalized data (e.g.
    // CosineDistance), normalize the dataset before handing it to ScaNN.
    // This also sets the normalization tag so that subsequent dynamic
    // Append operations auto-normalize new datapoints.
    if (config.config.has_distance_measure()) {
      const auto& dm = config.config.distance_measure().distance_measure();
      if (dm == "CosineDistance") {
        auto status = ds->NormalizeByTag(UNITL2NORM);
        if (!status.ok()) {
          return ToImplStatus(status);
        }
      }
    }

    ds_ptr = std::move(ds);
  }

  return ToImplStatus(impl->scann.Initialize(
      ScannInterface::ScannArtifacts{config.config, std::move(ds_ptr),
                                     std::move(opts)}));
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
                      std::vector<ImplSearchResult>* results) {
  NNResultsVector nn_results;
  auto status = impl->scann.Search(
      MakeDatapointPtr(query, static_cast<DimensionIndex>(query_size)),
      &nn_results, final_nn, pre_reorder_nn, leaves);
  if (!status.ok()) {
    return ToImplStatus(status);
  }
  ConvertResults(nn_results, results);
  return StatusOK;
}

ImplStatus ImplSearchBatched(const ScannImplOpaque* impl,
                             const float* queries, size_t queries_size,
                             size_t num_queries,
                             int final_nn, int pre_reorder_nn, int leaves,
                             std::vector<std::vector<ImplSearchResult>>* results) {
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
  return StatusOK;
}

ImplStatus ImplSearchBatchedParallel(const ScannImplOpaque* impl,
                                     const float* queries, size_t queries_size,
                                     size_t num_queries,
                                     int final_nn, int pre_reorder_nn,
                                     int leaves, int batch_size,
                                     std::vector<std::vector<ImplSearchResult>>* results) {
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
  return StatusOK;
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
  return StatusOK;
}

ImplStatus ImplInsertBatch(
    ScannImplOpaque* impl, const float* dataset, size_t dataset_size,
    size_t n_points, const std::vector<std::string>& docids,
    std::vector<int32_t>* assigned_indices, bool* retrain_needed) {
  using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;

  auto mutator_or = impl->scann.GetMutator();
  if (!mutator_or.ok()) {
    return ToImplStatus(mutator_or.status());
  }
  auto* mutator = mutator_or.value();

  // Use the parallel query pool for precomputation of mutation artifacts.
  auto pool = impl->scann.parallel_query_pool();
  if (pool) {
    mutator->set_mutation_threadpool(pool);
  }

  DimensionIndex dim = ImplDimensionality(impl);

  // Build DenseDataset for batch precomputation.
  DenseDataset<float> ds(
      std::vector<float>(dataset, dataset + dataset_size), n_points);

  // Precompute mutation artifacts in parallel (e.g. tree tokenization).
  // For brute-force configs this returns nullptrs, which is harmless.
  auto precomputed = mutator->ComputePrecomputedMutationArtifacts(ds, pool);

  assigned_indices->resize(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    MutationOptions mo;
    if (i < precomputed.size() && precomputed[i]) {
      mo.precomputed_mutation_artifacts = precomputed[i].get();
    }
    const float* dp = dataset + i * dim;
    auto index_or = mutator->AddDatapoint(
        MakeDatapointPtr(dp, static_cast<DimensionIndex>(dim)),
        docids[i], mo);
    if (!index_or.ok()) {
      return ToImplStatus(index_or.status());
    }
    (*assigned_indices)[i] = static_cast<int32_t>(index_or.value());
  }

  // Run maintenance once for the entire batch instead of per vector.
  auto maint_or = mutator->IncrementalMaintenance();
  if (!maint_or.ok()) {
    return ToImplStatus(maint_or.status());
  }
  *retrain_needed = maint_or.value().has_value();
  if (*retrain_needed) {
    auto status_or = impl->scann.RetrainAndReindex(
        ConfigToString(maint_or.value().value()));
    if (!status_or.ok()) {
      return ToImplStatus(status_or.status());
    }
  }

  return StatusOK;
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

ImplStatus ImplRunMaintenance(ScannImplOpaque* impl, bool* retrain_performed) {
  auto mutator_or = impl->scann.GetMutator();
  if (!mutator_or.ok()) {
    return ToImplStatus(mutator_or.status());
  }
  auto maint_or = mutator_or.value()->IncrementalMaintenance();
  if (!maint_or.ok()) {
    return ToImplStatus(maint_or.status());
  }
  // IncrementalMaintenance returns a ScannConfig when the index has drifted
  // enough that a full RetrainAndReindex is recommended.
  if (maint_or.value().has_value()) {
    *retrain_performed = true;
    auto status_or = impl->scann.RetrainAndReindex(ConfigToString(maint_or.value().value()));
    if (!status_or.ok()) {
      return ToImplStatus(status_or.status());
    }
  }
  return StatusOK;
}

ImplStatus ImplRetrainAndReindex(ScannImplOpaque* impl) {
  auto status_or = impl->scann.RetrainAndReindex(/*config=*/"");
  if (!status_or.ok()) {
    return ToImplStatus(status_or.status());
  }
  return StatusOK;
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
  return StatusOK;
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

// ---------------------------------------------------------------------------
// Config builders (proto → text format)
// ---------------------------------------------------------------------------

namespace {

// Shared helper: configures the AsymmetricHasherConfig block that is common to
// AH, TreeAH, and Reorder configs.
void SetupAh(AsymmetricHasherConfig* ah, int dim, bool use_residual) {
  ah->set_lookup_type(AsymmetricHasherConfig::INT8_LUT16);
  ah->set_use_residual_quantization(use_residual);
  ah->set_use_global_topn(true);
  ah->mutable_quantization_distance()->set_distance_measure("SquaredL2Distance");
  ah->set_num_clusters_per_block(16);

  auto* proj = ah->mutable_projection();
  proj->set_input_dim(dim);
  proj->set_projection_type(ProjectionConfig::CHUNK);
  proj->set_num_blocks(dim / 2);
  proj->set_num_dims_per_block(2);

  ah->mutable_fixed_point_lut_conversion_options()
      ->set_float_to_int_conversion_method(
          AsymmetricHasherConfig::FixedPointLUTConversionOptions::ROUND);
  ah->set_expected_sample_size(100000);
  ah->set_max_clustering_iterations(10);
}

}  // namespace

ScannConfigPtr ImplAhConfig(int num_neighbors, int dim,
                            const std::string& distance_measure) {
  auto cfg = ScannConfigPtr(new ScannConfigOpaque());
  auto& config = cfg->config;
  config.set_num_neighbors(num_neighbors);
  config.mutable_distance_measure()->set_distance_measure(distance_measure);

  SetupAh(config.mutable_hash()->mutable_asymmetric_hash(), dim,
           /*use_residual=*/false);

  config.mutable_input_output()->mutable_pure_dynamic_config()
      ->set_dimensionality(dim);

  return cfg;
}

ScannConfigPtr ImplTreeAhConfig(int num_neighbors, int dim,
                                const std::string& distance_measure) {
  auto cfg = ScannConfigPtr(new ScannConfigOpaque());
  auto& config = cfg->config;
  config.set_num_neighbors(num_neighbors);
  config.mutable_distance_measure()->set_distance_measure(distance_measure);

  auto* part = config.mutable_partitioning();
  part->set_num_children(100);
  part->set_min_cluster_size(20);
  part->set_max_clustering_iterations(12);
  part->set_single_machine_center_initialization(
      PartitioningConfig::RANDOM_INITIALIZATION);
  part->mutable_partitioning_distance()->set_distance_measure("SquaredL2Distance");
  auto* spill = part->mutable_query_spilling();
  spill->set_spilling_type(QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS);
  spill->set_max_spill_centers(20);
  part->set_expected_sample_size(100000);
  part->mutable_query_tokenization_distance_override()
      ->set_distance_measure(distance_measure);
  part->set_partitioning_type(PartitioningConfig::GENERIC);
  part->set_query_tokenization_type(PartitioningConfig::FLOAT);

  SetupAh(config.mutable_hash()->mutable_asymmetric_hash(), dim,
           /*use_residual=*/true);

  config.mutable_input_output()->mutable_pure_dynamic_config()
      ->set_dimensionality(dim);

  return cfg;
}

ScannConfigPtr ImplTreeBruteForceConfig(int num_neighbors, int dim,
                                        const std::string& distance_measure) {
  auto cfg = ScannConfigPtr(new ScannConfigOpaque());
  auto& config = cfg->config;
  config.set_num_neighbors(num_neighbors);
  config.mutable_distance_measure()->set_distance_measure(distance_measure);

  auto* part = config.mutable_partitioning();
  part->set_num_children(100);
  part->set_min_cluster_size(10);
  part->set_max_clustering_iterations(12);
  part->set_single_machine_center_initialization(
      PartitioningConfig::RANDOM_INITIALIZATION);
  part->mutable_partitioning_distance()->set_distance_measure("SquaredL2Distance");
  auto* spill = part->mutable_query_spilling();
  spill->set_spilling_type(QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS);
  spill->set_max_spill_centers(10);
  part->set_expected_sample_size(100000);
  part->mutable_query_tokenization_distance_override()
      ->set_distance_measure(distance_measure);
  part->set_partitioning_type(PartitioningConfig::GENERIC);
  part->set_query_tokenization_type(PartitioningConfig::FLOAT);

  config.mutable_brute_force()->mutable_fixed_point()->set_enabled(true);

  config.mutable_input_output()->mutable_pure_dynamic_config()
      ->set_dimensionality(dim);

  return cfg;
}

ScannConfigPtr ImplBruteForceConfig(int num_neighbors, int dim,
                                    bool fixed_point,
                                    const std::string& distance_measure) {
  auto cfg = ScannConfigPtr(new ScannConfigOpaque());
  auto& config = cfg->config;
  config.set_num_neighbors(num_neighbors);
  config.mutable_distance_measure()->set_distance_measure(distance_measure);

  if (fixed_point) {
    config.mutable_brute_force()->mutable_fixed_point()->set_enabled(true);
  } else {
    config.mutable_brute_force();
  }

  config.mutable_input_output()->mutable_pure_dynamic_config()
      ->set_dimensionality(dim);

  return cfg;
}

ScannConfigPtr ImplReorderConfig(int num_neighbors, int dim,
                                 const std::string& distance_measure) {
  auto cfg = ScannConfigPtr(new ScannConfigOpaque());
  auto& config = cfg->config;
  config.set_num_neighbors(num_neighbors);
  config.mutable_distance_measure()->set_distance_measure(distance_measure);

  SetupAh(config.mutable_hash()->mutable_asymmetric_hash(), dim,
           /*use_residual=*/false);

  auto* reorder = config.mutable_exact_reordering();
  reorder->set_approx_num_neighbors(40);
  reorder->mutable_fixed_point()->set_enabled(false);

  config.mutable_input_output()->mutable_pure_dynamic_config()
      ->set_dimensionality(dim);

  return cfg;
}

std::string ImplConfigToString(const ScannConfigOpaque& config) {
  return ConfigToString(config.config);
}

}  // namespace scann_internal
}  // namespace yb
