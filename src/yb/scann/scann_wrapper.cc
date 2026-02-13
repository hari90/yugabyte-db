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
  if (s.ok()) return Status();

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
                                const std::string& config,
                                int training_threads) {
  return ImplToYbStatus(
      scann_internal::ImplInitialize(impl_.get(), dataset.data(), dataset.size(),
                                     n_points, config, training_threads));
}

Status ScannWrapper::LoadFromDisk(const std::string& artifacts_dir,
                                  const std::string& scann_assets_pbtxt) {
  return ImplToYbStatus(
      scann_internal::ImplLoadFromDisk(impl_.get(), artifacts_dir,
                                       scann_assets_pbtxt));
}

// -- Search -------------------------------------------------------------------

Result<std::vector<ScannSearchResult>> ScannWrapper::Search(
    const std::vector<float>& query,
    int final_nn, int pre_reorder_nn, int leaves) const {
  std::vector<ScannSearchResult> results;
  auto impl_status = scann_internal::ImplSearch(
      impl_.get(), query.data(), query.size(),
      final_nn, pre_reorder_nn, leaves, &results);
  if (!impl_status.ok()) return ImplToYbStatus(impl_status);
  return results;
}

Result<std::vector<std::vector<ScannSearchResult>>> ScannWrapper::SearchBatched(
    const std::vector<float>& queries, size_t num_queries,
    int final_nn, int pre_reorder_nn, int leaves) const {
  std::vector<std::vector<ScannSearchResult>> results;
  auto impl_status = scann_internal::ImplSearchBatched(
      impl_.get(), queries.data(), queries.size(), num_queries,
      final_nn, pre_reorder_nn, leaves, &results);
  if (!impl_status.ok()) return ImplToYbStatus(impl_status);
  return results;
}

Result<std::vector<std::vector<ScannSearchResult>>>
ScannWrapper::SearchBatchedParallel(
    const std::vector<float>& queries, size_t num_queries,
    int final_nn, int pre_reorder_nn, int leaves, int batch_size) const {
  std::vector<std::vector<ScannSearchResult>> results;
  auto impl_status = scann_internal::ImplSearchBatchedParallel(
      impl_.get(), queries.data(), queries.size(), num_queries,
      final_nn, pre_reorder_nn, leaves, batch_size, &results);
  if (!impl_status.ok()) return ImplToYbStatus(impl_status);
  return results;
}

// -- Persistence --------------------------------------------------------------

Status ScannWrapper::Serialize(const std::string& path) {
  return ImplToYbStatus(scann_internal::ImplSerialize(impl_.get(), path));
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
// ScaNN configuration builders
// ---------------------------------------------------------------------------

std::string ScannConfigWithDimensionality(const std::string& base_config, int dim) {
  return base_config + R"(
    input_output {
      pure_dynamic_config {
        dimensionality: )" +
         std::to_string(dim) + R"(
      }
    }
  )";
}

std::string ScannAhConfig(int num_neighbors, int dim) {
  return ScannConfigWithDimensionality(
      R"(
        num_neighbors: )" +
          std::to_string(num_neighbors) + R"(
        distance_measure { distance_measure: "DotProductDistance" }
        hash {
          asymmetric_hash {
            lookup_type: INT8_LUT16
            use_residual_quantization: false
            use_global_topn: true
            quantization_distance { distance_measure: "SquaredL2Distance" }
            num_clusters_per_block: 16
            projection {
              input_dim: )" +
          std::to_string(dim) + R"(
              projection_type: CHUNK
              num_blocks: )" +
          std::to_string(dim / 2) + R"(
              num_dims_per_block: 2
            }
            fixed_point_lut_conversion_options {
              float_to_int_conversion_method: ROUND
            }
            expected_sample_size: 100000
            max_clustering_iterations: 10
          }
        }
      )",
      dim);
}

std::string ScannTreeAhConfig(int num_neighbors, int dim) {
  return ScannConfigWithDimensionality(
      R"(
        num_neighbors: )" +
          std::to_string(num_neighbors) + R"(
        distance_measure { distance_measure: "DotProductDistance" }
        partitioning {
          num_children: 100
          min_cluster_size: 20
          max_clustering_iterations: 12
          single_machine_center_initialization: RANDOM_INITIALIZATION
          partitioning_distance { distance_measure: "SquaredL2Distance" }
          query_spilling {
            spilling_type: FIXED_NUMBER_OF_CENTERS
            max_spill_centers: 20
          }
          expected_sample_size: 100000
          query_tokenization_distance_override { distance_measure: "DotProductDistance" }
          partitioning_type: GENERIC
          query_tokenization_type: FLOAT
        }
        hash {
          asymmetric_hash {
            lookup_type: INT8_LUT16
            use_residual_quantization: true
            use_global_topn: true
            quantization_distance { distance_measure: "SquaredL2Distance" }
            num_clusters_per_block: 16
            projection {
              input_dim: )" +
          std::to_string(dim) + R"(
              projection_type: CHUNK
              num_blocks: )" +
          std::to_string(dim / 2) + R"(
              num_dims_per_block: 2
            }
            fixed_point_lut_conversion_options {
              float_to_int_conversion_method: ROUND
            }
            expected_sample_size: 100000
            max_clustering_iterations: 10
          }
        }
      )",
      dim);
}

std::string ScannTreeBruteForceConfig(int num_neighbors, int dim) {
  return ScannConfigWithDimensionality(
      R"(
        num_neighbors: )" +
          std::to_string(num_neighbors) + R"(
        distance_measure { distance_measure: "DotProductDistance" }
        partitioning {
          num_children: 100
          min_cluster_size: 10
          max_clustering_iterations: 12
          single_machine_center_initialization: RANDOM_INITIALIZATION
          partitioning_distance { distance_measure: "SquaredL2Distance" }
          query_spilling {
            spilling_type: FIXED_NUMBER_OF_CENTERS
            max_spill_centers: 10
          }
          expected_sample_size: 100000
          query_tokenization_distance_override { distance_measure: "DotProductDistance" }
          partitioning_type: GENERIC
          query_tokenization_type: FLOAT
        }
        brute_force {
          fixed_point { enabled: true }
        }
      )",
      dim);
}

std::string ScannBruteForceConfig(int num_neighbors, int dim) {
  return ScannConfigWithDimensionality(
      R"(
        num_neighbors: )" +
          std::to_string(num_neighbors) + R"(
        distance_measure { distance_measure: "DotProductDistance" }
        brute_force {
          fixed_point { enabled: true }
        }
      )",
      dim);
}

std::string ScannReorderConfig(int num_neighbors, int dim) {
  return ScannConfigWithDimensionality(
      R"(
        num_neighbors: )" +
          std::to_string(num_neighbors) + R"(
        distance_measure { distance_measure: "DotProductDistance" }
        hash {
          asymmetric_hash {
            lookup_type: INT8_LUT16
            use_residual_quantization: false
            use_global_topn: true
            quantization_distance { distance_measure: "SquaredL2Distance" }
            num_clusters_per_block: 16
            projection {
              input_dim: )" +
          std::to_string(dim) + R"(
              projection_type: CHUNK
              num_blocks: )" +
          std::to_string(dim / 2) + R"(
              num_dims_per_block: 2
            }
            fixed_point_lut_conversion_options {
              float_to_int_conversion_method: ROUND
            }
            expected_sample_size: 100000
            max_clustering_iterations: 10
          }
        }
        exact_reordering {
          approx_num_neighbors: 40
          fixed_point { enabled: false }
        }
      )",
      dim);
}

}  // namespace yb
