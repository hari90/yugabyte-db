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
// Wrapper around Google ScaNN (Scalable Approximate Nearest Neighbor) library.
// Provides a clean API that hides ScaNN-internal types so that the rest of the
// YB codebase never needs to include ScaNN headers directly.
//
// IMPORTANT: This header intentionally avoids including any absl headers
// because absl/base/internal/dynamic_annotations.h and
// yb/gutil/dynamic_annotations.h define conflicting macros and cannot coexist
// in the same translation unit.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "yb/util/status_fwd.h"

#include "scann/scann_wrapper_impl.h"  // ScannSearchResult, ScannImplPtr

namespace yb::scann {

// High-level wrapper around ScaNN for building, querying, and persisting
// approximate nearest-neighbor indices.  All ScaNN-specific types are hidden
// behind the PIMPL so that callers only need standard C++ / YB types.
class ScannWrapper {
 public:
  ScannWrapper();
  ~ScannWrapper();

  // Move-only (ScaNN's searcher is not copyable).
  ScannWrapper(ScannWrapper&&) noexcept;
  ScannWrapper& operator=(ScannWrapper&&) noexcept;
  ScannWrapper(const ScannWrapper&) = delete;
  ScannWrapper& operator=(const ScannWrapper&) = delete;

  // ---------------------------------------------------------------------------
  // Index construction
  // ---------------------------------------------------------------------------

  // Build an index from a flat row-major float dataset and a text-format
  // ScannConfig proto.
  //
  //   dataset          – row-major float vector of size (n_points * dim).
  //   n_points         – number of vectors in the dataset.
  //   config           – text-format ScannConfig proto string.
  //   training_threads – number of threads for training/indexing.
  Status Initialize(const std::vector<float>& dataset, uint32_t n_points,
                    const std::string& config, int training_threads);

  // Load a previously serialized index from disk.
  //
  //   artifacts_dir       – directory written by Serialize().
  //   scann_assets_pbtxt  – optional path (within artifacts_dir) to a custom
  //                         scann_assets.pbtxt; pass "" for the default.
  Status LoadFromDisk(const std::string& artifacts_dir,
                      const std::string& scann_assets_pbtxt = "");

  // ---------------------------------------------------------------------------
  // Search
  // ---------------------------------------------------------------------------

  // Single-query ANN search.  `query` must have length == dimensionality().
  Result<std::vector<ScannSearchResult>> Search(
      const std::vector<float>& query,
      int final_nn, int pre_reorder_nn, int leaves) const;

  // Batched search (single-threaded over queries).
  //   queries     – flat row-major float vector of size (num_queries * dim).
  //   num_queries – number of query vectors.
  Result<std::vector<std::vector<ScannSearchResult>>> SearchBatched(
      const std::vector<float>& queries, size_t num_queries,
      int final_nn, int pre_reorder_nn, int leaves) const;

  // Batched search using the internal thread pool for parallelism.
  Result<std::vector<std::vector<ScannSearchResult>>> SearchBatchedParallel(
      const std::vector<float>& queries, size_t num_queries,
      int final_nn, int pre_reorder_nn,
      int leaves, int batch_size) const;

  // ---------------------------------------------------------------------------
  // Persistence
  // ---------------------------------------------------------------------------

  // Serialize the index to `path`.  Writes all artifacts and a
  // scann_assets.pbtxt manifest file.
  Status Serialize(const std::string& path);

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  // Set/change the number of threads for parallel batched search.
  void SetNumThreads(int num_threads);

  // Number of indexed datapoints.
  size_t n_points() const;

  // Dimensionality of vectors.
  size_t dimensionality() const;

 private:
  scann_internal::ScannImplPtr impl_;
};

// ---------------------------------------------------------------------------
// ScaNN configuration builders
//
// Each function returns a text-format ScannConfig proto string ready to pass
// to ScannWrapper::Initialize().  `num_neighbors` is the number of neighbors
// returned by a search and `dim` is the dimensionality of the vectors.
// ---------------------------------------------------------------------------

// Asymmetric-hashing (AH) scoring.
std::string ScannAhConfig(int num_neighbors, int dim);

// Tree-partitioned + AH scoring.
std::string ScannTreeAhConfig(int num_neighbors, int dim);

// Tree-partitioned + brute-force scoring.
std::string ScannTreeBruteForceConfig(int num_neighbors, int dim);

// Brute-force scoring (no partitioning).
std::string ScannBruteForceConfig(int num_neighbors, int dim);

// AH scoring + exact reordering.
std::string ScannReorderConfig(int num_neighbors, int dim);

}  // namespace yb
