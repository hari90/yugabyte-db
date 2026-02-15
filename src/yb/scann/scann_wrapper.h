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

#include "yb/util/slice.h"
#include "yb/util/status_fwd.h"

#include "scann/scann_label_map.h"     // ScannSearchResult, ScannLabelMap
#include "scann/scann_wrapper_impl.h"  // ScannImplPtr, ScannConfigPtr

namespace yb::scann {

// High-level wrapper around ScaNN for building, querying, and persisting
// approximate nearest-neighbor indices.  All ScaNN-specific types are hidden
// behind the PIMPL so that callers only need standard C++ / YB types.
//
// In addition to the vector data managed by ScaNN, the wrapper maintains a
// ScannLabelMap (index → variable-length byte label) that is stored alongside
// the ScaNN artifacts on Serialize and restored on LoadFromDisk.
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

  // Build an index from a flat row-major float dataset and a ScannConfig proto
  // (returned by the ScannXxxConfig builder functions below).
  //
  //   dataset          – row-major float vector of size (n_points * dim).
  //   n_points         – number of vectors in the dataset.
  //   config           – opaque ScannConfig proto built via the Scann*Config
  //                      helpers.
  //   training_threads – number of threads for training/indexing.
  //   labels           – vector of Slice labels, one per datapoint.
  //                      Each Slice can be any length (variable-width).
  //                      Must have size == n_points (or be empty for no labels).
  Status Initialize(const std::vector<float>& dataset, uint32_t n_points,
                    const scann_internal::ScannConfigPtr& config,
                    int training_threads,
                    const std::vector<Slice>& labels = {});

  // Load a previously serialized index from disk.
  //
  //   artifacts_dir       – directory written by Serialize().
  //   scann_assets_pbtxt  – optional path (within artifacts_dir) to a custom
  //                         scann_assets.pbtxt; pass "" for the default.
  Status LoadFromDisk(const std::string& artifacts_dir,
                      const std::string& scann_assets_pbtxt = "");

  // ---------------------------------------------------------------------------
  // Mutation
  // ---------------------------------------------------------------------------

  // Insert a single datapoint into the index.
  //
  //   datapoint – float vector of length dimensionality().
  //   docid     – unique string identifier for the datapoint.
  //   label     – variable-length byte label.
  //
  // Returns the assigned datapoint index on success.
  Result<int32_t> Insert(const std::vector<float>& datapoint,
                         const std::string& docid,
                         Slice label);

  // Delete a datapoint by its string docid.
  Status Delete(const std::string& docid);

  // Delete a datapoint by its numeric index.
  Status Delete(int32_t index);

  // Run incremental maintenance on the index.  Should be called after
  // insert / delete operations to keep tree-partitioned indexes balanced
  // (reassigns datapoints to correct partitions, may split/shrink partitions).
  // For brute-force indexes this is a lightweight no-op check.
  // If maintenance determines a full retrain is needed, Retrain() is called
  // automatically.
  Status RunMaintenance();

  // Fully retrain the model (e.g. partition centers) and rebuild the index
  // from the current dataset.  This is expensive and should only be called
  // when incremental maintenance signals that the index has drifted too far.
  Status Retrain();

  // ---------------------------------------------------------------------------
  // Search
  // ---------------------------------------------------------------------------

  // Single-query ANN search.  `query` must have length == dimensionality().
  // Results include the byte label for each hit.
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

  // Serialize the index to `path`.  Writes all ScaNN artifacts, a
  // scann_assets.pbtxt manifest file, and a scann_labels.bin file that stores
  // the label map.
  Status Serialize(const std::string& path);

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  // Set/change the number of threads for parallel batched search.
  void SetNumThreads(int num_threads);

  // Retrieve the label for a given datapoint index.
  // Returns an empty Slice if the index is out of range.
  Slice GetLabel(int32_t index) const;

  // A datapoint retrieved from the index: the float vector and its label.
  struct Datapoint {
    std::vector<float> vector;
    std::string label;
  };

  // Retrieve the float vector and its label for a given datapoint index.
  Result<Datapoint> GetDatapoint(int32_t index) const;

  // Number of indexed datapoints.
  size_t n_points() const;

  // Dimensionality of vectors.
  size_t dimensionality() const;

 private:
  scann_internal::ScannImplPtr impl_;
  ScannLabelMap labels_;
};

// ---------------------------------------------------------------------------
// ScaNN configuration builders
//
// Each function returns an opaque ScannConfigPtr ready to pass directly to
// ScannWrapper::Initialize().  `num_neighbors` is the number of neighbors
// returned by a search and `dim` is the dimensionality of the vectors.
// ---------------------------------------------------------------------------

// All config builders accept an optional distance_measure string.
// Supported values: "DotProductDistance", "SquaredL2Distance", "CosineDistance".
// The default is "DotProductDistance" for backward compatibility.

// Asymmetric-hashing (AH) scoring.
scann_internal::ScannConfigPtr ScannAhConfig(
    int num_neighbors, int dim,
    const std::string& distance_measure = "DotProductDistance");

// Tree-partitioned + AH scoring.
scann_internal::ScannConfigPtr ScannTreeAhConfig(
    int num_neighbors, int dim,
    const std::string& distance_measure = "DotProductDistance");

// Tree-partitioned + brute-force scoring.
scann_internal::ScannConfigPtr ScannTreeBruteForceConfig(
    int num_neighbors, int dim,
    const std::string& distance_measure = "DotProductDistance");

// Brute-force scoring (no partitioning).
//   fixed_point – when true (default), enables fixed-point quantization for
//                 faster scoring.  Set to false when the index will be mutated
//                 (Insert/Delete) since dynamically added points are not
//                 included in the pre-quantized representation.
scann_internal::ScannConfigPtr ScannBruteForceConfig(
    int num_neighbors, int dim, bool fixed_point = true,
    const std::string& distance_measure = "DotProductDistance");

// AH scoring + exact reordering.
scann_internal::ScannConfigPtr ScannReorderConfig(
    int num_neighbors, int dim,
    const std::string& distance_measure = "DotProductDistance");

// Serialize an opaque config to its text-format representation (useful for
// logging / debugging).
std::string ScannConfigToString(const scann_internal::ScannConfigPtr& config);

}  // namespace yb::scann
