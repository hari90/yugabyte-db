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
// Internal bridge between the public ScannWrapper (which uses yb::Status) and
// the ScaNN library (which aliases absl::Status as Status in namespace yb).
//
// This header deliberately includes NEITHER yb/util/status*.h NOR any absl
// headers, so it can be safely included from translation units that use either
// world without triggering macro conflicts between
// yb/gutil/dynamic_annotations.h and absl/base/internal/dynamic_annotations.h.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace yb {

// Result of a nearest-neighbor search: a (datapoint_index, distance) pair.
struct ScannSearchResult {
  int32_t index;
  float distance;
};

namespace scann_internal {

// Lightweight status POD that bridges absl::Status (ScaNN side) and
// yb::Status (YB side) without requiring either header.
struct ImplStatus {
  int code;  // 0 = OK, otherwise absl::StatusCode cast to int.
  std::string message;
  bool ok() const { return code == 0; }
};

// Opaque handle to the ScaNN implementation.  The real definition lives in
// scann_wrapper_impl.cc (the only TU that includes ScaNN headers).
struct ScannImplOpaque;

// Custom deleter so that unique_ptr can destroy the opaque type.
struct ScannImplDeleter {
  void operator()(ScannImplOpaque* p) const;
};

using ScannImplPtr = std::unique_ptr<ScannImplOpaque, ScannImplDeleter>;

// Factory.
ScannImplPtr CreateScannImpl();

// Opaque handle to a ScannConfig proto.  Hides the protobuf type from
// translation units that cannot include ScaNN/absl headers.
struct ScannConfigOpaque;

struct ScannConfigDeleter {
  void operator()(ScannConfigOpaque* p) const;
};

using ScannConfigPtr = std::unique_ptr<ScannConfigOpaque, ScannConfigDeleter>;

// ---------------------------------------------------------------------------
// Functions that mirror the ScannWrapper public API but return ImplStatus.
// ---------------------------------------------------------------------------

ImplStatus ImplInitialize(ScannImplOpaque* impl,
                          const float* dataset, size_t dataset_size,
                          uint32_t n_points,
                          const ScannConfigOpaque& config,
                          int training_threads);

ImplStatus ImplLoadFromDisk(ScannImplOpaque* impl,
                            const std::string& artifacts_dir,
                            const std::string& scann_assets_pbtxt);

ImplStatus ImplSearch(const ScannImplOpaque* impl,
                      const float* query, size_t query_size,
                      int final_nn, int pre_reorder_nn, int leaves,
                      std::vector<ScannSearchResult>* results);

ImplStatus ImplSearchBatched(const ScannImplOpaque* impl,
                             const float* queries, size_t queries_size,
                             size_t num_queries,
                             int final_nn, int pre_reorder_nn, int leaves,
                             std::vector<std::vector<ScannSearchResult>>* results);

ImplStatus ImplSearchBatchedParallel(const ScannImplOpaque* impl,
                                     const float* queries, size_t queries_size,
                                     size_t num_queries,
                                     int final_nn, int pre_reorder_nn,
                                     int leaves, int batch_size,
                                     std::vector<std::vector<ScannSearchResult>>* results);

ImplStatus ImplInsert(ScannImplOpaque* impl,
                      const float* datapoint, size_t datapoint_size,
                      const std::string& docid,
                      int32_t* assigned_index);

ImplStatus ImplDelete(ScannImplOpaque* impl, const std::string& docid);
ImplStatus ImplDelete(ScannImplOpaque* impl, int32_t index);

ImplStatus ImplSerialize(ScannImplOpaque* impl, const std::string& path);

void ImplSetNumThreads(ScannImplOpaque* impl, int num_threads);
size_t ImplNPoints(const ScannImplOpaque* impl);
size_t ImplDimensionality(const ScannImplOpaque* impl);

// ---------------------------------------------------------------------------
// Config builders — return an opaque ScannConfigPtr ready to pass directly
// to the proto-based ImplInitialize overload.
// ---------------------------------------------------------------------------

ScannConfigPtr ImplAhConfig(int num_neighbors, int dim);
ScannConfigPtr ImplTreeAhConfig(int num_neighbors, int dim);
ScannConfigPtr ImplTreeBruteForceConfig(int num_neighbors, int dim);
ScannConfigPtr ImplBruteForceConfig(int num_neighbors, int dim, bool fixed_point);
ScannConfigPtr ImplReorderConfig(int num_neighbors, int dim);

// Serialize an opaque config to its text-format representation (useful for
// logging / debugging).
std::string ImplConfigToString(const ScannConfigOpaque& config);

}  // namespace scann_internal
}  // namespace yb
