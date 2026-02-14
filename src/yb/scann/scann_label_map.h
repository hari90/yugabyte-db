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
// Manages the index → ScannVectorId label map that lives alongside a ScaNN
// index.  This is a separate class so that label bookkeeping is cleanly
// isolated from the ScaNN search/mutation logic in ScannWrapper.
//
// The label map is offset-based: the datapoint index is used as a direct
// offset into a flat vector of ScannVectorId values.  Deleted or unassigned
// slots hold ScannVectorId::Nil().
//
// On disk the labels are stored consecutively (no per-entry index field),
// so the label for datapoint index `i` lives at file offset `4 + i * 16`
// (after the 4-byte entry-count header).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "yb/util/status_fwd.h"
#include "yb/util/strongly_typed_uuid.h"

#include "scann/scann_wrapper_impl.h"  // ImplSearchResult

namespace yb::scann {

// ---------------------------------------------------------------------------
// ScannVectorId — strongly-typed UUID label for each indexed vector.
// ---------------------------------------------------------------------------

YB_STRONGLY_TYPED_UUID_DECL(ScannVectorId);

// ---------------------------------------------------------------------------
// Public search result: index + distance + label.
// ---------------------------------------------------------------------------

struct ScannSearchResult {
  int32_t index;
  float distance;
  ScannVectorId label;
};

// ---------------------------------------------------------------------------
// ScannLabelMap — offset-based datapoint-index → ScannVectorId mapping.
//
// Internally a flat std::vector<ScannVectorId> where position `i` holds the
// label for datapoint index `i`.  Absent or deleted entries are Nil.
//
// The map is persisted to / restored from a binary file ("scann_labels.bin")
// inside the ScaNN artifacts directory.
// ---------------------------------------------------------------------------

class ScannLabelMap {
 public:
  ScannLabelMap() = default;

  // Movable, not copyable (same semantics as the owning ScannWrapper).
  ScannLabelMap(ScannLabelMap&&) noexcept = default;
  ScannLabelMap& operator=(ScannLabelMap&&) noexcept = default;
  ScannLabelMap(const ScannLabelMap&) = delete;
  ScannLabelMap& operator=(const ScannLabelMap&) = delete;

  // ---------------------------------------------------------------------------
  // Bulk operations
  // ---------------------------------------------------------------------------

  // Replace the entire map with labels[0..n-1] mapped to indices 0..n-1.
  void Reset(const std::vector<ScannVectorId>& labels);

  // Drop all entries.
  void Clear();

  // ---------------------------------------------------------------------------
  // Single-entry operations
  // ---------------------------------------------------------------------------

  // Insert or overwrite the label for `index`.  Grows the backing vector as
  // needed, filling new slots with Nil.
  void Put(int32_t index, const ScannVectorId& label);

  // Look up the label for `index`.  Returns Nil if the index is out of range
  // or the slot has been erased.
  ScannVectorId Get(int32_t index) const;

  // Clear the label for `index` (sets it to Nil).
  void Erase(int32_t index);

  // ---------------------------------------------------------------------------
  // Search-result enrichment
  // ---------------------------------------------------------------------------

  // Convert a vector of internal bridge results into public results,
  // populating the label field from the map.
  std::vector<ScannSearchResult> ResolveLabels(
      const std::vector<scann_internal::ImplSearchResult>& impl_results) const;

  // ---------------------------------------------------------------------------
  // Persistence
  // ---------------------------------------------------------------------------

  // Write the label map to "scann_labels.bin" inside `artifacts_dir`.
  //
  // Binary format (offset-based, no per-entry index):
  //   [4 bytes]          uint32_t  count  — number of entries (= vector size)
  //   [count * 16 bytes] UUID raw bytes, one per index 0..count-1
  //
  // Label for index `i` is at file offset 4 + i * 16.
  Status Serialize(const std::string& artifacts_dir) const;

  // Read the label map from "scann_labels.bin" inside `artifacts_dir`.
  // Gracefully returns OK with an empty map if the file does not exist
  // (backwards compatibility with indices serialized before labels).
  Status Load(const std::string& artifacts_dir);

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------

  // Number of slots in the backing vector (max index + 1).
  size_t size() const { return labels_.size(); }
  bool empty() const { return labels_.empty(); }

 private:
  std::vector<ScannVectorId> labels_;
};

}  // namespace yb::scann
