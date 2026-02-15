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
// Manages the index → label map that lives alongside a ScaNN index.
//
// Labels are variable-length byte strings.  The datapoint index is used as
// an offset into a vector of strings.
//
// On disk the labels file stores:
//   [4 bytes] uint32_t  count (number of slots)
//   For each slot:
//     [4 bytes] uint32_t  label_length
//     [label_length bytes] label data

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "yb/util/slice.h"
#include "yb/util/status_fwd.h"

#include "scann/scann_wrapper_impl.h"  // ImplSearchResult

namespace yb::scann {

// ---------------------------------------------------------------------------
// Public search result: index + distance + label (owning byte string).
// ---------------------------------------------------------------------------

struct ScannSearchResult {
  int32_t index;
  float distance;
  std::string label;  // variable-length byte string
};

// ---------------------------------------------------------------------------
// ScannLabelMap — offset-based datapoint-index → variable-length label mapping.
//
// Internally a vector of strings where position `i` holds the label for
// datapoint index `i`.
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

  // Replace the entire map.  `labels` is a vector of Slices, one per datapoint.
  void Reset(const std::vector<Slice>& labels);

  // Drop all entries.
  void Clear();

  // ---------------------------------------------------------------------------
  // Single-entry operations
  // ---------------------------------------------------------------------------

  // Insert or overwrite the label for `index`.  `label` can be any size.
  // Grows the backing vector as needed, filling new slots with empty strings.
  void Put(int32_t index, Slice label);

  // Look up the label for `index`.  Returns a Slice pointing into the
  // internal buffer.  The Slice is invalidated by any mutation.
  // Returns an empty Slice if the index is out of range.
  Slice Get(int32_t index) const;

  // Clear the label for `index` (sets to empty string).
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
  // Binary format:
  //   [4 bytes] uint32_t  count — number of slots
  //   For each slot:
  //     [4 bytes] uint32_t  label_length
  //     [label_length bytes] label data
  Status Serialize(const std::string& artifacts_dir) const;

  // Read the label map from "scann_labels.bin" inside `artifacts_dir`.
  // Gracefully returns OK with an empty map if the file does not exist
  // (backwards compatibility with indices serialized before labels).
  Status Load(const std::string& artifacts_dir);

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------

  // Number of slots in the backing vector (max index + 1).
  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

 private:
  // Variable-length labels: entry i is data_[i].
  std::vector<std::string> data_;
};

}  // namespace yb::scann
