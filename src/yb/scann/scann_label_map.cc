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
// ScannLabelMap implementation — offset-based index → ScannVectorId storage.

#include "scann/scann_label_map.h"

#include <cstdint>
#include <fstream>

#include "yb/util/result.h"
#include "yb/util/status.h"

namespace yb::scann {

// ---------------------------------------------------------------------------
// ScannVectorId strongly-typed UUID implementation
// ---------------------------------------------------------------------------

YB_STRONGLY_TYPED_UUID_IMPL(ScannVectorId);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr size_t kUuidBytes = 16;
static const char* kLabelsFileName = "scann_labels.bin";

// ---------------------------------------------------------------------------
// Bulk operations
// ---------------------------------------------------------------------------

void ScannLabelMap::Reset(const std::vector<ScannVectorId>& labels) {
  labels_ = labels;
}

void ScannLabelMap::Clear() {
  labels_.clear();
}

// ---------------------------------------------------------------------------
// Single-entry operations
// ---------------------------------------------------------------------------

void ScannLabelMap::Put(int32_t index, const ScannVectorId& label) {
  if (index < 0) return;
  auto idx = static_cast<size_t>(index);
  if (idx >= labels_.size()) {
    labels_.resize(idx + 1000, ScannVectorId::Nil());
  }
  labels_[idx] = label;
}

ScannVectorId ScannLabelMap::Get(int32_t index) const {
  if (index < 0) return ScannVectorId::Nil();
  auto idx = static_cast<size_t>(index);
  return (idx < labels_.size()) ? labels_[idx] : ScannVectorId::Nil();
}

void ScannLabelMap::Erase(int32_t index) {
  if (index < 0) return;
  auto idx = static_cast<size_t>(index);
  if (idx < labels_.size()) {
    labels_[idx] = ScannVectorId::Nil();
  }
}

// ---------------------------------------------------------------------------
// Search-result enrichment
// ---------------------------------------------------------------------------

std::vector<ScannSearchResult> ScannLabelMap::ResolveLabels(
    const std::vector<scann_internal::ImplSearchResult>& impl_results) const {
  std::vector<ScannSearchResult> results;
  results.reserve(impl_results.size());
  for (const auto& r : impl_results) {
    ScannSearchResult sr;
    sr.index = r.index;
    sr.distance = r.distance;
    sr.label = Get(r.index);
    results.push_back(sr);
  }
  return results;
}

// ---------------------------------------------------------------------------
// Persistence
//
// Offset-based binary format:
//   [4 bytes]          uint32_t  count — vector size (max index + 1)
//   [count * 16 bytes] UUID raw bytes, consecutively for indices 0..count-1
//
// The label for datapoint index `i` is at file offset  4 + i * 16.
// No per-entry index field is stored — position implies index.
// ---------------------------------------------------------------------------

Status ScannLabelMap::Serialize(const std::string& artifacts_dir) const {
  std::string path = artifacts_dir + "/" + kLabelsFileName;

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Status(Status::kInternalError, __FILE__, __LINE__,
                  "Failed to open " + path + " for writing");
  }

  uint32_t count = static_cast<uint32_t>(labels_.size());
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));

  for (const auto& label : labels_) {
    out.write(reinterpret_cast<const char*>(label.data()), kUuidBytes);
  }

  if (!out) {
    return Status(Status::kInternalError, __FILE__, __LINE__,
                  "Failed writing label data to " + path);
  }
  return Status();
}

Status ScannLabelMap::Load(const std::string& artifacts_dir) {
  labels_.clear();

  std::string path = artifacts_dir + "/" + kLabelsFileName;

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    // File doesn't exist — that's fine, just no labels.
    return Status();
  }

  uint32_t count = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!in) {
    return Status(Status::kInternalError, __FILE__, __LINE__,
                  "Failed to read label count from " + path);
  }

  labels_.resize(count, ScannVectorId::Nil());
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t uuid_bytes[kUuidBytes];
    in.read(reinterpret_cast<char*>(uuid_bytes), kUuidBytes);

    if (!in) {
      return Status(Status::kInternalError, __FILE__, __LINE__,
                    "Truncated label data in " + path);
    }

    labels_[i] = ScannVectorId(
        Uuid::TryFullyDecode(Slice(uuid_bytes, kUuidBytes)));
  }

  return Status();
}

}  // namespace yb::scann
