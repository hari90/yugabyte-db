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
// ScannLabelMap implementation — offset-based index → fixed-width byte label storage.

#include "scann/scann_label_map.h"

#include <cstdint>
#include <cstring>
#include <fstream>

#include "yb/util/result.h"
#include "yb/util/status.h"

namespace yb::scann {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const char* kLabelsFileName = "scann_labels.bin";

// ---------------------------------------------------------------------------
// Bulk operations
// ---------------------------------------------------------------------------

void ScannLabelMap::Reset(size_t label_width, const std::vector<Slice>& labels) {
  label_width_ = label_width;
  data_.resize(labels.size() * label_width_, '\0');
  for (size_t i = 0; i < labels.size(); ++i) {
    DCHECK_EQ(labels[i].size(), label_width_);
    std::memcpy(&data_[i * label_width_], labels[i].data(), label_width_);
  }
}

void ScannLabelMap::Clear() {
  data_.clear();
}

// ---------------------------------------------------------------------------
// Single-entry operations
// ---------------------------------------------------------------------------

void ScannLabelMap::Put(int32_t index, Slice label) {
  if (index < 0) return;
  DCHECK_EQ(label.size(), label_width_);
  auto idx = static_cast<size_t>(index);
  size_t needed = (idx + 1) * label_width_;
  if (needed > data_.size()) {
    // Grow with some headroom (1000 extra slots).
    data_.resize(needed + 1000 * label_width_, '\0');
  }
  std::memcpy(&data_[idx * label_width_], label.data(), label_width_);
}

Slice ScannLabelMap::Get(int32_t index) const {
  if (index < 0 || label_width_ == 0) return Slice();
  auto idx = static_cast<size_t>(index);
  if ((idx + 1) * label_width_ > data_.size()) {
    return Slice();
  }
  return Slice(&data_[idx * label_width_], label_width_);
}

void ScannLabelMap::Erase(int32_t index) {
  if (index < 0 || label_width_ == 0) return;
  auto idx = static_cast<size_t>(index);
  if ((idx + 1) * label_width_ <= data_.size()) {
    std::memset(&data_[idx * label_width_], 0, label_width_);
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
    Slice label = Get(r.index);
    sr.label.assign(label.cdata(), label.size());
    results.push_back(std::move(sr));
  }
  return results;
}

// ---------------------------------------------------------------------------
// Persistence
//
// Binary format:
//   [4 bytes] uint32_t  label_width
//   [4 bytes] uint32_t  count — number of slots
//   [count * label_width bytes]  raw label data
// ---------------------------------------------------------------------------

Status ScannLabelMap::Serialize(const std::string& artifacts_dir) const {
  std::string path = artifacts_dir + "/" + kLabelsFileName;

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Status(Status::kInternalError, __FILE__, __LINE__,
                  "Failed to open " + path + " for writing");
  }

  auto lw = static_cast<uint32_t>(label_width_);
  auto count = static_cast<uint32_t>(size());
  out.write(reinterpret_cast<const char*>(&lw), sizeof(lw));
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  out.write(data_.data(), static_cast<std::streamsize>(count * label_width_));

  if (!out) {
    return Status(Status::kInternalError, __FILE__, __LINE__,
                  "Failed writing label data to " + path);
  }
  return Status();
}

Status ScannLabelMap::Load(const std::string& artifacts_dir) {
  data_.clear();
  label_width_ = 0;

  std::string path = artifacts_dir + "/" + kLabelsFileName;

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    // File doesn't exist — that's fine, just no labels.
    return Status();
  }

  uint32_t lw = 0;
  uint32_t count = 0;
  in.read(reinterpret_cast<char*>(&lw), sizeof(lw));
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!in) {
    return Status(Status::kInternalError, __FILE__, __LINE__,
                  "Failed to read label header from " + path);
  }

  label_width_ = lw;
  size_t total_bytes = static_cast<size_t>(count) * label_width_;
  data_.resize(total_bytes);
  in.read(&data_[0], static_cast<std::streamsize>(total_bytes));

  if (!in) {
    return Status(Status::kInternalError, __FILE__, __LINE__,
                  "Truncated label data in " + path);
  }

  return Status();
}

}  // namespace yb::scann
