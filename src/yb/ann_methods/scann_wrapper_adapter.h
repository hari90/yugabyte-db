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
// Adapter that exposes Google ScaNN as a VectorIndexIf backend for VectorLSM.
// Only insert and search are supported.  Iteration (compaction) is not
// implemented.  Deletes/updates will return an error.
//
// This header deliberately avoids including any ScaNN headers so that it can
// be safely included from any translation unit without triggering the
// absl/yb dynamic_annotations macro conflict.

#pragma once

#include "yb/util/result.h"

#include "yb/vector_index/hnsw_options.h"
#include "yb/vector_index/coordinate_types.h"
#include "yb/vector_index/vector_index_if.h"

namespace yb::ann_methods {

template<vector_index::IndexableVectorType Vector,
         vector_index::ValidDistanceResultType DistanceResult>
class ScannIndexFactory {
 public:
  static vector_index::VectorIndexIfPtr<Vector, DistanceResult> Create(
      vector_index::FactoryMode mode, const vector_index::HNSWOptions& options);
};

}  // namespace yb::ann_methods
