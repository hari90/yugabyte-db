//--------------------------------------------------------------------------------------------------
// Copyright (c) Yugabyte, Inc.
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
//--------------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include <rapidjson/document.h>

#include "yb/common/ql_value.h"

#include "yb/common/common_net.pb.h"

#include "yb/util/result.h"

namespace yb {

// enum depicting the locality level of two PeerMessageQueue::TrackedPeer s'.
YB_DEFINE_ENUM(LocalityLevel, (kNone)(kRegion)(kZone));

class TablespaceParser {
 public:

  static const std::string kWildcardPlacement;

  // If fail on fail_on_validation_error is true, the functions below will return an error if the
  // extra validation checks fail. If it is false, then the function will return the replication
  // info and just log a warning. This is required for upgrade safety (and eventually to support the
  // a force flag in the yb-admin APIs).
  static Result<ReplicationInfoPB> FromString(
      const std::string& placement, bool fail_on_validation_error = true);
  static Result<ReplicationInfoPB> FromQLValue(
      const std::vector<std::string>& placements, bool fail_on_validation_error = true);

  // Returns the locality level for given CloudInfoPB references.
  static LocalityLevel GetLocalityLevel(
      const CloudInfoPB& src_cloud_info, const CloudInfoPB& dest_cloud_info);

 private:
  static Result<ReplicationInfoPB> FromJson(
      const std::string& placement_str, const rapidjson::Document& placement,
      bool fail_on_validation_error);
};

} // namespace yb
