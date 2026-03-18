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

#include <gtest/gtest.h>

#include "yb/util/flags.h"
#include "yb/util/test_util.h"

#if defined(__linux__)
#include <sys/prctl.h>
#ifndef PR_SET_THP_DISABLE
#define PR_SET_THP_DISABLE 41
#endif
#ifndef PR_GET_THP_DISABLE
#define PR_GET_THP_DISABLE 42
#endif
#endif

DECLARE_bool(disable_transparent_hugepages);

namespace yb {

class InitTest : public YBTest {};

TEST_F(InitTest, DisableTransparentHugepagesFlagDefaultIsFalse) {
  ASSERT_FALSE(FLAGS_disable_transparent_hugepages);
}

#if defined(__linux__)

TEST_F(InitTest, PrctlDisablesTHP) {
  // Save original THP state so we can restore it.
  int original_state = 0;
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &original_state, 0, 0, 0))
      << "Failed to get initial THP state";

  // Disable THP for this process.
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0))
      << "prctl(PR_SET_THP_DISABLE, 1) failed";

  // Verify THP is disabled.
  int thp_disabled = 0;
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &thp_disabled, 0, 0, 0))
      << "Failed to get THP state after disable";
  ASSERT_TRUE(thp_disabled) << "THP should be disabled after prctl(PR_SET_THP_DISABLE, 1)";

  // Re-enable THP (restore original state).
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, original_state, 0, 0, 0))
      << "Failed to restore original THP state";

  // Verify restoration.
  int restored_state = -1;
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &restored_state, 0, 0, 0))
      << "Failed to get THP state after restore";
  ASSERT_EQ(original_state, restored_state) << "THP state should be restored to original";
}

TEST_F(InitTest, PrctlEnablesTHP) {
  // Save original state.
  int original_state = 0;
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &original_state, 0, 0, 0));

  // Explicitly enable THP (set disable=0).
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0))
      << "prctl(PR_SET_THP_DISABLE, 0) failed";

  int thp_disabled = 1;
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &thp_disabled, 0, 0, 0));
  ASSERT_FALSE(thp_disabled) << "THP should be enabled after prctl(PR_SET_THP_DISABLE, 0)";

  // Restore original state.
  prctl(PR_SET_THP_DISABLE, original_state, 0, 0, 0);
}

TEST_F(InitTest, PrctlTHPToggle) {
  // Save original state.
  int original_state = 0;
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &original_state, 0, 0, 0));

  // Disable THP.
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0));
  int state = 0;
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &state, 0, 0, 0));
  ASSERT_TRUE(state);

  // Re-enable THP.
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0));
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &state, 0, 0, 0));
  ASSERT_FALSE(state);

  // Disable again to confirm toggling works repeatedly.
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0));
  ASSERT_EQ(0, prctl(PR_GET_THP_DISABLE, &state, 0, 0, 0));
  ASSERT_TRUE(state);

  // Restore original state.
  prctl(PR_SET_THP_DISABLE, original_state, 0, 0, 0);
}

#endif  // defined(__linux__)

}  // namespace yb
