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

#include <gtest/gtest.h>

#include "yb/util/flags.h"
#include "yb/util/test_macros.h"
#include "yb/util/test_util.h"

#if defined(__linux__)
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

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

#if defined(__linux__)

TEST_F(InitTest, YB_LINUX_ONLY_TEST(DisableTHPViaPrctl)) {
  int thp_before = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  ASSERT_GE(thp_before, 0);

  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0));

  int thp_after = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  ASSERT_EQ(1, thp_after);

  // Re-enable for this process so we don't affect other tests.
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0));

  int thp_restored = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  ASSERT_EQ(thp_before, thp_restored);
}

TEST_F(InitTest, YB_LINUX_ONLY_TEST(DisableTHPInChildProcess)) {
  pid_t pid = fork();
  ASSERT_NE(-1, pid);

  if (pid == 0) {
    // Child: disable THP and verify.
    if (prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0) {
      _exit(1);
    }
    int thp_status = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
    if (thp_status < 0) {
      _exit(2);
    }
    _exit(thp_status == 1 ? 0 : 3);
  }

  // Parent: wait for child and verify it succeeded.
  int wstatus = 0;
  ASSERT_EQ(pid, waitpid(pid, &wstatus, 0));
  ASSERT_TRUE(WIFEXITED(wstatus));
  ASSERT_EQ(0, WEXITSTATUS(wstatus));

  // Verify parent process THP state is unaffected by child.
  int parent_thp = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  ASSERT_EQ(0, parent_thp);
}

TEST_F(InitTest, YB_LINUX_ONLY_TEST(THPDisableFlagDefaultsFalse)) {
  ASSERT_FALSE(FLAGS_disable_transparent_hugepages);
}

TEST_F(InitTest, YB_LINUX_ONLY_TEST(THPDisableIsIdempotent)) {
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0));
  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0));

  int thp_status = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  ASSERT_EQ(1, thp_status);

  ASSERT_EQ(0, prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0));
}

#endif  // defined(__linux__)

}  // namespace yb
