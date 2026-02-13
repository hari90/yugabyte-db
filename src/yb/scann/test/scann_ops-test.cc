// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// C++ port of scann_ops_test.py - tests ScaNN via ScannWrapper without
// directly depending on ScaNN internals.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "scann/scann_wrapper.h"
#include "yb/util/result.h"
#include "yb/util/test_util.h"

namespace yb::scann {
namespace {

constexpr int kNumDatasetPoints = 10000;
constexpr int kNumQueries = 100;
constexpr int kDimension = 32;
constexpr uint32_t kSeed = 518;

// Generates random float data in [0, 1).
std::vector<float> RandomDataset(size_t n_points, size_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> data(n_points * dim);
  for (auto& v : data) v = dist(rng);
  return data;
}

// Reference brute-force dot product: for each query, compute dot products with
// all dataset points and return top-k indices and distances. DotProductDistance
// returns -dot_product (smaller = better), so we match that convention.
void NumpyBruteForceDotProduct(
    const float* dataset, const float* queries, size_t n_dataset, size_t n_queries, size_t dim,
    int k, std::vector<int32_t>* out_indices, std::vector<float>* out_distances) {
  out_indices->resize(n_queries * k);
  out_distances->resize(n_queries * k);

  for (size_t q = 0; q < n_queries; ++q) {
    const float* qptr = queries + q * dim;
    std::vector<std::pair<float, int32_t>> products(n_dataset);

    for (size_t i = 0; i < n_dataset; ++i) {
      float dot = 0;
      const float* dptr = dataset + i * dim;
      for (size_t j = 0; j < dim; ++j) dot += qptr[j] * dptr[j];
      products[i] = {-dot, static_cast<int32_t>(i)};  // -dot for distance
    }

    std::partial_sort(
        products.begin(), products.begin() + k, products.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    for (int j = 0; j < k; ++j) {
      (*out_distances)[q * k + j] = products[j].first;
      (*out_indices)[q * k + j] = products[j].second;
    }
  }
}

class ScannOpsTest : public YBTest {
 protected:
  void SetUp() override {
    dataset_ = RandomDataset(kNumDatasetPoints, kDimension, kSeed);
    queries_ = RandomDataset(kNumQueries, kDimension, kSeed + 1);
  }

  void SerializationTester(const std::string& config) {
    ScannWrapper scann1;
    auto s1 = scann1.Initialize(dataset_, kNumDatasetPoints, config, 4);
    ASSERT_TRUE(s1.ok()) << s1.ToString();

    auto results1_or = scann1.SearchBatched(queries_, kNumQueries, 15, 15, 0);
    ASSERT_TRUE(results1_or.ok()) << results1_or.status().ToString();
    auto& results1 = *results1_or;

    auto tmpdir = std::filesystem::temp_directory_path() /
                  ("scann_ops_test_" + std::to_string(kSeed));
    std::filesystem::create_directories(tmpdir);

    auto s_ser = scann1.Serialize(tmpdir.string());
    ASSERT_TRUE(s_ser.ok()) << s_ser.ToString();

    ScannWrapper scann2;
    auto s3 = scann2.LoadFromDisk(tmpdir.string());
    ASSERT_TRUE(s3.ok()) << s3.ToString();

    auto results2_or = scann2.SearchBatched(queries_, kNumQueries, 15, 15, 0);
    ASSERT_TRUE(results2_or.ok()) << results2_or.status().ToString();
    auto& results2 = *results2_or;

    for (int q = 0; q < kNumQueries; ++q) {
      ASSERT_EQ(results1[q].size(), results2[q].size()) << "query " << q;
      for (size_t j = 0; j < results1[q].size(); ++j) {
        EXPECT_EQ(results1[q][j].index, results2[q][j].index) << "query " << q << " neighbor " << j;
        EXPECT_FLOAT_EQ(results1[q][j].distance, results2[q][j].distance)
            << "query " << q << " neighbor " << j;
      }
    }

    std::filesystem::remove_all(tmpdir);
  }

  std::vector<float> dataset_;
  std::vector<float> queries_;
};

TEST_F(ScannOpsTest, AhSerialization) {
  SerializationTester(ScannAhConfig(15, kDimension));
}

TEST_F(ScannOpsTest, TreeAhSerialization) {
  SerializationTester(ScannTreeAhConfig(15, kDimension));
}

TEST_F(ScannOpsTest, TreeBruteForceSerialization) {
  SerializationTester(ScannTreeBruteForceConfig(10, kDimension));
}

TEST_F(ScannOpsTest, BruteForceInt8Serialization) {
  SerializationTester(ScannBruteForceConfig(10, kDimension));
}

TEST_F(ScannOpsTest, ReorderingSerialization) {
  SerializationTester(ScannReorderConfig(10, kDimension));
}

TEST_F(ScannOpsTest, BruteForceMatchesReference) {
  ScannWrapper scann;
  ASSERT_TRUE(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4).ok());

  auto results_or = scann.SearchBatched(queries_, kNumQueries, 10, 10, 0);
  ASSERT_TRUE(results_or.ok()) << results_or.status().ToString();
  auto& results = *results_or;

  std::vector<int32_t> ref_indices;
  std::vector<float> ref_distances;
  NumpyBruteForceDotProduct(
      dataset_.data(), queries_.data(), kNumDatasetPoints, kNumQueries, kDimension, 15,
      &ref_indices, &ref_distances);

  for (int q = 0; q < kNumQueries; ++q) {
    auto q_ref_start = ref_indices.begin() + q * 15;
    auto q_ref_end = ref_indices.begin() + (q + 1) * 15;
    std::set<int32_t> ref_index_set(q_ref_start, q_ref_end);
    for (int j = 0; j < 10; ++j) {
      int32_t idx = results[q][j].index;
      EXPECT_TRUE(ref_index_set.count(idx))
          << "query " << q << " neighbor " << j << " index " << idx;
      int ref_pos = std::find(q_ref_start, q_ref_end, idx) - q_ref_start;
      ASSERT_LT(ref_pos, 15) << "query " << q << " index " << idx;
      EXPECT_NEAR(results[q][j].distance, ref_distances[q * 15 + ref_pos], 0.03f)
          << "query " << q << " neighbor " << j;
    }
  }
}

TEST_F(ScannOpsTest, BruteForceFinalNumNeighbors) {
  ScannWrapper scann;
  ASSERT_TRUE(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4).ok());

  auto results20_or = scann.SearchBatched(queries_, kNumQueries, 20, 20, 0);
  ASSERT_TRUE(results20_or.ok()) << results20_or.status().ToString();
  auto& results20 = *results20_or;

  std::vector<int32_t> ref_indices;
  std::vector<float> ref_distances;
  NumpyBruteForceDotProduct(
      dataset_.data(), queries_.data(), kNumDatasetPoints, kNumQueries, kDimension, 25,
      &ref_indices, &ref_distances);

  for (int q = 0; q < kNumQueries; ++q) {
    ASSERT_EQ(results20[q].size(), 20u) << "query " << q;
    auto q_ref_start = ref_indices.begin() + q * 25;
    auto q_ref_end = ref_indices.begin() + (q + 1) * 25;
    std::set<int32_t> ref_index_set(q_ref_start, q_ref_end);
    for (int j = 0; j < 20; ++j) {
      int32_t idx = results20[q][j].index;
      EXPECT_TRUE(ref_index_set.count(idx))
          << "query " << q << " neighbor " << j << " index " << idx;
      int ref_pos = std::find(q_ref_start, q_ref_end, idx) - q_ref_start;
      ASSERT_LT(ref_pos, 25) << "query " << q << " index " << idx;
      EXPECT_NEAR(results20[q][j].distance, ref_distances[q * 25 + ref_pos], 0.03f)
          << "query " << q << " neighbor " << j;
    }
  }
}

TEST_F(ScannOpsTest, SingleQuerySearch) {
  ScannWrapper scann;
  ASSERT_TRUE(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4).ok());

  std::vector<int32_t> ref_indices;
  std::vector<float> ref_distances;
  NumpyBruteForceDotProduct(
      dataset_.data(), queries_.data(), kNumDatasetPoints, 1, kDimension, 25, &ref_indices,
      &ref_distances);

  // Extract just the first query vector.
  std::vector<float> first_query(queries_.begin(), queries_.begin() + kDimension);
  auto single_result_or = scann.Search(first_query, 20, 20, 0);
  ASSERT_TRUE(single_result_or.ok()) << single_result_or.status().ToString();
  auto& single_result = *single_result_or;

  ASSERT_EQ(single_result.size(), 20u);
  std::set<int32_t> ref_index_set(ref_indices.begin(), ref_indices.end());
  for (int j = 0; j < 20; ++j) {
    int32_t idx = single_result[j].index;
    EXPECT_TRUE(ref_index_set.count(idx)) << "neighbor " << j << " index " << idx;
    int ref_pos = std::find(ref_indices.begin(), ref_indices.end(), idx) - ref_indices.begin();
    ASSERT_LT(ref_pos, static_cast<int>(ref_indices.size())) << "index " << idx;
    EXPECT_NEAR(single_result[j].distance, ref_distances[ref_pos], 0.03f) << "neighbor " << j;
  }
}

TEST_F(ScannOpsTest, ParallelMatchesSequential) {
  std::vector<float> dataset = RandomDataset(10000, 32, 518);
  std::vector<float> queries = RandomDataset(1000, 32, 519);

  ScannWrapper scann;
  ASSERT_TRUE(scann.Initialize(dataset, 10000, ScannTreeAhConfig(10, 32), 4).ok());

  scann.SetNumThreads(4);

  auto results_seq_or = scann.SearchBatched(queries, 1000, 10, 10, 0);
  ASSERT_TRUE(results_seq_or.ok()) << results_seq_or.status().ToString();
  auto& results_seq = *results_seq_or;

  auto results_par_or = scann.SearchBatchedParallel(queries, 1000, 10, 10, 0, 256);
  ASSERT_TRUE(results_par_or.ok()) << results_par_or.status().ToString();
  auto& results_par = *results_par_or;

  for (int q = 0; q < 1000; ++q) {
    ASSERT_EQ(results_seq[q].size(), results_par[q].size()) << "query " << q;
    for (size_t j = 0; j < results_seq[q].size(); ++j) {
      EXPECT_EQ(results_seq[q][j].index, results_par[q][j].index)
          << "query " << q << " neighbor " << j;
      EXPECT_FLOAT_EQ(results_seq[q][j].distance, results_par[q][j].distance)
          << "query " << q << " neighbor " << j;
    }
  }
}

TEST_F(ScannOpsTest, ReorderingShapes) {
  ScannWrapper scann;
  ASSERT_TRUE(scann.Initialize(dataset_, kNumDatasetPoints, ScannReorderConfig(5, kDimension), 4).ok());

  // -- Batched searches -------------------------------------------------------
  {
    auto r = scann.SearchBatched(queries_, kNumQueries, 5, 5, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ((*r)[q].size(), 5u) << "query " << q;
    }
  }
  {
    auto r = scann.SearchBatched(queries_, kNumQueries, 8, 8, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ((*r)[q].size(), 8u) << "query " << q;
    }
  }
  {
    auto r = scann.SearchBatched(queries_, kNumQueries, 5, 8, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ((*r)[q].size(), 5u) << "query " << q;
    }
  }
  {
    auto r = scann.SearchBatched(queries_, kNumQueries, 6, 8, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ((*r)[q].size(), 6u) << "query " << q;
    }
  }

  // -- Single-query searches --------------------------------------------------
  std::vector<float> first_query(queries_.begin(), queries_.begin() + kDimension);

  {
    auto r = scann.Search(first_query, 5, 5, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    EXPECT_EQ(r->size(), 5u);
  }
  {
    auto r = scann.Search(first_query, 8, 8, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    EXPECT_EQ(r->size(), 8u);
  }
  {
    auto r = scann.Search(first_query, 5, 8, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    EXPECT_EQ(r->size(), 5u);
  }
  {
    auto r = scann.Search(first_query, 6, 8, 0);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    EXPECT_EQ(r->size(), 6u);
  }
}

}  // namespace
}  // namespace yb
