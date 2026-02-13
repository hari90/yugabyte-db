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
  for (auto& v : data) {
    v = dist(rng);
  }
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
      for (size_t j = 0; j < dim; ++j) {
        dot += qptr[j] * dptr[j];
      }
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

  void SerializationTester(const scann_internal::ScannConfigPtr& config) {
    ScannWrapper scann1;
    ASSERT_OK(scann1.Initialize(dataset_, kNumDatasetPoints, config, 4));

    auto results1 = ASSERT_RESULT(scann1.SearchBatched(queries_, kNumQueries, 15, 15, 0));

    auto tmpdir = std::filesystem::temp_directory_path() /
                  ("scann_ops_test_" + std::to_string(kSeed));
    std::filesystem::create_directories(tmpdir);

    ASSERT_OK(scann1.Serialize(tmpdir.string()));

    ScannWrapper scann2;
    ASSERT_OK(scann2.LoadFromDisk(tmpdir.string()));

    auto results2 = ASSERT_RESULT(scann2.SearchBatched(queries_, kNumQueries, 15, 15, 0));

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
  ASSERT_OK(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4));

  auto results = ASSERT_RESULT(scann.SearchBatched(queries_, kNumQueries, 10, 10, 0));

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
  ASSERT_OK(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4));

  auto results20 = ASSERT_RESULT(scann.SearchBatched(queries_, kNumQueries, 20, 20, 0));

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
  ASSERT_OK(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4));

  std::vector<int32_t> ref_indices;
  std::vector<float> ref_distances;
  NumpyBruteForceDotProduct(
      dataset_.data(), queries_.data(), kNumDatasetPoints, 1, kDimension, 25, &ref_indices,
      &ref_distances);

  // Extract just the first query vector.
  std::vector<float> first_query(queries_.begin(), queries_.begin() + kDimension);
  auto single_result = ASSERT_RESULT(scann.Search(first_query, 20, 20, 0));

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
  ASSERT_OK(scann.Initialize(dataset, 10000, ScannTreeAhConfig(10, 32), 4));

  scann.SetNumThreads(4);

  auto results_seq = ASSERT_RESULT(scann.SearchBatched(queries, 1000, 10, 10, 0));
  auto results_par = ASSERT_RESULT(scann.SearchBatchedParallel(queries, 1000, 10, 10, 0, 256));

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
  ASSERT_OK(scann.Initialize(dataset_, kNumDatasetPoints, ScannReorderConfig(5, kDimension), 4));

  // -- Batched searches -------------------------------------------------------
  {
    auto r = ASSERT_RESULT(scann.SearchBatched(queries_, kNumQueries, 5, 5, 0));
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ(r[q].size(), 5u) << "query " << q;
    }
  }
  {
    auto r = ASSERT_RESULT(scann.SearchBatched(queries_, kNumQueries, 8, 8, 0));
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ(r[q].size(), 8u) << "query " << q;
    }
  }
  {
    auto r = ASSERT_RESULT(scann.SearchBatched(queries_, kNumQueries, 5, 8, 0));
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ(r[q].size(), 5u) << "query " << q;
    }
  }
  {
    auto r = ASSERT_RESULT(scann.SearchBatched(queries_, kNumQueries, 6, 8, 0));
    for (int q = 0; q < kNumQueries; ++q) {
      EXPECT_EQ(r[q].size(), 6u) << "query " << q;
    }
  }

  // -- Single-query searches --------------------------------------------------
  std::vector<float> first_query(queries_.begin(), queries_.begin() + kDimension);

  {
    auto r = ASSERT_RESULT(scann.Search(first_query, 5, 5, 0));
    EXPECT_EQ(r.size(), 5u);
  }
  {
    auto r = ASSERT_RESULT(scann.Search(first_query, 8, 8, 0));
    EXPECT_EQ(r.size(), 8u);
  }
  {
    auto r = ASSERT_RESULT(scann.Search(first_query, 5, 8, 0));
    EXPECT_EQ(r.size(), 5u);
  }
  {
    auto r = ASSERT_RESULT(scann.Search(first_query, 6, 8, 0));
    EXPECT_EQ(r.size(), 6u);
  }
}

TEST_F(ScannOpsTest, NPointsAndDimensionality) {
  ScannWrapper scann;
  ASSERT_OK(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4));

  EXPECT_EQ(scann.n_points(), kNumDatasetPoints);
  EXPECT_EQ(scann.dimensionality(), kDimension);
}

TEST_F(ScannOpsTest, MoveConstructor) {
  ScannWrapper scann1;
  ASSERT_OK(
      scann1.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4));

  // Search before move.
  std::vector<float> query(queries_.begin(), queries_.begin() + kDimension);
  auto before = ASSERT_RESULT(scann1.Search(query, 5, 5, 0));

  // Move-construct a new wrapper.
  ScannWrapper scann2(std::move(scann1));
  EXPECT_EQ(scann2.n_points(), kNumDatasetPoints);
  EXPECT_EQ(scann2.dimensionality(), kDimension);

  // Searching the moved-to wrapper should return the same results.
  auto after = ASSERT_RESULT(scann2.Search(query, 5, 5, 0));

  ASSERT_EQ(before.size(), after.size());
  for (size_t j = 0; j < before.size(); ++j) {
    EXPECT_EQ(before[j].index, after[j].index) << "neighbor " << j;
    EXPECT_FLOAT_EQ(before[j].distance, after[j].distance) << "neighbor " << j;
  }
}

TEST_F(ScannOpsTest, MoveAssignment) {
  ScannWrapper scann1;
  ASSERT_OK(
      scann1.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4));

  std::vector<float> query(queries_.begin(), queries_.begin() + kDimension);
  auto before = ASSERT_RESULT(scann1.Search(query, 5, 5, 0));

  // Move-assign into a default-constructed wrapper.
  ScannWrapper scann2;
  scann2 = std::move(scann1);

  auto after = ASSERT_RESULT(scann2.Search(query, 5, 5, 0));

  ASSERT_EQ(before.size(), after.size());
  for (size_t j = 0; j < before.size(); ++j) {
    EXPECT_EQ(before[j].index, after[j].index) << "neighbor " << j;
    EXPECT_FLOAT_EQ(before[j].distance, after[j].distance) << "neighbor " << j;
  }
}

TEST_F(ScannOpsTest, InsertIncreasesNPoints) {
  ScannWrapper scann;
  ASSERT_OK(
      scann.Initialize(dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4));

  ASSERT_EQ(scann.n_points(), kNumDatasetPoints);

  // Insert a single new datapoint.
  std::vector<float> new_point(kDimension, 0.5f);
  auto idx = ASSERT_RESULT(scann.Insert(new_point, "new_point_0"));
  EXPECT_EQ(scann.n_points(), kNumDatasetPoints + 1);

  // Insert another.
  std::vector<float> new_point2(kDimension, 0.7f);
  auto idx2 = ASSERT_RESULT(scann.Insert(new_point2, "new_point_1"));
  EXPECT_EQ(scann.n_points(), kNumDatasetPoints + 2);

  // The two inserts should return distinct indices.
  EXPECT_NE(idx, idx2);
}

TEST_F(ScannOpsTest, InsertedPointIsSearchable) {
  // Use a small dataset so that a carefully chosen inserted point dominates.
  constexpr int kSmallN = 100;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 42);

  // Plain brute force WITHOUT fixed_point quantization.  The fixed_point path
  // pre-quantizes the dataset at build time and dynamically inserted points
  // are not included in that representation, making them invisible to search.
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1));

  // Create a distinctive point: all ones.  Its dot product with an all-ones
  // query will be kDim (= 8), which is larger than a random [0,1) vector's
  // expected dot product (~kDim/4).
  std::vector<float> inserted(kDim, 1.0f);
  auto idx = ASSERT_RESULT(scann.Insert(inserted, "all_ones"));

  // Query with the same all-ones vector.  DotProductDistance = -dot, so the
  // inserted point should have the most negative (best) distance.
  std::vector<float> query(kDim, 1.0f);
  auto results = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
  ASSERT_FALSE(results.empty());
  // The inserted point should appear as the top-1 neighbor.
  EXPECT_EQ(results[0].index, idx)
      << "Expected inserted point (index " << idx << ") to be the nearest neighbor, got index "
      << results[0].index;
}

TEST_F(ScannOpsTest, InsertMultipleThenBatchSearch) {
  constexpr int kSmallN = 200;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 77);

  // Plain brute force WITHOUT fixed_point quantization.  The fixed_point path
  // pre-quantizes the dataset at build time and dynamically inserted points
  // are not included in that representation, making them invisible to search.
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1));

  // Insert kDim one-hot basis vectors.  Point i has a large value only in
  // dimension i, so each point is orthogonal to the others and the dot
  // product between query i and point j (i != j) is zero.  This guarantees
  // that query i uniquely matches inserted point i.
  constexpr int kInsertCount = kDim;
  constexpr float kLargeVal = 100.0f;  // dominates random [0,1) data
  std::vector<int32_t> inserted_indices;
  for (int i = 0; i < kInsertCount; ++i) {
    std::vector<float> pt(kDim, 0.0f);
    pt[i] = kLargeVal;
    auto idx = ASSERT_RESULT(scann.Insert(pt, "inserted_" + std::to_string(i)));
    inserted_indices.push_back(idx);
  }

  EXPECT_EQ(scann.n_points(), kSmallN + kInsertCount);

  // Build queries identical to the inserted one-hot vectors.
  std::vector<float> queries;
  queries.reserve(kInsertCount * kDim);
  for (int i = 0; i < kInsertCount; ++i) {
    for (int d = 0; d < kDim; ++d) {
      queries.push_back(d == i ? kLargeVal : 0.0f);
    }
  }

  auto results = ASSERT_RESULT(scann.SearchBatched(queries, kInsertCount, 1, 1, 0));
  ASSERT_EQ(results.size(), static_cast<size_t>(kInsertCount));
  for (int i = 0; i < kInsertCount; ++i) {
    ASSERT_EQ(results[i].size(), 1u) << "query " << i;
    // Each query should find its corresponding inserted point as top-1.
    EXPECT_EQ(results[i][0].index, inserted_indices[i])
        << "query " << i << ": expected index " << inserted_indices[i] << ", got "
        << results[i][0].index;
  }
}

TEST_F(ScannOpsTest, DeleteByDocidDecreasesNPoints) {
  constexpr int kSmallN = 200;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 99);

  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1));

  // Insert a point so we have a known docid.
  std::vector<float> pt(kDim, 1.0f);
  ASSERT_OK(scann.Insert(pt, "to_delete"));
  ASSERT_EQ(scann.n_points(), kSmallN + 1);

  // Delete by docid.
  ASSERT_OK(scann.Delete(std::string("to_delete")));
  EXPECT_EQ(scann.n_points(), kSmallN);
}

TEST_F(ScannOpsTest, DeleteByIndexDecreasesNPoints) {
  constexpr int kSmallN = 200;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 100);

  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1));

  // Insert a point and delete it by its numeric index.
  std::vector<float> pt(kDim, 1.0f);
  auto idx = ASSERT_RESULT(scann.Insert(pt, "by_index"));
  ASSERT_EQ(scann.n_points(), kSmallN + 1);

  ASSERT_OK(scann.Delete(idx));
  EXPECT_EQ(scann.n_points(), kSmallN);
}

TEST_F(ScannOpsTest, DeletedPointNotReturnedBySearch) {
  constexpr int kSmallN = 100;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 101);

  // Plain brute force without fixed_point so inserted points are searchable.
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1));

  // Insert a distinctive all-ones point.
  std::vector<float> pt(kDim, 1.0f);
  auto idx = ASSERT_RESULT(scann.Insert(pt, "delete_me"));

  // Confirm it is the top-1 result for an all-ones query.
  std::vector<float> query(kDim, 1.0f);
  {
    auto r = ASSERT_RESULT(scann.Search(query, 1, 1, 0));
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].index, idx);
  }

  // Delete it and search again — it should no longer appear.
  ASSERT_OK(scann.Delete(std::string("delete_me")));

  {
    auto r = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
    for (const auto& result : r) {
      EXPECT_NE(result.index, idx)
          << "Deleted point (index " << idx << ") should not appear in results";
    }
  }
}

}  // namespace
}  // namespace yb
