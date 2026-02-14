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

using scann_internal::ScannConfigPtr;

namespace {

constexpr int kNumDatasetPoints = 10000;
constexpr int kNumQueries = 100;
constexpr int kDimension = 32;
constexpr uint32_t kSeed = 518;

// Fixed label width for tests (16 bytes, like a UUID).
constexpr size_t kLabelWidth = 16;

// Returns a vector of n_points random labels (each kLabelWidth bytes) and
// the backing storage.  The returned Slices point into `storage`.
struct RandomLabelResult {
  std::vector<std::string> storage;
  std::vector<Slice> slices;
};

RandomLabelResult RandomLabels(size_t n_points) {
  RandomLabelResult result;
  result.storage.reserve(n_points);
  result.slices.reserve(n_points);
  std::mt19937 rng(42);
  for (size_t i = 0; i < n_points; ++i) {
    std::string bytes(kLabelWidth, '\0');
    for (size_t j = 0; j < kLabelWidth; ++j) {
      bytes[j] = static_cast<char>(rng() & 0xFF);
    }
    result.storage.push_back(std::move(bytes));
  }
  for (const auto& s : result.storage) {
    result.slices.emplace_back(s);
  }
  return result;
}

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
    auto rl = RandomLabels(kNumDatasetPoints);
    ScannWrapper scann1;
    ASSERT_OK(scann1.Initialize(dataset_, kNumDatasetPoints, config, 4,
                                kLabelWidth, rl.slices));

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
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4,
      kLabelWidth, rl.slices));

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
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4,
      kLabelWidth, rl.slices));

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
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4,
      kLabelWidth, rl.slices));

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

  auto rl = RandomLabels(10000);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(dataset, 10000, ScannTreeAhConfig(10, 32), 4,
                             kLabelWidth, rl.slices));

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
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      dataset_, kNumDatasetPoints, ScannReorderConfig(5, kDimension), 4,
      kLabelWidth, rl.slices));

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
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4,
      kLabelWidth, rl.slices));

  EXPECT_EQ(scann.n_points(), kNumDatasetPoints);
  EXPECT_EQ(scann.dimensionality(), kDimension);
}

TEST_F(ScannOpsTest, MoveConstructor) {
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann1;
  ASSERT_OK(scann1.Initialize(
      dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4,
      kLabelWidth, rl.slices));

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
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann1;
  ASSERT_OK(scann1.Initialize(
      dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4,
      kLabelWidth, rl.slices));

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

// Helper: make a single random label as an owning string.
std::string MakeRandomLabel() {
  std::string label(kLabelWidth, '\0');
  static std::mt19937 rng(999);
  for (size_t j = 0; j < kLabelWidth; ++j) {
    label[j] = static_cast<char>(rng() & 0xFF);
  }
  return label;
}

TEST_F(ScannOpsTest, InsertIncreasesNPoints) {
  auto rl = RandomLabels(kNumDatasetPoints);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      dataset_, kNumDatasetPoints, ScannBruteForceConfig(10, kDimension), 4,
      kLabelWidth, rl.slices));

  ASSERT_EQ(scann.n_points(), kNumDatasetPoints);

  // Insert a single new datapoint.
  std::vector<float> new_point(kDimension, 0.5f);
  auto lbl1 = MakeRandomLabel();
  auto idx = ASSERT_RESULT(scann.Insert(new_point, "new_point_0", Slice(lbl1)));
  EXPECT_EQ(scann.n_points(), kNumDatasetPoints + 1);

  // Insert another.
  std::vector<float> new_point2(kDimension, 0.7f);
  auto lbl2 = MakeRandomLabel();
  auto idx2 = ASSERT_RESULT(scann.Insert(new_point2, "new_point_1", Slice(lbl2)));
  EXPECT_EQ(scann.n_points(), kNumDatasetPoints + 2);

  // The two inserts should return distinct indices.
  EXPECT_NE(idx, idx2);
}

TEST_F(ScannOpsTest, InsertedPointIsSearchable) {
  // Use a small dataset so that a carefully chosen inserted point dominates.
  constexpr int kSmallN = 100;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 42);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1,
      kLabelWidth, rl.slices));

  // Create a distinctive point: all ones.
  std::vector<float> inserted(kDim, 1.0f);
  auto lbl = MakeRandomLabel();
  auto idx = ASSERT_RESULT(scann.Insert(inserted, "all_ones", Slice(lbl)));

  // Query with the same all-ones vector.
  std::vector<float> query(kDim, 1.0f);
  auto results = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].index, idx)
      << "Expected inserted point (index " << idx << ") to be the nearest neighbor, got index "
      << results[0].index;
}

TEST_F(ScannOpsTest, InsertMultipleThenBatchSearch) {
  constexpr int kSmallN = 200;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 77);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1,
      kLabelWidth, rl.slices));

  constexpr int kInsertCount = kDim;
  constexpr float kLargeVal = 100.0f;
  std::vector<int32_t> inserted_indices;
  for (int i = 0; i < kInsertCount; ++i) {
    std::vector<float> pt(kDim, 0.0f);
    pt[i] = kLargeVal;
    auto lbl = MakeRandomLabel();
    auto idx = ASSERT_RESULT(scann.Insert(pt, "inserted_" + std::to_string(i), Slice(lbl)));
    inserted_indices.push_back(idx);
  }

  EXPECT_EQ(scann.n_points(), kSmallN + kInsertCount);

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
    EXPECT_EQ(results[i][0].index, inserted_indices[i])
        << "query " << i << ": expected index " << inserted_indices[i] << ", got "
        << results[i][0].index;
  }
}

TEST_F(ScannOpsTest, DeleteByDocidDecreasesNPoints) {
  constexpr int kSmallN = 200;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 99);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1, kLabelWidth, rl.slices));

  std::vector<float> pt(kDim, 1.0f);
  auto lbl = MakeRandomLabel();
  ASSERT_OK(scann.Insert(pt, "to_delete", Slice(lbl)));
  ASSERT_EQ(scann.n_points(), kSmallN + 1);

  ASSERT_OK(scann.Delete(std::string("to_delete")));
  EXPECT_EQ(scann.n_points(), kSmallN);
}

TEST_F(ScannOpsTest, DeleteByIndexDecreasesNPoints) {
  constexpr int kSmallN = 200;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 100);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1, kLabelWidth, rl.slices));

  std::vector<float> pt(kDim, 1.0f);
  auto lbl = MakeRandomLabel();
  auto idx = ASSERT_RESULT(scann.Insert(pt, "by_index", Slice(lbl)));
  ASSERT_EQ(scann.n_points(), kSmallN + 1);

  ASSERT_OK(scann.Delete(idx));
  EXPECT_EQ(scann.n_points(), kSmallN);
}

TEST_F(ScannOpsTest, DeletedPointNotReturnedBySearch) {
  constexpr int kSmallN = 100;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 101);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1,
      kLabelWidth, rl.slices));

  std::vector<float> pt(kDim, 1.0f);
  auto lbl = MakeRandomLabel();
  auto idx = ASSERT_RESULT(scann.Insert(pt, "delete_me", Slice(lbl)));

  std::vector<float> query(kDim, 1.0f);
  {
    auto r = ASSERT_RESULT(scann.Search(query, 1, 1, 0));
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].index, idx);
  }

  ASSERT_OK(scann.Delete(std::string("delete_me")));

  {
    auto r = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
    for (const auto& result : r) {
      EXPECT_NE(result.index, idx)
          << "Deleted point (index " << idx << ") should not appear in results";
    }
  }
}

// ---------------------------------------------------------------------------
// Label tests (byte-array labels)
// ---------------------------------------------------------------------------

TEST_F(ScannOpsTest, InsertedLabelReturnedBySearch) {
  constexpr int kSmallN = 100;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 200);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1,
      kLabelWidth, rl.slices));

  // Insert a distinctive all-ones point with a known label.
  auto expected_label = MakeRandomLabel();
  std::vector<float> inserted(kDim, 1.0f);
  auto idx = ASSERT_RESULT(scann.Insert(inserted, "labeled_point", Slice(expected_label)));

  std::vector<float> query(kDim, 1.0f);
  auto results = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].index, idx);
  EXPECT_EQ(results[0].label, expected_label);
}

TEST_F(ScannOpsTest, InitializeWithLabels) {
  constexpr int kSmallN = 50;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 201);

  auto rl = RandomLabels(kSmallN);

  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1, kLabelWidth, rl.slices));

  // Search and verify the labels in the results match what we passed in.
  std::vector<float> query(small_dataset.begin(), small_dataset.begin() + kDim);
  auto results = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
  ASSERT_FALSE(results.empty());
  for (const auto& r : results) {
    EXPECT_EQ(r.label, rl.storage[r.index])
        << "index " << r.index << " label mismatch";
  }
}

TEST_F(ScannOpsTest, InitializeWithRandomLabelsGivesNonEmptyLabels) {
  constexpr int kSmallN = 50;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 202);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1, kLabelWidth, rl.slices));

  std::vector<float> query(small_dataset.begin(), small_dataset.begin() + kDim);
  auto results = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
  ASSERT_FALSE(results.empty());
  for (const auto& r : results) {
    EXPECT_EQ(r.label.size(), kLabelWidth) << "index " << r.index << " expected non-empty label";
    EXPECT_EQ(r.label, rl.storage[r.index])
        << "index " << r.index << " label mismatch";
  }
}

TEST_F(ScannOpsTest, InitializeWithWrongLabelCountFails) {
  constexpr int kSmallN = 50;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 203);

  // Provide the wrong number of labels.
  auto rl = RandomLabels(kSmallN + 5);

  ScannWrapper scann;
  auto status = scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1, kLabelWidth, rl.slices);
  EXPECT_NOK(status);
}

TEST_F(ScannOpsTest, LabelsSerializeAndLoad) {
  constexpr int kSmallN = 100;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 204);

  auto rl = RandomLabels(kSmallN);

  ScannWrapper scann1;
  ASSERT_OK(scann1.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1, kLabelWidth, rl.slices));

  auto tmpdir = std::filesystem::temp_directory_path() / "scann_label_test_204";
  std::filesystem::create_directories(tmpdir);
  ASSERT_OK(scann1.Serialize(tmpdir.string()));

  EXPECT_TRUE(std::filesystem::exists(tmpdir / "scann_labels.bin"));

  ScannWrapper scann2;
  ASSERT_OK(scann2.LoadFromDisk(tmpdir.string()));

  std::vector<float> query(small_dataset.begin(), small_dataset.begin() + kDim);
  auto results = ASSERT_RESULT(scann2.Search(query, 5, 5, 0));
  ASSERT_FALSE(results.empty());

  for (const auto& r : results) {
    EXPECT_EQ(r.label, rl.storage[r.index])
        << "label mismatch for index " << r.index << " after serialize/load";
  }

  std::filesystem::remove_all(tmpdir);
}

TEST_F(ScannOpsTest, DeleteByIndexCleansUpLabel) {
  constexpr int kSmallN = 50;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 205);

  auto rl = RandomLabels(kSmallN);
  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim, /*fixed_point=*/false), 1,
      kLabelWidth, rl.slices));

  auto label = MakeRandomLabel();
  std::vector<float> pt(kDim, 1.0f);
  auto idx = ASSERT_RESULT(scann.Insert(pt, "del_label", Slice(label)));

  std::vector<float> query(kDim, 1.0f);
  {
    auto results = ASSERT_RESULT(scann.Search(query, 1, 1, 0));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].index, idx);
    EXPECT_EQ(results[0].label, label);
  }

  ASSERT_OK(scann.Delete(idx));

  {
    auto results = ASSERT_RESULT(scann.Search(query, 5, 5, 0));
    for (const auto& r : results) {
      EXPECT_NE(r.index, idx);
    }
  }
}

TEST_F(ScannOpsTest, BatchedSearchReturnsLabels) {
  constexpr int kSmallN = 100;
  constexpr int kDim = 8;
  auto small_dataset = RandomDataset(kSmallN, kDim, 206);

  auto rl = RandomLabels(kSmallN);

  ScannWrapper scann;
  ASSERT_OK(scann.Initialize(
      small_dataset, kSmallN, ScannBruteForceConfig(5, kDim), 1, kLabelWidth, rl.slices));

  constexpr int kBatchQueries = 3;
  std::vector<float> queries(small_dataset.begin(),
                             small_dataset.begin() + kBatchQueries * kDim);

  auto results = ASSERT_RESULT(scann.SearchBatched(queries, kBatchQueries, 5, 5, 0));
  ASSERT_EQ(results.size(), kBatchQueries);

  for (int q = 0; q < kBatchQueries; ++q) {
    for (const auto& r : results[q]) {
      EXPECT_EQ(r.label, rl.storage[r.index])
          << "query " << q << " index " << r.index << " label mismatch";
    }
  }
}

}  // namespace
}  // namespace yb::scann
