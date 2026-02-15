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
// Interactive CLI for exercising ScaNN via ScannWrapper.
//
// Build:  ninja scann_cli
// Run:    ./tests-scann/scann_cli

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "scann/scann_wrapper.h"
#include "yb/util/result.h"
#include "yb/util/status.h"

using yb::Slice;
using yb::scann::ScannAhConfig;
using yb::scann::ScannBruteForceConfig;
using yb::scann::ScannConfigToString;
using yb::scann::ScannReorderConfig;
using yb::scann::ScannSearchResult;
using yb::scann::ScannTreeAhConfig;
using yb::scann::ScannTreeBruteForceConfig;
using yb::scann::ScannWrapper;
using yb::scann_internal::ScannConfigPtr;

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Shared REPL state
// ---------------------------------------------------------------------------

struct CliState {
  ScannWrapper scann;
  ScannConfigPtr current_config;
  bool initialized = false;
  uint32_t rng_counter = 1000;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class Timer {
 public:
  explicit Timer() : start_(Clock::now()) {}
  ~Timer() {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start_).count();
    std::cout << "[" << us / 1000.0 << " ms]\n";
  }
  private:
  Clock::time_point start_;
};

std::vector<float> RandomDataset(size_t n_points, size_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> data(n_points * dim);
  for (auto& v : data) {
    v = dist(rng);
  }
  return data;
}

std::vector<float> RandomQuery(size_t dim, uint32_t seed) {
  return RandomDataset(1, dim, seed);
}

std::string MakeRandomLabel() {
  static std::mt19937 rng(12345);
  size_t len = 8 + (rng() % 17);  // 8–24 bytes (variable length)
  std::string label(len, '\0');
  for (size_t j = 0; j < len; ++j) {
    label[j] = static_cast<char>(rng() & 0xFF);
  }
  return label;
}

std::string ToHex(const std::string& bytes) {
  std::ostringstream oss;
  for (unsigned char c : bytes) {
    oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
  }
  return oss.str();
}

void PrintResults(const std::vector<ScannSearchResult>& results) {
  for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "  #" << i << "  index=" << results[i].index
              << "  distance=" << std::fixed << std::setprecision(6)
              << results[i].distance
              << "  label=" << ToHex(results[i].label) << "\n";
  }
}

bool ReadInt(const std::string& prompt, int default_val, int* out) {
  std::cout << prompt << " [" << default_val << "]: ";
  std::string line;
  std::getline(std::cin, line);
  if (line.empty()) {
    *out = default_val;
    return true;
  }
  try {
    *out = std::stoi(line);
    return true;
  } catch (...) {
    std::cout << "  Invalid number, using default " << default_val << "\n";
    *out = default_val;
    return true;
  }
}

std::string ReadString(const std::string& prompt,
                       const std::string& default_val = "") {
  if (default_val.empty()) {
    std::cout << prompt << ": ";
  } else {
    std::cout << prompt << " [" << default_val << "]: ";
  }
  std::string line;
  std::getline(std::cin, line);
  return line.empty() ? default_val : line;
}

ScannConfigPtr ChooseConfig(int num_neighbors, int dim) {
  std::cout << "\nAvailable configs:\n"
            << "  1) brute_force (fixed_point=true)\n"
            << "  2) brute_force (fixed_point=false, mutable)\n"
            << "  3) ah\n"
            << "  4) tree_ah\n"
            << "  5) tree_brute_force\n"
            << "  6) reorder (ah + exact reordering)\n";
  int choice = 1;
  ReadInt("Config choice", 2, &choice);
  switch (choice) {
    case 1:
      return ScannBruteForceConfig(num_neighbors, dim, /*fixed_point=*/true);
    case 2:
      return ScannBruteForceConfig(num_neighbors, dim, /*fixed_point=*/false);
    case 3:
      return ScannAhConfig(num_neighbors, dim);
    case 4:
      return ScannTreeAhConfig(num_neighbors, dim);
    case 5:
      return ScannTreeBruteForceConfig(num_neighbors, dim);
    case 6:
      return ScannReorderConfig(num_neighbors, dim);
    default:
      std::cout << "  Unknown choice, using brute_force (mutable).\n";
      return ScannBruteForceConfig(num_neighbors, dim, /*fixed_point=*/false);
  }
}

void PrintHelp() {
  std::cout << "\n"
            << "Commands:\n"
            << "  init      - Initialize index with random data\n"
            << "  load      - Load index from disk\n"
            << "  save      - Serialize index to disk\n"
            << "  info      - Show index info (n_points, dimensionality)\n"
            << "  config    - Print the current config (text format)\n"
            << "  search    - Single-query nearest neighbor search\n"
            << "  bsearch   - Batched search (sequential)\n"
            << "  psearch   - Batched search (parallel)\n"
            << "  insert    - Insert a random datapoint\n"
            << "  insertn   - Insert N random datapoints\n"
            << "  delete    - Delete a datapoint by docid\n"
            << "  deletei   - Delete a datapoint by index\n"
            << "  threads   - Set number of threads for parallel search\n"
            << "  help      - Show this help\n"
            << "  exit/quit - Exit the program\n"
            << "\n";
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

void CmdInit(CliState& s) {
  int n_points = 10000, dim = 32, num_neighbors = 10, threads = 4;
  ReadInt("Number of points", 10000, &n_points);
  ReadInt("Dimensionality", 32, &dim);
  ReadInt("num_neighbors (k)", 10, &num_neighbors);
  ReadInt("Training threads", 4, &threads);

  s.current_config = ChooseConfig(num_neighbors, dim);

  std::cout << "Generating " << n_points << " random " << dim
            << "-d vectors...\n";
  auto dataset = RandomDataset(n_points, dim, /*seed=*/42);

  // Generate random labels.
  std::vector<std::string> label_storage;
  std::vector<Slice> labels;
  label_storage.reserve(n_points);
  labels.reserve(n_points);
  for (int i = 0; i < n_points; ++i) {
    label_storage.push_back(MakeRandomLabel());
  }
  for (const auto& ls : label_storage) {
    labels.emplace_back(ls);
  }

  std::cout << "Building index...\n";
  s.scann = ScannWrapper();

  Timer timer;
  auto status = s.scann.Initialize(dataset, n_points, s.current_config, threads, labels);
  if (!status.ok()) {
    std::cout << "  ERROR: " << status.ToString() << "\n";
  } else {
    s.initialized = true;
    std::cout << "  n_points=" << s.scann.n_points()
              << "  dim=" << s.scann.dimensionality() << "\n";
  }
}

void CmdLoad(CliState& s) {
  auto dir = ReadString("Artifacts directory");
  if (dir.empty()) {
    std::cout << "  No directory given.\n";
    return;
  }
  s.scann = ScannWrapper();

  Timer timer;
  auto status = s.scann.LoadFromDisk(dir);
  if (!status.ok()) {
    std::cout << "  ERROR: " << status.ToString() << "\n";
  } else {
    s.initialized = true;
    s.current_config = nullptr;
    std::cout << "  n_points=" << s.scann.n_points()
              << "  dim=" << s.scann.dimensionality() << "\n";
  }
}

void CmdSave(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  auto dir = ReadString("Output directory", "/tmp/scann_cli_save");
  std::filesystem::create_directories(dir);

  Timer timer;
  auto status = s.scann.Serialize(dir);
  if (!status.ok()) {
    std::cout << "  ERROR: " << status.ToString() << "\n";
  } else {
    std::cout << "  Written to " << dir << "\n";
  }
}

void CmdInfo(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  std::cout << "  n_points      = " << s.scann.n_points() << "\n"
            << "  dimensionality = " << s.scann.dimensionality() << "\n";
}

void CmdConfig(CliState& s) {
  if (!s.current_config) {
    std::cout << "  No config available (index was loaded from disk).\n";
  } else {
    std::cout << ScannConfigToString(s.current_config) << "\n";
  }
}

void CmdSearch(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  int k = 10, pre_reorder = 0, leaves = 0;
  ReadInt("k (num neighbors)", 10, &k);
  ReadInt("pre_reorder_nn (0 = k)", 0, &pre_reorder);
  ReadInt("leaves (0 = auto)", 0, &leaves);
  if (pre_reorder <= 0) {
    pre_reorder = k;
  }

  auto query = RandomQuery(s.scann.dimensionality(), s.rng_counter++);
  std::cout << "Searching (k=" << k << ", pre_reorder=" << pre_reorder
            << ", leaves=" << leaves << ") with random query (seed="
            << (s.rng_counter - 1) << ")...\n";

  Timer timer;
  auto result = s.scann.Search(query, k, pre_reorder, leaves);
  if (!result.ok()) {
    std::cout << "  ERROR: " << result.status().ToString() << "\n";
  } else {
    std::cout << "  (" << result->size() << " results)\n";
    PrintResults(*result);
  }
}

void CmdBsearch(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  int num_queries = 100, k = 10, pre_reorder = 0, leaves = 0;
  ReadInt("Number of queries", 100, &num_queries);
  ReadInt("k (num neighbors)", 10, &k);
  ReadInt("pre_reorder_nn (0 = k)", 0, &pre_reorder);
  ReadInt("leaves (0 = auto)", 0, &leaves);
  if (pre_reorder <= 0) {
    pre_reorder = k;
  }

  auto queries =
      RandomDataset(num_queries, s.scann.dimensionality(), s.rng_counter++);

  Timer timer;
  auto result =
      s.scann.SearchBatched(queries, num_queries, k, pre_reorder, leaves);
  if (!result.ok()) {
    std::cout << "  ERROR: " << result.status().ToString() << "\n";
  } else {
    std::cout << "  " << num_queries << " queries\n";
    if (!result->empty()) {
      std::cout << "  First query results:\n";
      PrintResults((*result)[0]);
    }
  }
}

void CmdPsearch(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  int num_queries = 1000, k = 10, pre_reorder = 0, leaves = 0;
  int batch_size = 256;
  ReadInt("Number of queries", 1000, &num_queries);
  ReadInt("k (num neighbors)", 10, &k);
  ReadInt("pre_reorder_nn (0 = k)", 0, &pre_reorder);
  ReadInt("leaves (0 = auto)", 0, &leaves);
  ReadInt("batch_size", 256, &batch_size);
  if (pre_reorder <= 0) {
    pre_reorder = k;
  }

  auto queries =
      RandomDataset(num_queries, s.scann.dimensionality(), s.rng_counter++);

  Timer timer;
  auto result = s.scann.SearchBatchedParallel(queries, num_queries, k,
                                              pre_reorder, leaves, batch_size);
  if (!result.ok()) {
    std::cout << "  ERROR: " << result.status().ToString() << "\n";
  } else {
    std::cout << "  " << num_queries << " queries\n";
    if (!result->empty()) {
      std::cout << "  First query results:\n";
      PrintResults((*result)[0]);
    }
  }
}

void CmdInsert(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  auto docid = ReadString("docid", "doc_" + std::to_string(s.rng_counter));
  auto point = RandomQuery(s.scann.dimensionality(), s.rng_counter++);
  auto label = MakeRandomLabel();

  Timer timer;
  auto result = s.scann.Insert(point, docid, Slice(label));
  if (!result.ok()) {
    std::cout << "  ERROR: " << result.status().ToString() << "\n";
  } else {
    std::cout << "  index=" << *result << ", docid=\"" << docid
              << "\", label=" << ToHex(label)
              << ", n_points=" << s.scann.n_points() << "\n";
  }
}

void CmdInsertN(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  int count = 100;
  ReadInt("Number of points to insert", 100, &count);

  size_t dim = s.scann.dimensionality();
  int success = 0;
  int fail = 0;

  Timer timer;
  for (int i = 0; i < count; ++i) {
    auto point = RandomQuery(dim, s.rng_counter++);
    std::string docid = "bulk_" + std::to_string(s.rng_counter - 1);
    auto label = MakeRandomLabel();
    auto result = s.scann.Insert(point, docid, Slice(label));
    if (result.ok()) {
      ++success;
    } else {
      ++fail;
    }
  }
  std::cout << "  " << count << " points, " << success << " ok, " << fail
            << " failed, n_points=" << s.scann.n_points() << "\n";
}

void CmdDelete(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  auto docid = ReadString("docid to delete");
  if (docid.empty()) {
    std::cout << "  No docid given.\n";
    return;
  }

  Timer timer;
  auto status = s.scann.Delete(docid);
  if (!status.ok()) {
    std::cout << "  ERROR: " << status.ToString() << "\n";
  } else {
    std::cout << "  n_points=" << s.scann.n_points() << "\n";
  }
}

void CmdDeleteI(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  int idx = 0;
  ReadInt("Datapoint index to delete", 0, &idx);

  Timer timer;
  auto status = s.scann.Delete(static_cast<int32_t>(idx));
  if (!status.ok()) {
    std::cout << "  ERROR: " << status.ToString() << "\n";
  } else {
    std::cout << "  n_points=" << s.scann.n_points() << "\n";
  }
}

void CmdThreads(CliState& s) {
  if (!s.initialized) {
    std::cout << "  Index not initialized.\n";
    return;
  }
  int n = 4;
  ReadInt("Number of threads", 4, &n);
  s.scann.SetNumThreads(n);
  std::cout << "  Threads set to " << n << ".\n";
}

// ---------------------------------------------------------------------------
// Main REPL
// ---------------------------------------------------------------------------

int main() {
  std::cout << "=== ScaNN Interactive CLI ===\n";
  PrintHelp();

  CliState state;

  while (true) {
    std::cout << "\nscann> ";
    std::string cmd;
    if (!std::getline(std::cin, cmd)) {
      break;  // EOF
    }

    // Trim whitespace.
    while (!cmd.empty() && cmd.front() == ' ') {
      cmd.erase(cmd.begin());
    }
    while (!cmd.empty() && cmd.back() == ' ') {
      cmd.pop_back();
    }

    if (cmd.empty()) {
      continue;
    }

    if (cmd == "exit" || cmd == "quit") {
      std::cout << "Goodbye.\n";
      break;
    }

    if (cmd == "help") {
      PrintHelp();
      continue;
    }

    if (cmd == "init") {
      CmdInit(state);
    } else if (cmd == "load") {
      CmdLoad(state);
    } else if (cmd == "save") {
      CmdSave(state);
    } else if (cmd == "info") {
      CmdInfo(state);
    } else if (cmd == "config") {
      CmdConfig(state);
    } else if (cmd == "search") {
      CmdSearch(state);
    } else if (cmd == "bsearch") {
      CmdBsearch(state);
    } else if (cmd == "psearch") {
      CmdPsearch(state);
    } else if (cmd == "insert") {
      CmdInsert(state);
    } else if (cmd == "insertn") {
      CmdInsertN(state);
    } else if (cmd == "delete") {
      CmdDelete(state);
    } else if (cmd == "deletei") {
      CmdDeleteI(state);
    } else if (cmd == "threads") {
      CmdThreads(state);
    } else {
      std::cout << "  Unknown command: '" << cmd
                << "'.  Type 'help' for usage.\n";
    }
  }

  return 0;
}
