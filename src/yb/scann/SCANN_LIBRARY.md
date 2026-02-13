# ScaNN Library for YugabyteDB

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Three-Layer Design](#three-layer-design)
   - [Header Isolation Strategy](#header-isolation-strategy)
3. [Directory Structure](#directory-structure)
4. [Search Algorithm Strategies](#search-algorithm-strategies)
   - [Brute Force](#1-brute-force)
   - [Asymmetric Hashing (AH)](#2-asymmetric-hashing-ah)
   - [Tree-AH Hybrid with Residual Quantization](#3-tree-ah-hybrid-with-residual-quantization)
   - [Tree-Brute Force Hybrid](#4-tree-brute-force-hybrid)
   - [AH with Exact Reordering](#5-ah-with-exact-reordering)
5. [Configuration System](#configuration-system)
   - [ScannConfig Protobuf](#scannconfig-protobuf)
   - [Preset Configuration Builders](#preset-configuration-builders)
6. [Core Type System](#core-type-system)
   - [Data Types](#data-types)
   - [Distance Measures](#distance-measures)
7. [Search Pipeline](#search-pipeline)
   - [Single Query Flow](#single-query-flow)
   - [Batched Search](#batched-search)
   - [Parallel Batched Search](#parallel-batched-search)
   - [Search Parameters](#search-parameters)
8. [Index Construction and Training](#index-construction-and-training)
   - [Factory System](#factory-system)
   - [K-Means Tree Training](#k-means-tree-training)
   - [Asymmetric Hashing Training](#asymmetric-hashing-training)
9. [Mutation (Insert / Delete / Update)](#mutation-insert--delete--update)
10. [Serialization and Persistence](#serialization-and-persistence)
    - [Artifact Types](#artifact-types)
    - [Serialization Flow](#serialization-flow)
    - [Deserialization Flow](#deserialization-flow)
11. [SIMD-Accelerated Kernels](#simd-accelerated-kernels)
12. [Reordering](#reordering)
13. [Searcher Class Hierarchy](#searcher-class-hierarchy)
14. [Build System](#build-system)
15. [Testing](#testing)
16. [Key Design Decisions](#key-design-decisions)

---

## Overview

`src/yb/scann` integrates Google's **ScaNN** (Scalable Approximate Nearest Neighbor) library into YugabyteDB. ScaNN is a high-performance vector similarity search library originally developed by Google Research for efficiently finding the closest vectors in a dataset given a query vector.

The integration consists of approximately 390 source files: the upstream Google ScaNN engine (adapted to the `yb` namespace) plus a custom YugabyteDB wrapper layer that provides a clean C++ API. The wrapper isolates ScaNN's internal dependencies (abseil, protobuf) from YugabyteDB's own type system, avoiding macro and symbol conflicts.

### What It Does

Given:
- A **dataset** of N float vectors, each of dimension D
- A **query** float vector of dimension D
- A desired number of **nearest neighbors** K

ScaNN returns the K vectors from the dataset that are closest to the query vector according to a configured distance measure (typically dot product or L2 distance), trading perfect accuracy for dramatically faster search when using approximate methods.

---

## Architecture

### Three-Layer Design

The library uses a strict three-layer architecture to manage the conflict between ScaNN's types and YugabyteDB's types:

```
┌──────────────────────────────────────────────────┐
│           Layer 1: Public API                    │
│     scann_wrapper.h / scann_wrapper.cc           │
│     Uses: yb::Status, yb::Result, std::vector    │
│     Never includes: ScaNN headers, abseil        │
├──────────────────────────────────────────────────┤
│           Layer 2: Bridge / PIMPL                │
│   scann_wrapper_impl.h / scann_wrapper_impl.cc   │
│     Uses: ImplStatus (POD), ScannImplOpaque*     │
│     impl.cc includes ScaNN headers ONLY          │
├──────────────────────────────────────────────────┤
│           Layer 3: ScaNN Core Engine             │
│        scann_ops/cc/scann.h / scann.cc           │
│     Uses: absl::Status, ScannConfig, etc.        │
│     Pure ScaNN + abseil world                    │
└──────────────────────────────────────────────────┘
```

### Header Isolation Strategy

The core problem: ScaNN operates in namespace `yb` and does `using ::absl::Status` to make `yb::Status` an alias for `absl::Status`. YugabyteDB defines its own `yb::Status` class. Additionally, `absl/base/internal/dynamic_annotations.h` defines macros that conflict with `yb/gutil/dynamic_annotations.h`.

The solution is a double-PIMPL pattern:

1. **`ScannImplOpaque`** — An opaque forward-declared struct. Only `scann_wrapper_impl.cc` knows its definition (wrapping `ScannInterface scann`).

2. **`ImplStatus`** — A plain-old-data struct `{int code; string message}` that carries error status without depending on either `absl::Status` or `yb::Status`:
   ```cpp
   struct ImplStatus {
     int code;  // 0 = OK, otherwise absl::StatusCode
     std::string message;
     bool ok() const { return code == 0; }
   };
   ```

3. **Free functions** (`ImplInitialize`, `ImplSearch`, etc.) act as the bridge API, accepting raw pointers and returning `ImplStatus`. These are the only functions that cross the boundary.

4. **Status conversion** happens at each boundary:
   - `scann_wrapper_impl.cc`: `absl::Status` → `ImplStatus` (via `ToImplStatus()`)
   - `scann_wrapper.cc`: `ImplStatus` → `yb::Status` (via `ImplToYbStatus()`)

---

## Directory Structure

```
src/yb/scann/
├── CMakeLists.txt                    # Build configuration
├── scann_wrapper.h                   # Public API (Layer 1)
├── scann_wrapper.cc                  # Public API implementation
├── scann_wrapper_impl.h              # Bridge API (Layer 2)
├── scann_wrapper_impl.cc             # Bridge implementation (only TU with ScaNN headers)
│
├── scann_ops/                        # ScaNN core interface
│   ├── cc/
│   │   ├── scann.h                   # ScannInterface class (Layer 3)
│   │   ├── scann.cc                  # Core implementation
│   │   ├── kernels/                  # TF kernel bindings (excluded from build)
│   │   └── python/                   # Python bindings (excluded from build)
│   └── scann_assets.proto            # Asset manifest proto
│
├── base/                             # Core searcher abstractions
│   ├── single_machine_base.h/.cc     # SingleMachineSearcherBase<T>
│   ├── single_machine_factory_options.h  # Factory configuration
│   ├── single_machine_factory_scann.h    # Main factory function
│   ├── single_machine_base_mutator.cc    # Base mutator logic
│   └── search_parameters.h/.cc       # Per-query search configuration
│
├── brute_force/                      # Exhaustive search implementations
│   ├── brute_force.h/.cc             # BruteForceSearcher<T>
│   ├── scalar_quantized_brute_force.h/.cc  # Int8-quantized brute force
│   └── bfloat16_brute_force_mutator.cc     # BFloat16 mutator
│
├── tree_x_hybrid/                    # Tree-partitioned hybrid searchers
│   ├── tree_x_hybrid_smmd.h/.cc      # TreeXHybridSMMD<T> (generic tree hybrid)
│   ├── tree_ah_hybrid_residual.h/.cc # TreeAHHybridResidual (tree + AH + residuals)
│   ├── mutator.h/.cc                 # TreeXHybridMutator<T>
│   ├── tree_x_params.h              # TreeXOptionalParameters
│   └── internal/                     # Internal utilities
│
├── hashes/                           # Hashing / quantization
│   ├── asymmetric_hashing2/          # Asymmetric hashing v2
│   │   ├── searcher.h                # AH searcher
│   │   ├── querying.h                # LUT computation and distance lookup
│   │   ├── indexing.h                # Encoding database vectors
│   │   ├── training.h               # K-means training for codebooks
│   │   ├── training_model.cc         # Trained model persistence
│   │   └── training_options_base.h/.cc  # Training configuration
│   └── internal/                     # SIMD LUT16 kernels
│       ├── lut16_sse4.h              # SSE4 implementation
│       ├── lut16_avx512.h            # AVX-512 implementation
│       ├── lut16_avx512_swizzle.cc   # AVX-512 swizzle helpers
│       └── bazel_templates/          # Template files for batch-sharded kernels
│           ├── lut16_sse4.tpl.cc
│           ├── lut16_avx2.tpl.cc
│           ├── lut16_avx512_prefetch.tpl.cc
│           ├── lut16_avx512_smart.tpl.cc
│           ├── lut16_avx512_noprefetch.tpl.cc
│           └── lut16_highway.tpl.cc
│
├── partitioning/                     # Dataset partitioning (clustering)
│   ├── kmeans_tree_partitioner.h/.cc # K-means based partitioner
│   ├── partitioner_base.h           # Partitioner interface
│   ├── partitioner_factory_base.cc   # Partitioner construction
│   └── kmeans_tree_like_partitioner.h  # Abstract K-means partitioner
│
├── distance_measures/                # Distance computation
│   ├── distance_measure_base.h       # DistanceMeasure abstract class
│   ├── distance_measure_factory.h/.cc  # Factory: string name → DistanceMeasure
│   ├── distance_measures.h           # Convenience header (all measures)
│   ├── one_to_one/                   # Pairwise distance functions
│   │   ├── dot_product.h/.cc         # DotProductDistance
│   │   ├── l2_distance.h/.cc         # SquaredL2Distance
│   │   ├── cosine_distance.h/.cc     # CosineDistance
│   │   ├── l1_distance.h/.cc         # L1Distance
│   │   ├── hamming_distance.h/.cc    # HammingDistance
│   │   ├── limited_inner_product.h   # LimitedInnerProductDistance
│   │   ├── dot_product_avx1.cc       # AVX1-optimized dot product
│   │   ├── dot_product_sse4.h/.cc    # SSE4-optimized dot product
│   │   └── dot_product_highway.cc    # Highway-portable SIMD dot product
│   ├── one_to_many/                  # One query vs. many database vectors
│   │   └── one_to_many_*.cc
│   └── many_to_many/                 # Batch distance computation
│       └── many_to_many_*.cc
│
├── data_format/                      # Data structures
│   ├── datapoint.h/.cc               # DatapointPtr<T>, Datapoint<T>
│   ├── dataset.h/.cc                 # DenseDataset<T>, TypedDataset<T>
│   └── docid_collection_interface.h/.cc  # Document ID management
│
├── projection/                       # Dimensionality reduction / chunking
│   └── projection_base.h            # Projection interface
│
├── proto/                            # Protocol Buffer definitions
│   ├── scann.proto                   # Top-level ScannConfig
│   ├── partitioning.proto            # PartitioningConfig
│   ├── hash.proto                    # HashConfig, AsymmetricHasherConfig
│   ├── distance_measure.proto        # DistanceMeasureConfig
│   ├── exact_reordering.proto        # ExactReordering, FixedPoint
│   ├── brute_force.proto             # BruteForceConfig
│   ├── input_output.proto            # InputOutputConfig
│   ├── centers.proto                 # CentersForAllSubspaces
│   └── ...
│
├── trees/                            # Tree data structures
│   └── kmeans_tree/                  # K-means tree implementation
│
├── restricts/                        # Allowlist/denylist filtering
│   └── restrict_allowlist.h/.cc
│
├── metadata/                         # Metadata attachment
│   └── metadata_getter.h
│
├── utils/                            # Shared utilities (~70 files)
│   ├── types.h                       # DatapointIndex, DimensionIndex, NNResultsVector
│   ├── common.h                      # Common includes and aliases
│   ├── scann_config_utils.h/.cc      # Config parsing helpers
│   ├── reordering_helper.h/.cc       # ExactReorderingHelper, FixedPointReorderingHelper
│   ├── single_machine_autopilot.cc   # Automatic parameter tuning
│   ├── gmm_utils.h/.cc              # GMM / K-means training
│   ├── intrinsics/                   # CPU feature detection
│   │   ├── sse4.h
│   │   ├── avx1.h
│   │   ├── avx512.h (amx.h)
│   │   ├── flags.cc                  # Runtime ISA detection
│   │   └── highway.h (via hwy)
│   ├── linear_algebra/               # Eigen-based linear algebra
│   ├── fixed_point/                  # Fixed-point quantization helpers
│   ├── io_npy.h                      # NumPy file I/O
│   └── io_oss_wrapper.h/.cc          # File I/O utilities
│
├── oss_wrappers/                     # OS / platform abstraction
│   ├── scann_status.h                # Status macros (SCANN_RETURN_IF_ERROR, etc.)
│   ├── scann_threadpool.h            # Thread pool wrapper
│   ├── scann_aligned_malloc.h        # Aligned memory allocation
│   └── scann_serialize.h             # Serialization helpers
│
└── test/
    └── scann_ops-test.cc             # Comprehensive test suite
```

---

## Search Algorithm Strategies

The library supports five index strategies, each optimized for different accuracy/speed tradeoffs. The strategy is selected entirely by the `ScannConfig` protobuf.

### 1. Brute Force

**Class:** `BruteForceSearcher<T>`

The simplest strategy: compute exact distances between the query and every point in the dataset, then return the top-K closest. Optionally uses **int8 scalar quantization** (`ScalarQuantizedBruteForceSearcher`) to speed up distance computation.

**How scalar quantization works:**
- Each float dimension is linearly mapped to an int8 value in [-127, 127]
- Per-dimension multipliers store the scale factors
- Distance computation uses int8 dot products (much faster with SIMD)
- Per-datapoint squared L2 norms are precomputed for exact distance reconstruction

**Config:**
```protobuf
brute_force {
  fixed_point { enabled: true }  # Enable int8 quantization
}
```

**Complexity:** O(N * D) per query, where N = dataset size, D = dimensionality.

**Best for:** Small datasets (< 20K vectors) or when exact results are required.

### 2. Asymmetric Hashing (AH)

**Classes:** `asymmetric_hashing2::Searcher<T>`, `AsymmetricQueryer<T>`, `Indexer<T>`

Product quantization with asymmetric distance computation. The key insight is that database vectors are quantized (lossy), but query vectors are not — distances are computed between the original query and quantized database points ("asymmetric").

**Training phase:**
1. The D-dimensional vector space is chunked into B blocks of `num_dims_per_block` dimensions each (typically 2).
2. Within each block, K-means clustering is run on the projected sub-vectors to produce C centroids (typically `num_clusters_per_block = 16`).
3. Each database vector is encoded as B centroid indices (one per block), requiring B * log2(C) bits total.

**Query phase:**
1. For the query, compute a **Lookup Table (LUT)**: for each block b and each centroid c, precompute `distance(query_block_b, centroid_b_c)`. This is a B × C table.
2. The approximate distance to any database point is simply the sum of B lookups into this table.

**INT8_LUT16 optimization:**
With `num_clusters_per_block = 16`, each centroid index fits in 4 bits. The LUT values are quantized to int8. This enables extremely fast SIMD computation using `pshufb` (packed shuffle bytes) instructions: a single SIMD instruction can perform 16 simultaneous table lookups, and accumulation is done with integer addition.

**Config:**
```protobuf
hash {
  asymmetric_hash {
    lookup_type: INT8_LUT16
    num_clusters_per_block: 16
    projection {
      input_dim: <D>
      projection_type: CHUNK
      num_blocks: <D/2>
      num_dims_per_block: 2
    }
  }
}
```

**Complexity:** O(N * B) per query (B lookups per database point, where B = D/2).

**Best for:** Medium-sized datasets where approximate results are acceptable.

### 3. Tree-AH Hybrid with Residual Quantization

**Class:** `TreeAHHybridResidual`

The most sophisticated and generally highest-performing strategy. Combines K-means tree partitioning with asymmetric hashing and residual quantization.

**How it works:**

1. **Partitioning (offline):** The dataset is partitioned into ~100 clusters using K-means. Each database vector is assigned to its nearest cluster center.

2. **Residual computation:** Instead of storing the original vectors, the system stores the **residual** (difference between the vector and its cluster center). Residuals have smaller magnitude and are more uniform, leading to better quantization accuracy.

3. **Per-partition AH searchers:** Each partition gets its own asymmetric hashing searcher, applied to the residuals within that partition.

4. **Query processing:**
   - The query is compared against all cluster centers to find the closest L centers ("leaves to search").
   - For each selected partition, the query residual (query minus center) is used to build a partition-specific LUT.
   - AH search is run within each selected partition.
   - Results from all partitions are merged using a **global top-N** structure.

5. **Query spilling:** The query searches multiple partitions (controlled by `max_spill_centers`) to avoid missing near-boundary neighbors.

6. **Global top-N:** When enabled, results from all partitions are tracked in a single global priority queue rather than per-partition queues. This ensures the overall best K results are found even when they are spread across partitions. Enabled by packing the partition index and intra-partition index into a single 32-bit integer.

**Config:**
```protobuf
partitioning {
  num_children: 100
  min_cluster_size: 20
  query_spilling { spilling_type: FIXED_NUMBER_OF_CENTERS  max_spill_centers: 20 }
  partitioning_type: GENERIC
}
hash {
  asymmetric_hash {
    lookup_type: INT8_LUT16
    use_residual_quantization: true
    use_global_topn: true
    num_clusters_per_block: 16
  }
}
```

**Complexity:** O(L * (N/P) * B) per query, where L = leaves searched, P = number of partitions, B = blocks per vector.

**Best for:** Large datasets (100K+ vectors) where speed is critical.

### 4. Tree-Brute Force Hybrid

**Class:** `TreeXHybridSMMD<float>` with `ScalarQuantizedBruteForceSearcher` leaf searchers

Same K-means tree partitioning as Tree-AH, but each leaf uses scalar-quantized brute-force search instead of asymmetric hashing. This is simpler and has better recall than AH at the cost of somewhat slower search.

**Config:**
```protobuf
partitioning {
  num_children: 100
  min_cluster_size: 10
  query_spilling { spilling_type: FIXED_NUMBER_OF_CENTERS  max_spill_centers: 10 }
}
brute_force {
  fixed_point { enabled: true }
}
```

**Complexity:** O(L * (N/P) * D) per query with int8 acceleration.

**Best for:** Medium-large datasets where higher recall is needed than Tree-AH provides.

### 5. AH with Exact Reordering

**Classes:** `asymmetric_hashing2::Searcher<T>` + `ExactReorderingHelper<T>`

AH is used as a **coarse pass** to retrieve a larger candidate set (`approx_num_neighbors`), then **exact distance computation** is used to rerank and return the final top-K. This achieves near-exact recall with AH-level speed for the initial candidate retrieval.

**Config:**
```protobuf
hash {
  asymmetric_hash { ... }
}
exact_reordering {
  approx_num_neighbors: 40  # Retrieve 40 candidates with AH
  fixed_point { enabled: false }  # Use float reordering
}
```

**Complexity:** O(N * B + R * D) per query, where R = `approx_num_neighbors`.

**Best for:** When you need near-exact results but the dataset is too large for full brute force.

---

## Configuration System

### ScannConfig Protobuf

All algorithm selection is driven by the `ScannConfig` protobuf (`proto/scann.proto`). The key fields are:

| Field | Proto Type | Purpose |
|---|---|---|
| `num_neighbors` | `int32` | Default K for search |
| `distance_measure` | `DistanceMeasureConfig` | Distance function name |
| `brute_force` | `BruteForceConfig` | Enables brute-force search |
| `partitioning` | `PartitioningConfig` | Enables tree partitioning |
| `hash` | `HashConfig` | Enables asymmetric hashing |
| `exact_reordering` | `ExactReordering` | Enables exact reranking |
| `input_output` | `InputOutputConfig` | Dimensionality, data types |

The factory function `SingleMachineFactoryScann<float>()` reads this config and instantiates the appropriate composition of searchers.

### Preset Configuration Builders

The wrapper provides five convenience functions that generate text-format `ScannConfig` strings:

| Function | Strategy |
|---|---|
| `ScannBruteForceConfig(k, dim)` | Scalar-quantized brute force |
| `ScannAhConfig(k, dim)` | Asymmetric hashing |
| `ScannTreeAhConfig(k, dim)` | Tree + AH + residual quantization |
| `ScannTreeBruteForceConfig(k, dim)` | Tree + scalar-quantized brute force |
| `ScannReorderConfig(k, dim)` | AH + exact reordering |

All use `DotProductDistance` as the distance measure by default.

---

## Core Type System

### Data Types

**`DatapointIndex`** (`uint32_t`) — Index of a vector in the dataset. Maximum ~2 billion datapoints.

**`DimensionIndex`** (`uint64_t`) — Index of a dimension within a vector.

**`NNResultsVector`** — `std::vector<std::pair<DatapointIndex, float>>` — Raw search results as (index, distance) pairs.

**`ScannSearchResult`** — The wrapper's public result type:
```cpp
struct ScannSearchResult {
  int32_t index;
  float distance;
};
```

**`DatapointPtr<T>`** — A non-owning pointer to a vector. Can represent both dense and sparse vectors. Carries a pointer to the values, dimensionality, and optional indices for sparse data.

**`Datapoint<T>`** — An owning version of `DatapointPtr<T>`.

**`DenseDataset<T>`** — A collection of dense vectors stored in a contiguous row-major buffer. Provides random access via `operator[]` returning `DatapointPtr<T>`.

**`SingleMachineFactoryOptions`** — A bundle of all pre-computed artifacts needed to construct a searcher:
```cpp
struct SingleMachineFactoryOptions {
  shared_ptr<vector<vector<DatapointIndex>>> datapoints_by_token;  // Partition assignments
  shared_ptr<PreQuantizedFixedPoint> pre_quantized_fixed_point;     // Int8 quantized data
  shared_ptr<DenseDataset<uint8_t>> hashed_dataset;                 // AH-encoded data
  shared_ptr<CentersForAllSubspaces> ah_codebook;                   // AH cluster centers
  shared_ptr<SerializedPartitioner> serialized_partitioner;         // K-means tree
  shared_ptr<DenseDataset<int16_t>> bfloat16_dataset;               // BFloat16 data
  shared_ptr<ThreadPool> parallelization_pool;                      // Thread pool for training
};
```

### Distance Measures

The `DistanceMeasure` abstract class supports both dense and sparse vectors with the following concrete implementations:

| Name String | Class | Description | Direction |
|---|---|---|---|
| `"DotProductDistance"` | `DotProductDistance` | Negated dot product | Lower = more similar |
| `"SquaredL2Distance"` | `SquaredL2Distance` | Squared Euclidean | Lower = more similar |
| `"CosineDistance"` | `CosineDistance` | 1 - cosine similarity | Lower = more similar |
| `"L1Distance"` | `L1Distance` | Manhattan distance | Lower = more similar |
| `"L2Distance"` | `L2Distance` | Euclidean distance | Lower = more similar |
| `"LimitedInnerProductDistance"` | `LimitedInnerProductDistance` | Bounded inner product | Lower = more similar |
| `"GeneralHammingDistance"` | `GeneralHammingDistance` | Hamming distance | Lower = more similar |

For `DotProductDistance`, the library stores `-dot_product` internally so that "smaller distance = more similar" holds universally. A `result_multiplier_` of -1 flips the sign back when returning results to the caller.

Each distance measure has SIMD-optimized implementations for one-to-one, one-to-many, and many-to-many computation, with specializations for SSE4, AVX1, AVX2, AVX-512, and Highway (portable SIMD).

---

## Search Pipeline

### Single Query Flow

```
ScannWrapper::Search(query, final_nn, pre_reorder_nn, leaves)
    │
    ▼
ImplSearch() [scann_wrapper_impl.cc]
    │  wraps float* into DatapointPtr<float>
    ▼
ScannInterface::Search(query, res, final_nn, pre_reorder_nn, leaves)
    │  builds SearchParameters
    │  validates dimensionality
    ▼
scann_->FindNeighbors(query, params, result)
    │
    ├── [BruteForceSearcher] exact distance to all N points → top-K
    │
    ├── [AH Searcher] build LUT → approximate distance to all N → top-K
    │
    └── [TreeAHHybridResidual]
           │  tokenize query → select L partitions
           │  for each partition:
           │    compute residual query
           │    build partition-specific LUT
           │    AH search within partition
           │  merge results across partitions (global top-N)
           ▼
        NNResultsVector (index, distance) pairs
    │
    ▼  [If exact_reordering enabled]
ReorderingHelper::ComputeDistancesForReordering()
    │  recompute exact distances for candidates
    │  sort and truncate to final_nn
    ▼
ConvertResults() → vector<ScannSearchResult>
    │
    ▼
ImplToYbStatus() → yb::Status / yb::Result<>
```

### Batched Search

`SearchBatched` processes multiple queries sequentially in a single thread, but takes advantage of batched internal optimizations:

- `BruteForceSearcher` uses optimized many-to-many distance computation
- The optimal batch size varies by strategy: 256 for brute force, 16 for AH, 1 for tree-partitioned

### Parallel Batched Search

`SearchBatchedParallel` splits queries into chunks and processes them in parallel using an internal thread pool:

```cpp
const size_t kBatchSize = min(max(min_batch_size_, numQueries / numCPUs), batch_size);
ParallelForWithStatus(numBatches, pool, [&](size_t i) {
    // slice queries[begin..begin+curSize] and call SearchBatched
});
```

The thread pool is created at initialization time with `NumCPUs - 1` threads and can be reconfigured via `SetNumThreads()`.

### Search Parameters

`SearchParameters` controls per-query behavior:

| Parameter | Meaning |
|---|---|
| `pre_reordering_num_neighbors` | Number of candidates to retrieve in the approximate phase |
| `post_reordering_num_neighbors` | Number of final results after exact reordering (-1 = disabled) |
| `searcher_specific_optional_parameters` | E.g., `TreeXOptionalParameters` to override number of partitions to search |

The `leaves` parameter in the wrapper API maps to `TreeXOptionalParameters::num_partitions_to_search_override`, controlling how many K-means tree partitions are searched per query.

---

## Index Construction and Training

### Factory System

The `SingleMachineFactoryScann<float>(config, dataset, opts)` function is the central factory. It reads the `ScannConfig` and:

1. Creates the appropriate partitioner (if `partitioning` is set)
2. Trains AH codebooks (if `hash` is set)
3. Quantizes the dataset
4. Builds the searcher hierarchy
5. Optionally attaches a reordering helper (if `exact_reordering` is set)

### K-Means Tree Training

When `partitioning` is configured:

1. `KMeansTreePartitioner<float>` is created with the specified distance measures for database tokenization and query tokenization (these can differ).
2. `CreatePartitioning()` runs K-means with the configured parameters:
   - `num_children`: Number of clusters (partitions)
   - `max_clustering_iterations`: Max K-means iterations
   - `single_machine_center_initialization`: K-means++ or random initialization
   - `min_cluster_size`: Minimum points per cluster
3. The resulting cluster centers define the partitioner, which is used to assign database vectors to partitions (tokenization) and to route queries to relevant partitions at search time.

**Query spilling** is configured separately:
- `FIXED_NUMBER_OF_CENTERS`: Always search a fixed number of partitions
- `MULTIPLICATIVE`: Search partitions within a distance multiplier of the closest
- `ADDITIVE`: Search partitions within an additive distance threshold

### Asymmetric Hashing Training

When `hash.asymmetric_hash` is configured:

1. `TrainSingleMachine<T>()` is called with the dataset and training options.
2. The projection (`CHUNK` type) splits each D-dimensional vector into B blocks of `num_dims_per_block` dimensions.
3. K-means is run independently within each block to learn `num_clusters_per_block` centroids.
4. The resulting `Model<T>` (codebook) is used for both indexing (encoding database vectors) and querying (building LUTs).

The training supports three quantization schemes:
- **PRODUCT** (default): Standard product quantization
- **STACKED**: Stacked quantizers for better reconstruction
- **PRODUCT_AND_BIAS**: Product quantization with a learned bias term

---

## Mutation (Insert / Delete / Update)

Each searcher type implements the `Mutator` interface for live index updates:

### Mutator Interface

```cpp
class Mutator {
  StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                         string_view docid, const MutationOptions& mo);
  Status RemoveDatapoint(string_view docid);
  Status RemoveDatapoint(DatapointIndex index);
  StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                            string_view docid, const MutationOptions& mo);
};
```

### Per-Searcher Mutator Behavior

**`BruteForceSearcher::Mutator`:**
- `AddDatapoint`: Appends to the dataset, updates docid collection
- `RemoveDatapoint`: Swaps with last element, shrinks dataset

**`ScalarQuantizedBruteForceSearcher::Mutator`:**
- Additionally quantizes the new point to int8 and updates the quantized dataset
- Updates squared L2 norms

**`TreeXHybridMutator<T>`:**
- Must tokenize the new point (assign to a partition)
- Add to the correct leaf searcher
- For `TreeAHHybridResidual`: compute residual, hash, and insert into the partition's AH searcher

**Important caveat** (observed in tests): The fixed-point brute force path pre-quantizes the entire dataset at build time. Dynamically inserted points are not included in that quantized representation, making them invisible to search. For dynamic workloads, use plain brute force without `fixed_point: true`.

### Wrapper API

The wrapper exposes simplified insert/delete:
```cpp
Result<int32_t> Insert(const vector<float>& datapoint, const string& docid);
Status Delete(const string& docid);
Status Delete(int32_t index);
```

These obtain a `Mutator` via `scann.GetMutator()` and delegate to `AddDatapoint` / `RemoveDatapoint`.

---

## Serialization and Persistence

### Artifact Types

The `ScannAssets` protobuf (`scann_ops/scann_assets.proto`) tracks all serialized files:

| AssetType | File | Format | Description |
|---|---|---|---|
| `AH_CENTERS` | `ah_codebook.pb` | Binary protobuf | AH cluster centers (codebook) |
| `PARTITIONER` | `serialized_partitioner.pb` | Binary protobuf | K-means tree structure |
| `TOKENIZATION_NPY` | `datapoint_to_token.npy` | NumPy int32 | Partition assignment per datapoint |
| `AH_DATASET_NPY` | `hashed_dataset.npy` | NumPy uint8 | AH-encoded database vectors |
| `AH_DATASET_SOAR_NPY` | `hashed_dataset_soar.npy` | NumPy uint8 | SOAR secondary hashed data |
| `DATASET_NPY` | `dataset.npy` | NumPy float32 | Original float vectors (for reordering) |
| `INT8_DATASET_NPY` | `int8_dataset.npy` | NumPy int8 | Scalar-quantized database |
| `INT8_MULTIPLIERS_NPY` | `int8_multipliers.npy` | NumPy float32 | Per-dimension quantization scales |
| `INT8_NORMS_NPY` | `dp_norms.npy` | NumPy float32 | Per-datapoint squared L2 norms |
| `BF16_DATASET_NPY` | `bfloat16_dataset.npy` | NumPy int16 | BFloat16 dataset |
| — | `scann_config.pb` | Binary protobuf | The `ScannConfig` used to build the index |
| — | `scann_assets.pbtxt` | Text protobuf | Manifest listing all assets |

### Serialization Flow

```
ScannWrapper::Serialize(path)
    └── ScannInterface::Serialize(path)
        ├── ExtractSingleMachineFactoryOptions() from searcher
        ├── WriteProtobufToFile(scann_config.pb)
        ├── For each non-null option:
        │   ├── ah_codebook → ah_codebook.pb
        │   ├── serialized_partitioner → serialized_partitioner.pb
        │   ├── datapoints_by_token → datapoint_to_token.npy
        │   ├── hashed_dataset → hashed_dataset.npy
        │   ├── pre_quantized_fixed_point → int8_dataset.npy, int8_multipliers.npy, dp_norms.npy
        │   └── Float32DatasetIfNeeded() → dataset.npy
        └── Write scann_assets.pbtxt manifest
```

### Deserialization Flow

```
ScannWrapper::LoadFromDisk(artifacts_dir)
    └── ScannInterface::LoadArtifacts(artifacts_dir)
        ├── Read scann_config.pb
        ├── Read scann_assets.pbtxt
        ├── Rewrite relative paths to absolute
        ├── Sort assets in dependency order:
        │   PARTITIONER first → TOKENIZATION → everything else
        ├── Load each asset:
        │   ├── AH_CENTERS → opts.ah_codebook
        │   ├── PARTITIONER → opts.serialized_partitioner
        │   ├── TOKENIZATION_NPY → opts.datapoints_by_token
        │   ├── AH_DATASET_NPY → opts.hashed_dataset
        │   ├── DATASET_NPY → dataset (shared_ptr)
        │   ├── INT8_* → opts.pre_quantized_fixed_point
        │   └── BF16_DATASET_NPY → opts.bfloat16_dataset
        └── Return (config, dataset, opts) tuple → Initialize()
```

---

## SIMD-Accelerated Kernels

The `hashes/internal/` directory contains the performance-critical lookup table distance computation. The LUT16 (4-bit index, 16 centroids) kernels use SIMD shuffle instructions to perform 16 table lookups simultaneously.

### Template-Generated Batch Kernels

The CMake build generates batch-specific kernels from `.tpl.cc` templates by substituting `{BATCH_SIZE}` with values 1 through 9. This avoids runtime branching on batch size:

```
lut16_sse4.tpl.cc    → lut16_sse4_batches_1.cc ... lut16_sse4_batches_9.cc
lut16_avx2.tpl.cc    → lut16_avx2_batches_1.cc ... lut16_avx2_batches_9.cc
lut16_avx512_prefetch.tpl.cc → ...
lut16_avx512_smart.tpl.cc → ...
lut16_avx512_noprefetch.tpl.cc → ...
lut16_highway.tpl.cc → lut16_highway_batches_1.cc ... lut16_highway_batches_9.cc
```

Total: 6 ISA targets × 9 batch sizes = **54 generated source files**.

### ISA Targets

| Target | Instructions Used | Availability |
|---|---|---|
| SSE4 | `pshufb`, `paddb`, `pmaddubsw` | All x86-64 CPUs |
| AVX2 | 256-bit versions of SSE4 ops | Haswell+ (2013+) |
| AVX-512 (prefetch) | 512-bit ops + SW prefetch | Skylake-SP+ |
| AVX-512 (smart) | Adaptive prefetch strategy | Skylake-SP+ |
| AVX-512 (no prefetch) | 512-bit ops, no prefetch | Skylake-SP+ |
| Highway | Portable SIMD via Google Highway | Any platform (ARM NEON, RISC-V V, etc.) |

Runtime ISA detection (`utils/intrinsics/flags.cc`) selects the best available kernel at initialization.

All batch-sharded sources and `lut16_avx512_swizzle.cc` are compiled at **`-O3`** even in debug builds to prevent stack overflows from large unoptimized stack frames.

---

## Reordering

Reordering is a two-pass search strategy:

1. **Approximate pass**: Use a fast approximate method (AH, scalar quantization) to retrieve R candidate neighbors (where R > K).
2. **Exact pass**: Compute exact distances for the R candidates and return the top-K.

The `ExactReorderingHelper<T>` computes exact float distances using the original dataset. The `FixedPointFloatDenseSquaredL2ReorderingHelper` uses int8-quantized vectors for the reranking pass (faster but still slightly approximate).

The reordering helper is attached to any searcher via `EnableReordering()`, and the base class `FindNeighbors()` automatically invokes reordering after the approximate search if enabled.

**Config parameters:**
- `approx_num_neighbors`: Size of the candidate set from the approximate pass
- `fixed_point.enabled`: Whether to use int8 quantized reordering instead of float exact reordering
- `bfloat16.enabled`: Whether to use BFloat16 reordering

---

## Searcher Class Hierarchy

```
UntypedSingleMachineSearcherBase
    │
    └── SingleMachineSearcherBase<T>
            │
            ├── BruteForceSearcher<T>
            │     Direct exhaustive search
            │
            ├── ScalarQuantizedBruteForceSearcher (T=float only)
            │     Int8-quantized brute force
            │
            ├── TreeXHybridSMMD<T>
            │     Generic tree-partitioned hybrid
            │     Contains: vector<unique_ptr<SingleMachineSearcherBase<T>>> leaf_searchers_
            │     Used by: Tree-BruteForce, Tree-SQ configs
            │
            ├── TreeAHHybridResidual (T=float only)
            │     Tree + AH + residual quantization
            │     Contains: vector<unique_ptr<asymmetric_hashing2::SearcherBase<float>>> leaf_searchers_
            │     Specialized for LUT16 global top-N
            │
            └── asymmetric_hashing2::Searcher<T>
                  Standalone AH searcher (also used as leaf in tree hybrids)
```

Each searcher implements:
- `FindNeighborsImpl()` — single-query search
- `FindNeighborsBatchedImpl()` — multi-query search
- `GetMutator()` — returns a `Mutator` for live updates
- `ExtractSingleMachineFactoryOptions()` — extracts all artifacts for serialization

---

## Build System

The `CMakeLists.txt` builds everything into a single `scann` library target.

### Dependencies

| Library | Purpose |
|---|---|
| `protobuf` | Configuration and serialization protos |
| `abseil` | Containers, status types, threading |
| `cnpy` | NumPy file format I/O (`.npy` files) |
| `hwy` | Google Highway portable SIMD library |
| `hwy_contrib` | Highway contributed algorithms |

### Build Steps

1. **Proto generation**: 16 `.proto` files → C++ code via `PROTOBUF_GENERATE_CPP`
2. **Batch kernel generation**: 6 `.tpl.cc` templates × 9 batch sizes = 54 `.cc` files
3. **Source collection**: All `.cc` files under `src/yb/scann/` except:
   - `scann_ops/cc/ops/` (TensorFlow ops)
   - `scann_ops/cc/kernels/` (TF kernels)
   - `scann_ops/cc/python/` (Python bindings)
   - `test/` directory
   - `.tpl.cc` template files
4. **Compilation**: With extensive warning suppressions for upstream code
5. **Optimization**: LUT16 kernels forced to `-O3` even in debug

### Test Target

```cmake
ADD_YB_TEST(test/scann_ops-test)
```

Links against `scann` and `YB_MIN_TEST_LIBS`.

---

## Testing

The test suite (`test/scann_ops-test.cc`) covers all major functionality through the `ScannWrapper` API:

| Test | What It Validates |
|---|---|
| `AhSerialization` | Serialize + deserialize AH index, results match |
| `TreeAhSerialization` | Serialize + deserialize Tree-AH index |
| `TreeBruteForceSerialization` | Serialize + deserialize Tree-BruteForce index |
| `BruteForceInt8Serialization` | Serialize + deserialize int8 brute force index |
| `ReorderingSerialization` | Serialize + deserialize AH + reordering index |
| `BruteForceMatchesReference` | Brute force results match a C++ reference implementation |
| `BruteForceFinalNumNeighbors` | Correct number of neighbors returned |
| `SingleQuerySearch` | Single-query API works correctly |
| `ParallelMatchesSequential` | Parallel search results match sequential search |
| `ReorderingShapes` | Various (final_nn, pre_reorder_nn) combinations return correct shapes |
| `NPointsAndDimensionality` | Metadata accessors return correct values |
| `MoveConstructor` | Move semantics work correctly |
| `MoveAssignment` | Move assignment works correctly |
| `InsertIncreasesNPoints` | Insert increases dataset size |
| `InsertedPointIsSearchable` | Inserted points appear in search results |
| `InsertMultipleThenBatchSearch` | Multiple inserts + batch search |
| `DeleteByDocidDecreasesNPoints` | Delete by docid works |
| `DeleteByIndexDecreasesNPoints` | Delete by numeric index works |
| `DeletedPointNotReturnedBySearch` | Deleted points no longer appear in results |

All tests use random datasets (10K points, 32 dimensions) with fixed seeds for reproducibility.

---

## Key Design Decisions

1. **Header isolation via double-PIMPL**: The most distinctive architectural choice. The `ImplStatus` POD + opaque struct + free-function bridge ensures that `scann_wrapper_impl.cc` is the **only** translation unit that includes both ScaNN and abseil headers. This prevents macro conflicts that would otherwise make compilation impossible.

2. **Configuration-driven algorithm selection**: All algorithm choices are encoded in the `ScannConfig` protobuf. The factory pattern means adding a new search strategy requires no API changes — just a new config.

3. **Composable searcher hierarchy**: `TreeXHybridSMMD<T>` owns per-partition `SingleMachineSearcherBase<T>` instances. Any searcher type can serve as a leaf: brute force, scalar quantized, AH, etc. The `ReorderingHelper` wraps any searcher with exact reranking, orthogonal to the underlying approximate method.

4. **NumPy serialization format**: Using `.npy` files (via the `cnpy` library) for large array artifacts makes the serialized index interoperable with Python tools and easy to inspect for debugging.

5. **Move-only semantics**: `ScannWrapper` is move-only (non-copyable), reflecting the fact that the underlying ScaNN searcher owns substantial resources and is not designed for copying.

6. **Thread pool management**: A dedicated "ScannQueryingPool" is created at initialization with `NumCPUs - 1` threads. Training uses a separate "scann_threadpool". This separation prevents training work from interfering with query processing.

7. **SIMD kernel generation**: The template-based batch-sharded code generation (6 ISA × 9 batch sizes) is ported from the upstream Bazel build. It ensures that the innermost distance computation loops are fully specialized and inlineable for each batch size, maximizing SIMD utilization.
