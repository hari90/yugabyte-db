# Unused Code Report for src/yb

Analysis date: 2026-03-20

## 1. .cc Files Not Built (Not Referenced in Any CMakeLists.txt)

### 1.1 Dead Source File (YugabyteDB-owned)

| File | Notes |
|------|-------|
| `src/yb/yql/pggate/pg_type.cc` | **Dead code.** Defines methods (`PgTypeInfo::PgTypeInfo(const YbcPgTypeEntity*, int)`, `PgTypeInfo::GetTypeEntity`) using a member `type_map_` that no longer exists in the header class. The header (`pg_type.h`) was rewritten with inline methods using different member names (`map_`, `Find`) but the .cc file was never removed. Not in `PGGATE_SRCS`. |

### 1.2 RocksDB Files Not Built (forked third-party code)

These are RocksDB files that exist in the codebase but are not built. Since RocksDB is forked
third-party code, these are lower priority but still represent dead code.

**Tests/benchmarks not in CMakeLists.txt:**

| File | Type |
|------|------|
| `src/yb/rocksdb/db/compaction_iterator_test.cc` | Test |
| `src/yb/rocksdb/db/memtable_list_test.cc` | Test |
| `src/yb/rocksdb/db/merge_helper_test.cc` | Test |
| `src/yb/rocksdb/utilities/env_mirror_test.cc` | Test |
| `src/yb/rocksdb/utilities/merge_operators/string_append/stringappend_test.cc` | Test |
| `src/yb/rocksdb/db/forward_iterator_bench.cc` | Bench |
| `src/yb/rocksdb/db/memtablerep_bench.cc` | Bench |
| `src/yb/rocksdb/table/table_reader_bench.cc` | Bench |
| `src/yb/rocksdb/util/cache_bench.cc` | Bench |
| `src/yb/rocksdb/util/log_write_bench.cc` | Bench |

**Example programs (never built):**

| File |
|------|
| `src/yb/rocksdb/examples/column_families_example.cc` |
| `src/yb/rocksdb/examples/compact_files_example.cc` |
| `src/yb/rocksdb/examples/compaction_filter_example.cc` |
| `src/yb/rocksdb/examples/options_file_example.cc` |
| `src/yb/rocksdb/examples/simple_example.cc` |

**Windows-only files (not applicable to Linux builds):**

| File |
|------|
| `src/yb/rocksdb/port/win/port_win.cc` |
| `src/yb/rocksdb/port/win/win_logger.cc` |

---

## 2. Unreferenced Header Files

### 2.1 Headers with completely unused classes/types (HIGH confidence, safe to remove)

| File | What it defines | Evidence |
|------|----------------|----------|
| `src/yb/common/columnblock.h` | `ColumnBlock`, `ColumnBlockCell` classes | Not included by any file. Types not referenced anywhere. |
| `src/yb/common/rowid.h` | `rowid_t` typedef, `EncodeRowId`/`DecodeRowId` functions | Not included by any file. Type not referenced anywhere. |
| `src/yb/util/auto_release_pool.h` | `AutoReleasePool` class | Not included by any file. Class not referenced anywhere. |
| `src/yb/util/debug_ref_counted.h` | `DebugRefCountedThreadSafe` template class | Not included by any file. Class not referenced anywhere. |
| `src/yb/util/bit-stream-utils.inline.h` | `BitWriter`/`BitReader` inline implementations | Only included by itself (includes bit-stream-utils.h). Neither BitWriter nor BitReader are used anywhere. |
| `src/yb/rpc/lw_pb_conversion.h` | Lightweight protobuf conversion helpers | Not included by any file. |
| `src/yb/yql/pggate/pg_coldesc.h` | `ColumnDesc` class for pggate | Not included by any file. |
| `src/yb/yql/pggate/type_mapping.h` | C API to C++ type mapping templates | Not included by any file. |
| `src/yb/yql/cql/ql/parser/parser_inactive_nodes.h` | Inactive CQL parser node types (237 lines) | Not included by any file. Self-contained dead code. |
| `src/yb/gutil/fixedarray.h` | `FixedArray` container class | Not included by any file. Class not referenced anywhere. |
| `src/yb/gutil/move.h` | Move semantics macros (pre-C++11 legacy) | Not included by any file. |
| `src/yb/gutil/paranoid.h` | `GOOGLE_PARANOID` assertion macros | Not included by any file. Macros not used anywhere. |

### 2.2 Headers with marginal/test-only usage (MEDIUM confidence)

| File | What it defines | Evidence |
|------|----------------|----------|
| `src/yb/server/mock_hybrid_clock.h` | `MockHybridClock` test class | Only used in `hybrid_clock-test.cc`. Could be moved inline to the test. |
| `src/yb/util/protobuf-annotations.h` | Protobuf annotation macros | Only included via generated code from `protoc-gen-insertions.cc`. May be unused if no protobuf insertion points exist. |

### 2.3 Platform/architecture-specific headers (conditionally included)

| File | Notes |
|------|-------|
| `src/yb/gutil/auxiliary/atomicops-internals-arm-generic.h` | ARM-generic atomics (file is in `auxiliary/` but include paths reference `gutil/` directly — path mismatch) |
| `src/yb/gutil/auxiliary/atomicops-internals-arm-v6plus.h` | ARM v6+ atomics (same path mismatch issue) |
| `src/yb/gutil/auxiliary/atomicops-internals-windows.h` | Windows atomics (Linux-only build) |
| `src/yb/gutil/valgrind.h` | Valgrind annotations | Only mentioned in comments within `dynamic_annotations.h`, never actually `#include`'d. |
| `src/yb/gutil/utf/utfdef.h` | UTF definitions | Included from `rune.c` (a .c file, technically used). |

### 2.4 RocksDB unused headers (forked third-party, lower priority)

| File | Notes |
|------|-------|
| `src/yb/rocksdb/port/port_example.h` | Documentation/example header, never included |
| `src/yb/rocksdb/precompiled_header.h` | Unused precompiled header |
| `src/yb/rocksdb/table/forwarding_iterator.h` | `ForwardingIterator` class — not used anywhere |
| `src/yb/rocksdb/util/channel.h` | Channel utility — not included anywhere |
| `src/yb/rocksdb/utilities/info_log_finder.h` | Has an implementation .cc but header is unused by other code |
| `src/yb/rocksdb/utilities/optimistic_transaction_db.h` | Referenced in `options.h` comments but appears unused in practice |

### 2.5 PCH (precompiled header) files — used implicitly by build system

32 `*_pch.h` files exist and are used by the CMake PCH mechanism. These are NOT dead code.

---

## 3. Potentially Unused Functions/Classes

### 3.1 Dead code confirmed (entire files unused)

| Item | Location | Details |
|------|----------|---------|
| `PgTypeInfo` old implementation | `src/yb/yql/pggate/pg_type.cc` | Entire file is dead. Defines constructor and `GetTypeEntity()` using non-existent member `type_map_`. Header was rewritten with different API. |
| `ColumnBlock` class | `src/yb/common/columnblock.h` | Entire class unused. Legacy from Apache Kudu codebase. |
| `rowid_t` and helpers | `src/yb/common/rowid.h` | Type and serialization functions unused. Legacy from Kudu. |
| `AutoReleasePool` class | `src/yb/util/auto_release_pool.h` | Memory pool class unused anywhere. |
| `DebugRefCountedThreadSafe` | `src/yb/util/debug_ref_counted.h` | Debug ref-counting template unused anywhere. |
| `BitWriter`/`BitReader` | `src/yb/util/bit-stream-utils.inline.h` | Bit stream utilities unused anywhere. |
| `FixedArray` class | `src/yb/gutil/fixedarray.h` | Stack-allocated array class unused. |
| `ColumnDesc` class | `src/yb/yql/pggate/pg_coldesc.h` | PgGate column descriptor class unused. |
| Type mapping templates | `src/yb/yql/pggate/type_mapping.h` | C API/C++ type mapping templates unused. |
| Inactive parser nodes | `src/yb/yql/cql/ql/parser/parser_inactive_nodes.h` | 237 lines of dead CQL parser node definitions. |

---

## Summary

| Category | Count |
|----------|-------|
| Dead .cc files (YB-owned, not built) | 1 |
| Dead .cc files (RocksDB, not built) | 17 |
| Unreferenced .h files (high confidence, safe to remove) | 12 |
| Unreferenced .h files (RocksDB) | 6 |
| Unreferenced .h files (platform-conditional) | 5 |
| Unused classes/types (confirmed dead) | 10 |

### Recommended Cleanup Priority

1. **Immediate:** Remove `src/yb/yql/pggate/pg_type.cc` — completely dead, won't compile if built
2. **High:** Remove the 12 high-confidence unreferenced headers — they define classes/types never used
3. **Medium:** Clean up the 5 platform-conditional headers (verify ARM/Windows paths are truly dead)
4. **Low:** Clean up RocksDB dead files (forked code, may diverge from upstream)
