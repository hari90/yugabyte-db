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

## 3. Unused Functions and Methods

### 3.1 Functions Declared in Headers with NO Definition (definitively dead)

These are declared but have no corresponding definition anywhere — safe to remove.

| Function | Declared in | Notes |
|----------|-------------|-------|
| `RegisterGLogMetrics` | `src/yb/server/glog_metrics.h:64` | Free function, no definition exists |
| `HandleTabletServersPage` | `src/yb/master/master-path-handlers.h:335` | Free function, no definition exists |
| `EncodeFixed32` | `src/yb/util/coding.h:70` | Declared `extern`, no definition. Only `InlineEncodeFixed32` exists |
| `EncodeFixed64` | `src/yb/util/coding.h:71` | Declared `extern`, no definition. Only `InlineEncodeFixed64` exists |
| `EncodeVarint32` | `src/yb/util/coding.h:76` | Declared `extern`, no definition, no callers |
| `FlushOutboundQueueAborted` | `src/yb/rpc/rpc_with_queue.h:104` | Class method declared, no definition |
| `Tablet::KeyValueBatchFromRedisWriteBatch` | `src/yb/tablet/tablet.h:370` | Class method declared, no definition |
| `Tablet::PreparePgsqlWriteOperations` | `src/yb/tablet/tablet.h:414` | Class method declared, no definition |
| `Tablet::KeyValueBatchFromPgsqlWriteBatch` | `src/yb/tablet/tablet.h:415` | Class method declared, no definition |
| `Tablet::UpdateHistoryCutoff` | `src/yb/tablet/tablet.h:572` | Class method declared, no definition |
| `RaftGroupMetadata::ReadSuperBlock` | `src/yb/tablet/tablet_metadata.h:789` | Class method declared, no definition. Only `ReadSuperBlockFromDisk` exists |
| `TableInfo::SetMatview` | `src/yb/master/catalog_entity_info.h:920` | Class method declared, no definition |

### 3.2 Functions Declared and Defined but Never Called

These have both declaration and definition but zero callers in production code.

| Function | Header | Definition | Notes |
|----------|--------|------------|-------|
| `PutFixed64` | `util/coding.h:39` | `util/coding.cc:48` | `InlinePutFixed64` is used instead |
| `PutLengthPrefixedSlice` | `util/coding.h:46` | `util/coding.cc:63` | No callers |
| `PutFixed32LengthPrefixedSlice` | `util/coding.h:50` | `util/coding.cc:68` | No callers |
| `VarintLength` | `util/coding.h:66` | `util/coding.cc:73` | No callers |
| `ShutdownLoggingSafe` | `util/logging.h:305` | `util/logging.cc:397` | No callers |
| `DoNothingStatusClosure` | `util/status_callback.h:61` | `util/status_callback.cc:40` | No callers |
| `GetLogFormatStackTraceHex` | `util/debug-util.h:96` | `util/debug-util.cc:158` | No callers |
| `IsCloudInfoEqual` | `common/common_net.h:26` | `common/common_net.cc:41` | No callers |
| `EndpointFromHostPortPB` | `common/wire_protocol.h:83` | `common/wire_protocol.cc:332` | No callers |
| `GetCurrentUser` | `server/call_home.h:91` | `server/call_home.cc:93` | No callers |
| `DocKeyBelongsTo` | `dockv/doc_key.h:468` | `dockv/doc_key.cc:1410` | Free function, no callers |
| `UniverseReplicationInfo::SetSetupUniverseReplicationErrorStatus` | `master/catalog_entity_info.h:1467` | Defined in .cc | Zero callers anywhere |

### 3.3 Signature Mismatch (Declaration Does Not Match Definition)

| Function | Issue |
|----------|-------|
| `TCMallocReleaseMemoryToSystem` | Header (`util/tcmalloc_util.h:45`) declares `void TCMallocReleaseMemoryToSystem()` but `.cc` defines `void TCMallocReleaseMemoryToSystem(int64_t bytes)` — the no-arg version has no definition |

### 3.4 Functions Only Used in Test Code

These have declaration and definition but are only referenced from test files.

| Function | Header | Notes |
|----------|--------|-------|
| `SleepUntil` | `util/monotime.h:333` | Only called from `pg_mini-test.cc` |
| `Base64Decode` | `util/url-coding.h:84` | Only called from `url-coding-test.cc` |
| `DumpTrackedStackTracesToLog` | `util/stack_trace_tracker.h:43` | Only called from `debug-util-test.cc` |
| `IsBoolean` | `util/string_util.h:103` | Only called from `string_util-test.cc` |
| `TrimCppComments` | `util/string_trim.h:60` | Only called from `string_trim-test.cc` |
| `ErrnoToCString` | `util/errno.h:47` | Only called from `errno-test.cc` |

### 3.5 Dead code confirmed (entire files/classes unused)

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

**Notable hotspot:** `src/yb/util/coding.h` has 7 unused declarations (`PutFixed64`, `PutLengthPrefixedSlice`, `PutFixed32LengthPrefixedSlice`, `VarintLength`, `EncodeFixed32`, `EncodeFixed64`, `EncodeVarint32`) — leftovers from LevelDB/RocksDB heritage where inline versions replaced the originals.

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
| Functions declared with no definition | 12 |
| Functions defined but never called | 12 |
| Signature mismatches | 1 |
| Functions only used in tests | 6 |

### Recommended Cleanup Priority

1. **Immediate:** Remove `src/yb/yql/pggate/pg_type.cc` — completely dead, won't compile if built
2. **Immediate:** Remove the 12 function declarations that have no definition (Section 3.1) — these are definitively dead
3. **High:** Remove the 12 high-confidence unreferenced headers — they define classes/types never used
4. **High:** Remove the 12 defined-but-never-called functions (Section 3.2) — clean up dead implementations
5. **High:** Fix `TCMallocReleaseMemoryToSystem` signature mismatch (Section 3.3)
6. **Medium:** Clean up the 5 platform-conditional headers (verify ARM/Windows paths are truly dead)
7. **Medium:** Consider moving test-only functions (Section 3.4) to test utilities
8. **Low:** Clean up RocksDB dead files (forked code, may diverge from upstream)
