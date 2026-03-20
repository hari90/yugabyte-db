# Unused Code Report for `src/yb/`

Automated analysis of unused code under `src/yb/`. Findings are grouped into three
categories: unbuilt `.cc` files, unreferenced `.h` files, and unused functions.

---

## 1. `.cc` Files Not Referenced in Any CMakeLists.txt (Not Being Built)

The YugabyteDB build system uses explicit source file lists in CMakeLists.txt
(via `ADD_YB_LIBRARY`, `ADD_YB_TEST`, `add_executable`, etc.). The following `.cc`
files exist on disk but are not referenced by any build target.

### 1a. Production Code (non-test, non-example)

| File | Notes |
|------|-------|
| `src/yb/yql/pggate/pg_type.cc` | Old implementation of `PgTypeInfo`; the `.h` now has everything inline. The `.cc` uses a different constructor signature (`const YbcPgTypeEntity*, int`) vs the header's (`YbcPgTypeEntities`). Dead code. |
| `src/yb/rocksdb/port/win/port_win.cc` | Windows port — not built on Linux/macOS |
| `src/yb/rocksdb/port/win/win_logger.cc` | Windows port — not built on Linux/macOS |

### 1b. Integration Tests Not Wired Into Build

| File |
|------|
| `src/yb/integration-tests/async_writes-test.cc` |
| `src/yb/integration-tests/cql-index-test.cc` |
| `src/yb/integration-tests/cql-packed-row-test.cc` |
| `src/yb/integration-tests/cql-tablet-split-test.cc` |
| `src/yb/integration-tests/cql-test.cc` |
| `src/yb/integration-tests/cql_geo_transactions-test.cc` |
| `src/yb/integration-tests/disk_full-test.cc` |
| `src/yb/integration-tests/documentdb/documentdb_test.cc` |
| `src/yb/integration-tests/external_mini_cluster_secure_test.cc` |
| `src/yb/integration-tests/minicluster-snapshot-test.cc` |
| `src/yb/integration-tests/sequence_utility-itest.cc` |
| `src/yb/integration-tests/tablet_limits_integration_test.cc` |
| `src/yb/integration-tests/upgrade-tests/ysql_ddl_whitelist-test.cc` |
| `src/yb/integration-tests/upgrade-tests/ysql_major_upgrade_role_profiles-test.cc` |

### 1c. RocksDB Tests/Benchmarks/Tools Not Wired Into Build

| File |
|------|
| `src/yb/rocksdb/db/compaction_iterator_test.cc` |
| `src/yb/rocksdb/db/forward_iterator_bench.cc` |
| `src/yb/rocksdb/db/memtable_list_test.cc` |
| `src/yb/rocksdb/db/memtablerep_bench.cc` |
| `src/yb/rocksdb/db/merge_helper_test.cc` |
| `src/yb/rocksdb/table/table_reader_bench.cc` |
| `src/yb/rocksdb/tools/db_repl_stress.cc` |
| `src/yb/rocksdb/tools/db_sanity_test.cc` |
| `src/yb/rocksdb/tools/db_stress.cc` |
| `src/yb/rocksdb/tools/write_stress.cc` |
| `src/yb/rocksdb/util/cache_bench.cc` |
| `src/yb/rocksdb/util/log_write_bench.cc` |
| `src/yb/rocksdb/utilities/env_mirror_test.cc` |
| `src/yb/rocksdb/utilities/merge_operators/string_append/stringappend_test.cc` |

### 1d. RocksDB Examples

| File |
|------|
| `src/yb/rocksdb/examples/column_families_example.cc` |
| `src/yb/rocksdb/examples/compact_files_example.cc` |
| `src/yb/rocksdb/examples/compaction_filter_example.cc` |
| `src/yb/rocksdb/examples/options_file_example.cc` |
| `src/yb/rocksdb/examples/simple_example.cc` |

### 1e. Fuzz Targets

| File |
|------|
| `src/yb/docdb/fuzz-targets/doc_key-fuzz_target.cc` |
| `src/yb/docdb/fuzz-targets/subdoc_key-fuzz_target.cc` |
| `src/yb/rocksdb/util/fuzz-targets/coding_fuzz_target.cc` |

**Total: 39 unbuilt `.cc` files**

---

## 2. Unreferenced `.h` Files (Not `#include`d by Anything)

These header files are not `#include`d by any `.cc`, `.c`, or `.h` file under `src/yb/`
or `src/postgres/`. Precompiled header files (`*_pch.h`) are excluded from this list
as they are consumed by the build system's PCH mechanism.

### 2a. Dead Production Headers

| File | Notes |
|------|-------|
| `src/yb/bfpg/base_operator.h` | Unused operator base class |
| `src/yb/bfpg/bfunc.h` | Unused built-in function declarations |
| `src/yb/bfpg/bfunc_convert.h` | Unused conversion functions |
| `src/yb/bfpg/bfunc_standard.h` | Unused standard functions |
| `src/yb/bfql/bfunc.h` | Unused CQL built-in function declarations |
| `src/yb/common/columnblock.h` | Legacy Kudu column block (never used in YB) |
| `src/yb/common/rowid.h` | Legacy Kudu row ID utilities (never used in YB) |
| `src/yb/gutil/fixedarray.h` | Unused fixed-size array template |
| `src/yb/gutil/move.h` | Obsolete C++03 move emulation macro — superseded by C++11 `std::move` |
| `src/yb/gutil/paranoid.h` | Unused paranoid assertion macros |
| `src/yb/rpc/lw_pb_conversion.h` | Unused lightweight protobuf conversion header |
| `src/yb/server/mock_hybrid_clock.h` | Unused mock clock for tests |
| `src/yb/util/auto_release_pool.h` | Unused auto-release pool |
| `src/yb/util/bit-stream-utils.inline.h` | Unused bit-stream inline utilities |
| `src/yb/util/debug_ref_counted.h` | Unused debug ref-counting utilities |
| `src/yb/util/protobuf-annotations.h` | Unused protobuf annotation macros |
| `src/yb/yql/cql/ql/parser/parser_inactive_nodes.h` | Unused CQL parser node types |
| `src/yb/yql/pggate/pg_coldesc.h` | Old column descriptor class (replaced by different implementation) |
| `src/yb/yql/pggate/type_mapping.h` | Unused C/C++ type mapping templates |

### 2b. Dead RocksDB/Vendor Headers

| File | Notes |
|------|-------|
| `src/yb/gutil/auxiliary/atomicops-internals-arm-generic.h` | ARM architecture — not used on x86_64 builds |
| `src/yb/gutil/auxiliary/atomicops-internals-arm-v6plus.h` | ARM architecture — not used on x86_64 builds |
| `src/yb/gutil/auxiliary/atomicops-internals-windows.h` | Windows platform — not built |
| `src/yb/rocksdb/port/port_example.h` | Example port header (reference only) |
| `src/yb/rocksdb/precompiled_header.h` | Unused precompiled header |
| `src/yb/rocksdb/table/forwarding_iterator.h` | Unused forwarding iterator |
| `src/yb/rocksdb/util/channel.h` | Unused channel utility |
| `src/yb/rocksdb/utilities/convenience.h` | Unused convenience utilities |
| `src/yb/rocksdb/utilities/info_log_finder.h` | Unused info log finder |
| `src/yb/rocksdb/utilities/optimistic_transaction_db.h` | Unused optimistic transaction DB |

### 2c. Companion Headers for Unbuilt `.cc` Files

These headers have companion `.cc` files that are also not built:

| Header | Companion `.cc` |
|--------|-----------------|
| `src/yb/yql/pggate/pg_type.h` | `pg_type.cc` (stale `.cc`, `.h` may still be used elsewhere) |
| `src/yb/rocksdb/port/win/port_win.h` | `port_win.cc` |
| `src/yb/rocksdb/port/win/win_logger.h` | `win_logger.cc` |

**Total: 29 unreferenced `.h` files** (excluding `*_pch.h` and platform-specific files)

---

## 3. Unused Functions

Functions defined in production `.cc` files whose names are never referenced outside
their defining file (and optional `.h` companion). Virtual/override methods are excluded
to avoid false positives from dynamic dispatch.

### 3a. Unused Static Functions (dead code within translation unit)

| File | Line | Function |
|------|------|----------|
| `src/yb/rocksdb/db/db_iter.cc` | 78 | `static DumpInternalIter()` |
| `src/yb/tserver/remote_bootstrap_session.cc` | 521 | `static AddImmutableFileToMap()` |

### 3b. Unused Class Member Functions

#### Client Library (`src/yb/client/`)

| File | Line | Function |
|------|------|----------|
| `client/client.cc` | 434 | `YBClientBuilder::set_callback_threadpool_size()` |
| `client/client.cc` | 2039 | `YBClient::AddTablesToUniverseReplication()` |
| `client/client.cc` | 2058 | `YBClient::RemoveTablesFromUniverseReplication()` |
| `client/client.cc` | 3069 | `YBClient::SetLatestObservedHybridTime()` |
| `client/error.cc` | 40 | `YBError::was_possibly_successful()` |
| `client/meta_cache.cc` | 306 | `RemoteTabletServer::HasHostFrom()` |
| `client/meta_cache.cc` | 490 | `RemoteTablet::ReplicasCountToString()` |
| `client/schema.cc` | 514 | `YBSchema::ColumnById()` |
| `client/schema.cc` | 570 | `YBSchema::GetPrimaryKeyColumnIndexes()` |
| `client/session.cc` | 449 | `YBSession::ResetArena()` |
| `client/value.cc` | 101 | `YBValue::FromBool()` |
| `client/yb_op.cc` | 491 | `YBOperation::IsYsqlCatalogOp()` |

#### CDC (`src/yb/cdc/`)

| File | Line | Function |
|------|------|----------|
| `cdc/cdc_service.cc` | 1040 | `CDCServiceImpl::InitNewTabletStreamEntry()` |
| `cdc/cdc_state_table.cc` | 842 | `CDCStateTableEntrySelector::IncludePubRefreshTimes()` |
| `cdc/cdc_state_table.cc` | 846 | `CDCStateTableEntrySelector::IncludeLastDecidedPubRefreshTime()` |
| `cdc/cdc_state_table.cc` | 850 | `CDCStateTableEntrySelector::IncludeStartHashRange()` |
| `cdc/cdc_state_table.cc` | 854 | `CDCStateTableEntrySelector::IncludeEndHashRange()` |

#### Common / DocDB / Dockv

| File | Line | Function |
|------|------|----------|
| `common/schema.cc` | 540 | `Schema::VerifyProjectionCompatibility()` |
| `common/version_info.cc` | 86 | `VersionInfo::GetAllVersionInfo()` |
| `docdb/docdb_util.cc` | 675 | `DocDBRocksDBUtil::MakeLWValue()` |
| `docdb/redis_operation.cc` | 1387 | `RedisWriteOperation::ApplyRemove()` |
| `dockv/doc_key.cc` | 734 | `DocKey::BelongsTo()` |
| `dockv/doc_key.cc` | 1190 | `SubDocKey::RemoveLastSubKey()` |
| `dockv/partial_row.cc` | 746 | `YBPartialRow::IsHashOrPrimaryKeySet()` |
| `dockv/partial_row.cc` | 750 | `YBPartialRow::AllColumnsSet()` |
| `dockv/primitive_value.cc` | 1828 | `PrimitiveValue::DecodeVector()` |
| `dockv/primitive_value.cc` | 3593 | `PrimitiveValue::NullSlice()` |
| `dockv/primitive_value.cc` | 3598 | `PrimitiveValue::TombstoneSlice()` |
| `dockv/schema_packing.cc` | 534 | `SchemaPacking::GetBounds()` |

#### Consensus / Master

| File | Line | Function |
|------|------|----------|
| `consensus/log_cache.cc` | 635 | `LogCache::DumpToLog()` |
| `consensus/raft_consensus.cc` | 1462 | `RaftConsensus::AppendEmptyBatchToLeaderLog()` |
| `consensus/raft_consensus.cc` | 2957 | `RaftConsensus::GetActiveRole()` |
| `master/catalog_entity_info.cc` | 169 | `TabletReplica::IsStarting()` |
| `master/catalog_entity_info.cc` | 1621 | `UniverseReplicationInfo::SetSetupUniverseReplicationErrorStatus()` |
| `master/catalog_entity_info.cc` | 1769 | `SnapshotInfo::IsRestoreInProgress()` |
| `master/catalog_entity_info.cc` | 1773 | `SnapshotInfo::IsDeleteInProgress()` |
| `master/cluster_balance.cc` | 302 | `ClusterLoadBalancer::get_total_blacklisted_servers()` |
| `master/cluster_balance.cc` | 306 | `ClusterLoadBalancer::get_total_leader_blacklisted_servers()` |
| `master/cluster_balance.cc` | 314 | `ClusterLoadBalancer::get_total_under_replication()` |
| `master/cluster_balance.cc` | 1997 | `ClusterLoadBalancer::GetLiveClusterPlacementInfo()` |
| `master/snapshot_state.cc` | 203 | `SnapshotState::SetVersion()` |
| `master/state_with_tablets.cc` | 203 | `StateWithTablets::HasInState()` |
| `master/sys_catalog.cc` | 184 | `SysCatalogTable::schema_column_type()` |
| `master/sys_catalog.cc` | 186 | `SysCatalogTable::schema_column_id()` |
| `master/sys_catalog.cc` | 188 | `SysCatalogTable::schema_column_metadata()` |
| `master/xcluster/xcluster_safe_time_service.cc` | 762 | `XClusterSafeTimeService::EnterIdleMode()` |

#### RPC / Server

| File | Line | Function |
|------|------|----------|
| `rpc/messenger.cc` | 296 | `Messenger::BreakConnectivityFrom()` |
| `rpc/messenger.cc` | 354 | `Messenger::RestoreConnectivityFrom()` |
| `rpc/reactor.cc` | 831 | `Reactor::DropWithRemoteAddress()` |
| `server/server_base.cc` | 267 | `RpcServerBase::get_hostname()` |

#### Tablet / TServer

| File | Line | Function |
|------|------|----------|
| `tablet/tablet_metadata.cc` | 1456 | `RaftGroupMetadata::SetTableName()` |
| `tablet/tablet_snapshots.cc` | 177 | `TabletSnapshots::IsLastSnapshotTimeFilePath()` |
| `tserver/remote_bootstrap_service.cc` | 668 | `RemoteBootstrapServiceImpl::UpdateLogAnchor()` |
| `tserver/remote_bootstrap_service.cc` | 730 | `RemoteBootstrapServiceImpl::ChangePeerRole()` |
| `tserver/stateful_services/pg_auto_analyze_service.cc` | 900 | `PgAutoAnalyzeService::IncreaseMutationCountersImpl()` |
| `tserver/stateful_services/pg_cron_leader_service.cc` | 216 | `PgCronLeaderService::PgCronSetLastMinuteImpl()` |
| `tserver/stateful_services/pg_cron_leader_service.cc` | 229 | `PgCronLeaderService::PgCronGetLastMinuteImpl()` |
| `tserver/tablet_server.cc` | 2086 | `TabletServer::GetUniverseKeyManager()` |
| `tserver/tablet_service.cc` | 1666 | `TabletServiceImpl::GetCompatibleSchemaVersion()` |
| `tserver/tablet_service.cc` | 2388 | `TabletServiceAdminImpl::EnableDbConns()` |
| `tserver/tablet_service.cc` | 2837 | `ConsensusServiceImpl::MultiRaftUpdateConsensus()` |
| `tserver/tablet_service.cc` | 3007 | `ConsensusServiceImpl::GetNodeInstance()` |
| `tserver/tablet_service.cc` | 4047 | `TabletServiceAdminImpl::GetActiveRbsInfo()` |

#### Utilities (`src/yb/util/`)

| File | Line | Function |
|------|------|----------|
| `util/bloom_filter.cc` | 63 | `BloomFilterSizing::BySizeAndFPRate()` |
| `util/debug/trace_event_impl.cc` | 1419 | `TraceLog::AddEnabledStateObserver()` |
| `util/debug/trace_event_impl.cc` | 1423 | `TraceLog::RemoveEnabledStateObserver()` |
| `util/debug/trace_event_impl.cc` | 1430 | `TraceLog::HasEnabledStateObserver()` |
| `util/debug/trace_event_impl.cc` | 2028 | `TraceLog::CancelWatchEvent()` |
| `util/debug/trace_event_impl.cc` | 2145 | `TraceLog::SetProcessSortIndex()` |
| `util/debug/trace_event_impl.cc` | 2150 | `TraceLog::SetProcessName()` |
| `util/debug/trace_event_impl.cc` | 2155 | `TraceLog::UpdateProcessLabel()` |
| `util/debug/trace_event_impl.cc` | 2173 | `TraceLog::SetThreadSortIndex()` |
| `util/debug/trace_event_impl.cc` | 2178 | `TraceLog::SetTimeOffset()` |
| `util/debug/trace_event_impl.cc` | 2182 | `TraceLog::GetObserverCountForTest()` |
| `util/hdr_histogram.cc` | 306 | `HdrHistogram::ValuesAreEquivalent()` |
| `util/metrics.cc` | 768 | `Histogram::CountInBucketForValueForTests()` |
| `util/monotime.cc` | 204 | `MonoDelta::ToDays()` |
| `util/net/inetaddress.cc` | 189 | `InetAddress::isV4()` |
| `util/net/inetaddress.cc` | 194 | `InetAddress::isV6()` |
| `util/net/net_util.cc` | 120 | `HostPort::RemoveAndGetHostPortList()` |
| `util/net/rate_limiter.cc` | 162 | `RateLimiter::SetTargetRate()` |
| `util/net/socket.cc` | 235 | `Socket::IsNonBlocking()` |
| `util/net/socket.cc` | 405 | `Socket::GetSockError()` |
| `util/slice.cc` | 198 | `Slice::MakeNoLongerThan()` |
| `util/write_buffer.cc` | 288 | `WriteBuffer::FirstBlockSlice()` |

#### RocksDB

| File | Line | Function |
|------|------|----------|
| `rocksdb/db/db_impl.cc` | 3367 | `DBImpl::IsEmptyCompactionQueue()` |
| `rocksdb/db/db_impl.cc` | 5187 | `DBImpl::GetSnapshotForWriteConflictBoundary()` |
| `rocksdb/db/db_impl.cc` | 6027 | `DBImpl::GetAndRefSuperVersionUnlocked()` |
| `rocksdb/db/db_impl.cc` | 6073 | `DBImpl::ReturnAndCleanupSuperVersionUnlocked()` |
| `rocksdb/db/db_impl.cc` | 6101 | `DBImpl::GetColumnFamilyHandleUnlocked()` |
| `rocksdb/db/db_impl.cc` | 6126 | `DBImpl::AreWritesStopped()` |
| `rocksdb/db/db_impl.cc` | 7075 | `DBImpl::GetEarliestMemTableSequenceNumber()` |
| `rocksdb/db/db_impl.cc` | 7089 | `DBImpl::GetLatestSequenceForKey()` |
| `rocksdb/db/file_indexer.cc` | 34 | `FileIndexer::NumLevelIndex()` |
| `rocksdb/db/version_set.cc` | 1954 | `VersionStorageInfo::LevelFileSummary()` |

#### CQL / PgGate / YQL

| File | Line | Function |
|------|------|----------|
| `yql/cql/ql/parser/scanner.cc` | 157 | `GramProcessor::make_NOT_LA()` |
| `yql/cql/ql/parser/scanner.cc` | 172 | `GramProcessor::make_NULLS_LA()` |
| `yql/cql/ql/parser/scanner.cc` | 187 | `GramProcessor::make_WITH_LA()` |
| `yql/cql/ql/parser/scanner.cc` | 207 | `LexProcessor::LexerInput()` |
| `yql/cql/ql/parser/scanner.cc` | 228 | `GramProcessor::make_SCAN_ERROR()` |
| `yql/cql/ql/parser/scanner.cc` | 248 | `LexProcessor::ScanLiteral()` |
| `yql/cql/ql/parser/scanner.cc` | 259 | `LexProcessor::MakeIdentifier()` |
| `yql/cql/ql/parser/scanner.cc` | 335 | `LexProcessor::addlitchar()` |
| `yql/cql/ql/parser/scanner.cc` | 457 | `LexProcessor::check_string_escape_warning()` |
| `yql/cql/ql/parser/scanner.cc` | 486 | `LexProcessor::unescape_single_char()` |
| `yql/cql/ql/parser/scanner.cc` | 509 | `LexProcessor::addunicode()` |
| `yql/cql/ql/parser/scanner.cc` | 533 | `LexProcessor::ScanKeywordLookup()` |
| `yql/cql/ql/ptree/pt_expr.cc` | 328 | `PTExpr::CreateConst()` |
| `yql/cql/ql/ptree/pt_type.cc` | 27 | `PTBaseType::FromQLType()` |
| `yql/cql/ql/ptree/sem_context.cc` | 74 | `SemContext::set_current_create_index_stmt()` |
| `yql/cql/ql/util/cql_message.cc` | 439 | `CQLRequest::ParseUUID()` |
| `yql/cql/ql/util/cql_message.cc` | 447 | `CQLRequest::ParseTimeUUID()` |
| `yql/cql/ql/util/cql_message.cc` | 466 | `CQLRequest::ParseInet()` |
| `yql/cql/ql/util/cql_message.cc` | 504 | `CQLRequest::ParseStringMultiMap()` |
| `yql/cql/ql/util/cql_message.cc` | 518 | `CQLRequest::ParseBytesMap()` |
| `yql/pggate/insert_on_conflict_buffer.cc` | 91 | `InsertOnConflictBuffer::ClearIntents()` |
| `yql/pggate/pg_doc_op.cc` | 1493 | `PgDocWriteOp::GetWriteOp()` |
| `yql/pggate/util/pg_wire.cc` | 21 | `PgWire::WriteBool()` |
| `yql/pggate/util/pg_wire.cc` | 25 | `PgWire::WriteInt8()` |
| `yql/pggate/util/pg_wire.cc` | 29 | `PgWire::WriteUint8()` |
| `yql/pggate/util/pg_wire.cc` | 33 | `PgWire::WriteUint16()` |
| `yql/pggate/util/pg_wire.cc` | 37 | `PgWire::WriteInt16()` |
| `yql/pggate/util/pg_wire.cc` | 41 | `PgWire::WriteUint32()` |
| `yql/pggate/util/pg_wire.cc` | 45 | `PgWire::WriteInt32()` |
| `yql/pggate/util/pg_wire.cc` | 49 | `PgWire::WriteUint64()` |
| `yql/pggate/util/pg_wire.cc` | 61 | `PgWire::WriteFloat()` |
| `yql/pggate/util/pg_wire.cc` | 66 | `PgWire::WriteDouble()` |
| `yql/pggate/util/pg_wire.cc` | 71 | `PgWire::WriteText()` |
| `yql/pggate/util/pg_wire.cc` | 79 | `PgWire::WriteBinary()` |
| `yql/pggate/util/pg_wire.cc` | 86 | `PgWire::ReadBytes()` |
| `yql/pggate/util/pg_wire.cc` | 92 | `PgWire::ReadString()` |
| `yql/process_wrapper/process_wrapper.cc` | 141 | `ProcessSupervisor::InitializeProcessWrapperUnlocked()` |
| `yql/ysql_conn_mgr_wrapper/ysql_conn_mgr_conf.cc` | 73 | `PutConfigValue()` |
| `yql/ysql_conn_mgr_wrapper/ysql_conn_mgr_conf.cc` | 129 | `get_num_workers()` |
| `yql/ysql_conn_mgr_wrapper/ysql_conn_mgr_conf.cc` | 250 | `getMaxConnectionsFromYsqlPgConf()` |
| `yql/ysql_conn_mgr_wrapper/ysql_conn_mgr_wrapper.cc` | 170 | `ValidateLogSettings()` |
| `yql/ysql_conn_mgr_wrapper/ysql_conn_mgr_wrapper.cc` | 205 | `YsqlConnMgrWrapper::GetYsqlConnMgrExecutablePath()` |

#### Integration Test Utilities (unused by any test)

| File | Line | Function |
|------|------|----------|
| `integration-tests/cdcsdk_ysql_test_base.cc` | 2817 | `CDCSDKYsqlTest::TriggerCompaction()` |
| `integration-tests/cdcsdk_ysql_test_base.cc` | 2829 | `CDCSDKYsqlTest::CompactSystemTable()` |
| `integration-tests/cdcsdk_ysql_test_base.cc` | 3176 | `CDCSDKYsqlTest::GetTotalNumRecordsInTablet()` |
| `integration-tests/cdcsdk_ysql_test_base.cc` | 4440 | `CDCSDKYsqlTest::GetRecordsAndSplitCount()` |
| `integration-tests/cluster_verifier.cc` | 81 | `ClusterVerifier::SetScanConcurrency()` |
| `integration-tests/cql_test_util.cc` | 202 | `CassandraIterator::MoveToRow()` |
| `integration-tests/external_mini_cluster.cc` | 651 | `ExternalMiniCluster::GetLeaderConsensusProxy()` |
| `integration-tests/external_mini_cluster.cc` | 1244 | `ExternalMiniCluster::GetTabletServerAddresses()` |
| `integration-tests/external_mini_cluster.cc` | 2679 | `ExternalTabletServer::GetInt64CQLMetric()` |
| `integration-tests/ts_itest-base.cc` | 279 | `TabletServerIntegrationTestBase::PruneFromReplicas()` |
| `integration-tests/ts_itest-base.cc` | 323 | `TabletServerIntegrationTestBase::GetFurthestAheadReplicaIdx()` |
| `integration-tests/upgrade-tests/ysql_major_upgrade_test_base.cc` | 202 | `YsqlMajorUpgradeTestBase::ExecuteStatementsInFiles()` |
| `integration-tests/xcluster/xcluster_test_base.cc` | 1059 | `XClusterTestBase::GetProducerMasterProxy()` |
| `integration-tests/yb_table_test_base.cc` | 91 | `YBTableTestBase::need_redis_table()` |
| `integration-tests/yb_table_test_base.cc` | 346 | `YBTableTestBase::FetchTSMetricsPage()` |
| `integration-tests/rpc-test-base.cc` | 227 | `GenericCalculatorService::DoRepeatedEcho()` |
| `rocksdb/db/db_test_util.cc` | 847 | `DBHolder::DumpFileCounts()` |
| `rocksdb/db/db_test_util.cc` | 859 | `DBHolder::DumpSSTableList()` |
| `tserver/tablet_server-test-base.cc` | 178 | `TabletServerTestBase::UpdateTestRowRemote()` |
| `tserver/tablet_server-test-base.cc` | 208 | `TabletServerTestBase::InsertTestRowsDirect()` |
| `tools/ts-cli.cc` | 471 | `TsAdminClient::GetTabletSchema()` |
| `gen_yrpc/model.cc` | 43 | `WireFormatLite::WireTypeForFieldType()` |
| `gutil/bits.cc` | 61 | `Bits::CappedDifference()` |
| `gutil/bits.cc` | 71 | `Bits::Log2Floor_Portable()` |
| `gutil/bits.cc` | 104 | `Bits::FindLSBSetNonZero_Portable()` |
| `gutil/cpu.cc` | 292 | `CPU::GetIntelMicroArchitecture()` |
| `gutil/ref_counted_memory.cc` | 55 | `RefCountedBytes::TakeVector()` |
| `gutil/ref_counted_memory.cc` | 79 | `RefCountedString::TakeString()` |
| `hnsw/hnsw.cc` | 639 | `vector_index::TryFullyDecodeVectorId()` |
| `qlexpr/ql_rowblock.cc` | 101 | `QLRow::SetColumnValues()` |
| `rocksdb/utilities/merge_operators/string_append/stringappend2.cc` | 89 | `StringAppendTESTOperator::_AssocPartialMergeMulti()` |
| `yql/pgwrapper/pg_locks_test_base.cc` | 252 | `PgLocksTestBase::get_pg_client_service_proxies()` |

**Total: 167 unused functions** (2 static + 165 member functions)

---

## Summary

| Category | Count |
|----------|-------|
| Unbuilt `.cc` files | 39 |
| Unreferenced `.h` files | 29 (excl. `*_pch.h`) |
| Unused functions | 167 |

### Methodology

- **Unbuilt `.cc` files**: All CMakeLists.txt files under `src/yb/` were parsed for
  references to `.cc` files via `ADD_YB_LIBRARY`, `ADD_YB_TEST`, `add_executable`, and
  `add_library` directives. Files on disk not referenced by any target were flagged.

- **Unreferenced `.h` files**: All `#include` directives across `src/yb/` and
  `src/postgres/` source files were collected. Header files under `src/yb/` not
  matching any include path were flagged. Precompiled headers (`*_pch.h`) were excluded
  as they are consumed by the CMake PCH mechanism rather than `#include`.

- **Unused functions**: Function definitions were extracted from production `.cc` files
  using regex-based parsing. A word-frequency index was built across all source files.
  Functions whose names appear only in their defining file (and optional `.h` companion),
  with at most 2 total occurrences (definition + declaration), were flagged.
  Virtual/override methods were excluded to avoid false positives from dynamic dispatch.

### Caveats

- Some functions may be called via templates, macros, or reflection not captured by
  simple text matching.
- Platform-specific files (Windows, ARM) are reported but may be intentionally unbuilt
  on the current platform.
- Integration test `.cc` files may be intentionally excluded from the build if they
  require special infrastructure.
- The CQL scanner functions may be used by generated parser code not analyzed here.
