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
package org.yb.pgsql;

import static org.yb.AssertionWrappers.assertEquals;
import static org.yb.AssertionWrappers.assertTrue;

import java.sql.ResultSet;
import java.sql.Statement;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.yb.util.YBTestRunnerNonSanOrAArch64Mac;

/**
 * RUM index integration tests for the DocumentDB extension on YugabyteDB.
 *
 * Mirrors the C++ test {@code DocumentDBTest.RumIndexOnYBCollection} in
 * {@code src/yb/integration-tests/documentdb/documentdb_test.cc} but exercised
 * through the YSQL JDBC client. The MongoDB protocol gateway is not required.
 */
@RunWith(value = YBTestRunnerNonSanOrAArch64Mac.class)
public class TestPgDocumentDBRumIndex extends BaseDocumentDBTest {

  private static final String DB_NAME = "rumdb";
  private static final String COLL_NAME = "rum_yb_coll";

  @Override
  protected boolean useGateway() {
    return false;
  }

  @Before
  public void setSearchPath() throws Exception {
    try (Statement stmt = connection.createStatement()) {
      stmt.execute("SET search_path TO documentdb_api, documentdb_core");
      stmt.execute("SET documentdb_core.bsonUseEJson TO TRUE");
    }
  }

  @Test
  public void testRumAccessMethodAndOpClasses() throws Exception {
    try (Statement stmt = connection.createStatement()) {
      // documentdb_rum AM is registered.
      try (ResultSet rs = stmt.executeQuery(
          "SELECT count(*) FROM pg_am WHERE amname = 'documentdb_rum'")) {
        assertTrue(rs.next());
        assertEquals(1L, rs.getLong(1));
      }

      // RUM-backed opclasses are registered.
      try (ResultSet rs = stmt.executeQuery(
          "SELECT count(*) FROM pg_opclass opc " +
          "JOIN pg_am am ON opc.opcmethod = am.oid " +
          "WHERE am.amname = 'documentdb_rum' " +
          "AND opc.opcname IN ('bson_rum_single_path_ops', 'bson_rum_text_path_ops', " +
          "'bson_rum_wildcard_project_path_ops', 'documentdb_rum_hashed_ops', " +
          "'bson_rum_exclusion_ops')")) {
        assertTrue(rs.next());
        assertEquals(5L, rs.getLong(1));
      }

      // RUM handler function in documentdb_api_catalog.
      try (ResultSet rs = stmt.executeQuery(
          "SELECT count(*) FROM pg_proc p " +
          "JOIN pg_namespace n ON p.pronamespace = n.oid " +
          "WHERE p.proname = 'documentdbrumhandler' " +
          "AND n.nspname = 'documentdb_api_catalog'")) {
        assertTrue(rs.next());
        assertEquals(1L, rs.getLong(1));
      }

      // RUM text search adapter functions in documentdb_api_internal.
      try (ResultSet rs = stmt.executeQuery(
          "SELECT count(*) FROM pg_proc p " +
          "JOIN pg_namespace n ON p.pronamespace = n.oid " +
          "WHERE p.proname IN ('rum_extract_tsquery', 'rum_tsquery_consistent', " +
          "'rum_tsvector_config', 'rum_tsquery_pre_consistent', " +
          "'rum_tsquery_distance', 'rum_ts_join_pos') " +
          "AND n.nspname = 'documentdb_api_internal'")) {
        assertTrue(rs.next());
        assertEquals(6L, rs.getLong(1));
      }
    }
  }

  @Test
  public void testRumIndexOnYBCollection() throws Exception {
    try (Statement stmt = connection.createStatement()) {
      // Insert documents into a YB-backed collection.
      stmt.execute(String.format(
          "SELECT documentdb_api.insert('%s', '{\"insert\":\"%s\", \"documents\":[" +
          "{\"_id\": 1, \"a\": 10, \"b\": \"hello\"}," +
          "{\"_id\": 2, \"a\": 5, \"b\": \"world\"}," +
          "{\"_id\": 3, \"a\": 20, \"b\": \"foo\"}," +
          "{\"_id\": 4, \"a\": 15, \"b\": \"bar\"}," +
          "{\"_id\": 5, \"a\": 1, \"b\": \"baz\"}]}')",
          DB_NAME, COLL_NAME));

      // Create a single-field RUM index via DocumentDB API.
      stmt.execute(String.format(
          "SELECT documentdb_api_internal.create_indexes_non_concurrently('%s', " +
          "'{\"createIndexes\":\"%s\", \"indexes\":[{\"key\":{\"a\":1}, \"name\":\"a_1\"}]}', " +
          "true)",
          DB_NAME, COLL_NAME));

      // Verify a RUM index was created in the catalog.
      try (ResultSet rs = stmt.executeQuery(
          "SELECT count(*) FROM pg_indexes " +
          "WHERE indexname LIKE 'documents_rum_index_%'")) {
        assertTrue(rs.next());
        assertTrue("expected at least one RUM index, got " + rs.getLong(1), rs.getLong(1) >= 1);
      }

      // Verify data is still accessible after index creation.
      try (ResultSet rs = stmt.executeQuery(String.format(
          "SELECT count(*) FROM documentdb_api.collection('%s','%s')",
          DB_NAME, COLL_NAME))) {
        assertTrue(rs.next());
        assertEquals(5L, rs.getLong(1));
      }

      // Validate that the query plan uses the RUM index (forward scan).
      String fwdPlan = explainAsString(stmt, String.format(
          "EXPLAIN (COSTS OFF) SELECT document FROM " +
          "documentdb_api_catalog.bson_aggregation_find('%s', " +
          "'{ \"find\": \"%s\", \"filter\": {\"a\": {\"$lte\": 10}}, " +
          "\"sort\": {\"a\": 1}, \"limit\": 1}')",
          DB_NAME, COLL_NAME));
      assertTrue("expected Index Scan in plan, got: " + fwdPlan, fwdPlan.contains("Index Scan"));

      // Forward sort query - smallest "a" value <= 10 is 1.
      try (ResultSet rs = stmt.executeQuery(String.format(
          "SELECT (((cursorpage->>'cursor')::bson->>'firstBatch')::bson->>'0')::bson->>'a' " +
          "FROM documentdb_api.find_cursor_first_page('%s', " +
          "'{\"find\":\"%s\", \"filter\":{\"a\":{\"$lte\":10}}, \"sort\":{\"a\":1}, " +
          "\"limit\":1}')",
          DB_NAME, COLL_NAME))) {
        assertTrue(rs.next());
        assertEquals("1", rs.getString(1));
      }

      // Backward sort query - largest "a" value is 20.
      try (ResultSet rs = stmt.executeQuery(String.format(
          "SELECT (((cursorpage->>'cursor')::bson->>'firstBatch')::bson->>'0')::bson->>'a' " +
          "FROM documentdb_api.find_cursor_first_page('%s', " +
          "'{\"find\":\"%s\", \"sort\":{\"a\":-1}, \"limit\":1}')",
          DB_NAME, COLL_NAME))) {
        assertTrue(rs.next());
        assertEquals("20", rs.getString(1));
      }

      // Delete a document and verify count.
      stmt.execute(String.format(
          "SELECT documentdb_api.delete('%s', '{\"delete\": \"%s\", " +
          "\"deletes\": [{\"q\": {\"_id\": 1}, \"limit\": 1}]}')",
          DB_NAME, COLL_NAME));

      try (ResultSet rs = stmt.executeQuery(String.format(
          "SELECT count(*) FROM documentdb_api.collection('%s','%s')",
          DB_NAME, COLL_NAME))) {
        assertTrue(rs.next());
        assertEquals(4L, rs.getLong(1));
      }
    }
  }

  private static String explainAsString(Statement stmt, String query) throws Exception {
    StringBuilder sb = new StringBuilder();
    try (ResultSet rs = stmt.executeQuery(query)) {
      while (rs.next()) {
        sb.append(rs.getString(1)).append('\n');
      }
    }
    return sb.toString();
  }
}
