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
import static org.yb.AssertionWrappers.assertNotNull;
import static org.yb.AssertionWrappers.assertTrue;

import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.Indexes;
import com.mongodb.client.model.Sorts;
import com.mongodb.client.result.DeleteResult;
import com.mongodb.client.result.InsertManyResult;

import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Arrays;
import java.util.List;

import org.bson.Document;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.yb.util.YBTestRunnerNonSanOrAArch64Mac;

/**
 * RUM index integration test for the DocumentDB extension on YugabyteDB.
 *
 * Drives the data path through the MongoDB protocol gateway (insertMany,
 * createIndex, find, deleteOne) and uses the inherited YSQL connection for
 * the catalog-level RUM assertions (AM/opclass/function registration, index
 * presence, plan validation) that the MongoDB API does not expose.
 */
@RunWith(value = YBTestRunnerNonSanOrAArch64Mac.class)
public class TestPgDocumentDBRumIndex extends BaseDocumentDBTest {

  @Test
  public void testRumIndexOnYBCollection() throws Exception {
    // Catalog-level checks: the RUM AM, opclasses, handler, and text-search
    // adapter functions should be registered as soon as the extension is
    // created.
    try (Statement stmt = connection.createStatement()) {
      try (ResultSet rs = stmt.executeQuery(
          "SELECT count(*) FROM pg_am WHERE amname = 'documentdb_rum'")) {
        assertTrue(rs.next());
        assertEquals(1L, rs.getLong(1));
      }

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

      try (ResultSet rs = stmt.executeQuery(
          "SELECT count(*) FROM pg_proc p " +
          "JOIN pg_namespace n ON p.pronamespace = n.oid " +
          "WHERE p.proname = 'documentdbrumhandler' " +
          "AND n.nspname = 'documentdb_api_catalog'")) {
        assertTrue(rs.next());
        assertEquals(1L, rs.getLong(1));
      }

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

    // Data path: drive everything through the MongoDB protocol gateway.
    String collName = "rum_yb_coll";
    MongoDatabase db = mongoClient.getDatabase(TEST_DB);
    MongoCollection<Document> collection = db.getCollection(collName);

    List<Document> docs = Arrays.asList(
        new Document("_id", 1).append("a", 10).append("b", "hello"),
        new Document("_id", 2).append("a", 5).append("b", "world"),
        new Document("_id", 3).append("a", 20).append("b", "foo"),
        new Document("_id", 4).append("a", 15).append("b", "bar"),
        new Document("_id", 5).append("a", 1).append("b", "baz"));
    InsertManyResult insertResult = collection.insertMany(docs);
    assertEquals(5, insertResult.getInsertedIds().size());

    // Creating a single-field index goes through DocumentDB's createIndexes
    // path which materializes a RUM index on the YB-backed table.
    collection.createIndex(Indexes.ascending("a"));

    // RUM index is present in the catalog.
    try (Statement stmt = connection.createStatement();
         ResultSet rs = stmt.executeQuery(
             "SELECT count(*) FROM pg_indexes " +
             "WHERE indexname LIKE 'documents_rum_index_%'")) {
      assertTrue(rs.next());
      assertTrue("expected at least one RUM index, got " + rs.getLong(1), rs.getLong(1) >= 1);
    }

    assertEquals(5, collection.countDocuments());

    // Forward sort with a filter goes through the RUM index scan path.
    // Smallest "a" value <= 10 is 1.
    Document firstAsc = collection.find(Filters.lte("a", 10))
        .sort(Sorts.ascending("a"))
        .limit(1)
        .first();
    assertNotNull(firstAsc);
    assertEquals(1, firstAsc.getInteger("a").intValue());

    // Backward sort - largest "a" value is 20.
    Document firstDesc = collection.find()
        .sort(Sorts.descending("a"))
        .limit(1)
        .first();
    assertNotNull(firstDesc);
    assertEquals(20, firstDesc.getInteger("a").intValue());

    // Plan validation for the forward-sort query: the planner should pick
    // an Index Scan over the RUM index.
    try (Statement stmt = connection.createStatement()) {
      String plan = explainAsString(stmt,
          "EXPLAIN (COSTS OFF) SELECT document FROM " +
          "documentdb_api_catalog.bson_aggregation_find('" + TEST_DB + "', " +
          "'{ \"find\": \"" + collName + "\", \"filter\": {\"a\": {\"$lte\": 10}}, " +
          "\"sort\": {\"a\": 1}, \"limit\": 1}')");
      assertTrue("expected Index Scan in plan, got: " + plan, plan.contains("Index Scan"));
    }

    // Delete one document and verify count.
    DeleteResult deleteResult = collection.deleteOne(Filters.eq("_id", 1));
    assertEquals(1, deleteResult.getDeletedCount());
    assertEquals(4, collection.countDocuments());
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
