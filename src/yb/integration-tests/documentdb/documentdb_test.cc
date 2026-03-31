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

#include "yb/yql/pgwrapper/pg_mini_test_base.h"

DECLARE_bool(ysql_enable_documentdb);
DECLARE_bool(enable_pg_cron);

namespace yb {
class DocumentDBTest : public pgwrapper::PgMiniTestBase {
 public:
  void SetUp() override {
#ifndef YB_ENABLE_YSQL_DOCUMENTDB_EXT
    GTEST_SKIP() << "DocumentDB extension is not available in build type";
#endif

    ANNOTATE_UNPROTECTED_WRITE(FLAGS_ysql_enable_documentdb) = true;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_enable_pg_cron) = true;

    TEST_SETUP_SUPER(pgwrapper::PgMiniTestBase);

    conn_ = std::make_unique<pgwrapper::PGConn>(ASSERT_RESULT(Connect()));
    ASSERT_OK(conn_->ExecuteFormat("CREATE EXTENSION documentdb CASCADE"));
    ASSERT_OK(conn_->Execute("SET search_path TO documentdb_api, documentdb_core"));
    ASSERT_OK(conn_->Execute("SET documentdb_core.bsonUseEJson TO TRUE"));
  }

  std::unique_ptr<pgwrapper::PGConn> conn_;
};

TEST_F(DocumentDBTest, SimpleCollection) {
  const auto db_name = "documentdb";
  const auto collection_name = "patient";
  const auto patient_1 = "P001";
  const auto patient_2 = "P002";

  // Insert 5 documents into patient.
  ASSERT_OK(conn_->FetchFormat(
      R"(
  SELECT documentdb_api.insert('$0', '{"insert":"$1", "documents":[
    { "patient_id": "$2", "name": "Alice Smith", "age": 30, "phone_number": "555-0123",
        "registration_year": "2023","conditions": ["Diabetes", "Hypertension"]},
    { "patient_id": "$3", "name": "Bob Johnson", "age": 45, "phone_number": "555-0456",
        "registration_year": "2023", "conditions": ["Asthma"]},
    { "patient_id": "P003", "name": "Charlie Brown", "age": 29, "phone_number": "555-0789",
        "registration_year": "2024", "conditions": ["Allergy", "Anemia"]},
    { "patient_id": "P004", "name": "Diana Prince", "age": 40, "phone_number": "555-0987",
        "registration_year": "2024", "conditions": ["Migraine"]},
    { "patient_id": "P005", "name": "Edward Norton", "age": 55, "phone_number": "555-1111",
        "registration_year": "2025", "conditions": ["Hypertension", "Heart Disease"]}]}');
  )",
      db_name, collection_name, patient_1, patient_2));

  auto get_document_count = [&]() {
    return CHECK_RESULT(conn_->FetchRow<int64_t>(Format(
        "SELECT count(*) FROM documentdb_api.collection('$0','$1')", db_name, collection_name)));
  };

  ASSERT_EQ(get_document_count(), 5);

  // Update 1 document.
  auto get_patient_age = [&](const std::string& patient_id) {
    auto age_str = CHECK_RESULT(conn_->FetchRow<std::string>(Format(
        R"(
      SELECT (((cursorpage->>'cursor')::bson->>'firstBatch')::bson->>'0')::bson->>'age'
        FROM documentdb_api.find_cursor_first_page('$0', '{ "find" : "$1",
          "filter" : {"patient_id":"$2"}}');
      )",
        db_name, collection_name, patient_id)));

    return std::stoi(age_str);
  };

  ASSERT_EQ(get_patient_age(patient_1), 30);
  ASSERT_EQ(get_patient_age(patient_2), 45);

  ASSERT_OK(conn_->FetchFormat(
      R"(
  SELECT documentdb_api.update('$0', '{"update":"$1",
      "updates":[{"q":{"patient_id":"$2"},"u":{"$$set":{"age":14}}}]}')
  )",
      db_name, collection_name, patient_1));

  ASSERT_EQ(get_patient_age(patient_1), 14);
  ASSERT_EQ(get_patient_age(patient_2), 45);

  // Update all documents.
  ASSERT_OK(conn_->FetchFormat(
      R"(
    SELECT documentdb_api.update('$0', '{"update":"$1",
      "updates":[{"q":{},"u":{"$$set":{"age":24}},"multi":true}]}')
    )",
      db_name, collection_name));

  ASSERT_EQ(get_patient_age(patient_1), 24);
  ASSERT_EQ(get_patient_age(patient_2), 24);

  // Delete one documents.
  ASSERT_OK(conn_->FetchFormat(
      R"(
    SELECT documentdb_api.delete('$0', '{"delete": "$1",
      "deletes": [{"q": {"patient_id": "$2"}, "limit": 1}]}')
    )",
      db_name, collection_name, patient_2));

  ASSERT_EQ(get_document_count(), 4);
}

// Validates that primary key (_id) range queries with $gt and $lt return correct results.
// This exercises the BSON comparison logic on the primary key, which is stored as a range
// key in DocDB. Numeric _id values including negative numbers must sort correctly
// (e.g., -10 < -1 < 0 < 5 < 42), which requires proper BSON comparison rather than
// byte-wise ordering of little-endian integers.
TEST_F(DocumentDBTest, PrimaryKeyRangeQuery) {
  const auto db_name = "pktest";
  const auto collection_name = "numbers";

  // Insert documents with numeric _id values, including negatives.
  // BSON stores int32 as little-endian, so byte-wise comparison would give wrong order
  // for negative values (e.g., -1 = 0xFFFFFFFF would sort after 1 = 0x01000000).
  ASSERT_OK(conn_->FetchFormat(
      R"(
  SELECT documentdb_api.insert('$0', '{"insert":"$1", "documents":[
    { "_id": -10, "label": "neg10" },
    { "_id": -1,  "label": "neg1" },
    { "_id": 0,   "label": "zero" },
    { "_id": 5,   "label": "five" },
    { "_id": 42,  "label": "fortytwo" },
    { "_id": 100, "label": "hundred" }]}');
  )",
      db_name, collection_name));

  // Helper: get the label of the single document matching a filter.
  auto get_label = [&](const std::string& filter) {
    return CHECK_RESULT(conn_->FetchRow<std::string>(Format(
        R"(
      SELECT (((cursorpage->>'cursor')::bson->>'firstBatch')::bson->>'0')::bson->>'label'
        FROM documentdb_api.find_cursor_first_page('$0',
          '{ "find": "$1", "filter": $2 }');
      )",
        db_name, collection_name, filter)));
  };

  // Helper: count documents matching a filter using documentdb_api.count_query.
  auto count_matching = [&](const std::string& filter) {
    auto n_str = CHECK_RESULT(conn_->FetchRow<std::string>(Format(
        R"(
      SELECT document->>'n'
        FROM documentdb_api.count_query('$0',
          '{ "count": "$1", "query": $2 }');
      )",
        db_name, collection_name, filter)));
    return std::stol(n_str);
  };

  // Verify total count.
  ASSERT_EQ(
      ASSERT_RESULT(conn_->FetchRow<int64_t>(Format(
          "SELECT count(*) FROM documentdb_api.collection('$0','$1')", db_name, collection_name))),
      6);

  // Test exact _id lookup for a negative value.
  EXPECT_EQ(get_label(R"({"_id": -1})"), "neg1");
  EXPECT_EQ(get_label(R"({"_id": 0})"), "zero");
  EXPECT_EQ(get_label(R"({"_id": 42})"), "fortytwo");

  // Test $gt: _id > 0 should return 3 documents (5, 42, 100).
  EXPECT_EQ(count_matching(R"({"_id": {"$gt": 0}})"), 3);

  // Test $lt: _id < 0 should return 2 documents (-10, -1).
  EXPECT_EQ(count_matching(R"({"_id": {"$lt": 0}})"), 2);

  // Test $gt with negative boundary: _id > -5 should return 5 documents (-1, 0, 5, 42, 100).
  EXPECT_EQ(count_matching(R"({"_id": {"$gt": -5}})"), 5);

  // Test $lt with positive boundary: _id < 42 should return 4 documents (-10, -1, 0, 5).
  EXPECT_EQ(count_matching(R"({"_id": {"$lt": 42}})"), 4);

  // Test combined $gt and $lt: -1 < _id < 42 should return 2 documents (0, 5).
  EXPECT_EQ(count_matching(R"({"_id": {"$gt": -1, "$lt": 42}})"), 2);

  // Test $gte and $lte: -1 <= _id <= 5 should return 3 documents (-1, 0, 5).
  EXPECT_EQ(count_matching(R"({"_id": {"$gte": -1, "$lte": 5}})"), 3);

  // Test $gte at lower bound: _id >= -10 should return all 6 documents.
  EXPECT_EQ(count_matching(R"({"_id": {"$gte": -10}})"), 6);

  // Test $lte at upper bound: _id <= 100 should return all 6 documents.
  EXPECT_EQ(count_matching(R"({"_id": {"$lte": 100}})"), 6);

  // Test range that excludes everything: _id > 100 should return 0.
  EXPECT_EQ(count_matching(R"({"_id": {"$gt": 100}})"), 0);

  // Test range that excludes everything: _id < -10 should return 0.
  EXPECT_EQ(count_matching(R"({"_id": {"$lt": -10}})"), 0);
}

}  // namespace yb
