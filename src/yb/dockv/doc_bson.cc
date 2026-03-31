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

#include "yb/dockv/doc_bson.h"

#include <string.h>

#include <algorithm>
#include <cmath>

#include "yb/dockv/doc_kv_util.h"
#include "yb/util/result.h"

namespace yb::dockv {

namespace {

// BSON element type constants.
constexpr uint8_t kBsonTypeDouble = 0x01;
constexpr uint8_t kBsonTypeString = 0x02;
constexpr uint8_t kBsonTypeDocument = 0x03;
constexpr uint8_t kBsonTypeArray = 0x04;
constexpr uint8_t kBsonTypeBinary = 0x05;
constexpr uint8_t kBsonTypeUndefined = 0x06;
constexpr uint8_t kBsonTypeObjectId = 0x07;
constexpr uint8_t kBsonTypeBoolean = 0x08;
constexpr uint8_t kBsonTypeDatetime = 0x09;
constexpr uint8_t kBsonTypeNull = 0x0A;
constexpr uint8_t kBsonTypeRegex = 0x0B;
constexpr uint8_t kBsonTypeSymbol = 0x0E;
constexpr uint8_t kBsonTypeInt32 = 0x10;
constexpr uint8_t kBsonTypeTimestamp = 0x11;
constexpr uint8_t kBsonTypeInt64 = 0x12;
constexpr uint8_t kBsonTypeDecimal128 = 0x13;
constexpr uint8_t kBsonTypeMinKey = 0xFF;
constexpr uint8_t kBsonTypeMaxKey = 0x7F;

// Returns the canonical type order for BSON comparison, following MongoDB's ordering:
// MinKey < Null < Number < String < Object < Array < BinData < ObjectId <
// Boolean < Date < Timestamp < RegEx < MaxKey
int BsonTypeOrder(uint8_t bson_type) {
  switch (bson_type) {
    case kBsonTypeMinKey:     return -1;
    case kBsonTypeNull:       return 1;
    case kBsonTypeUndefined:  return 1;  // Deprecated, treated as null.
    case kBsonTypeDouble:     return 2;
    case kBsonTypeInt32:      return 2;
    case kBsonTypeInt64:      return 2;
    case kBsonTypeDecimal128: return 2;
    case kBsonTypeString:     return 3;
    case kBsonTypeSymbol:     return 3;  // Deprecated, treated as string.
    case kBsonTypeDocument:   return 4;
    case kBsonTypeArray:      return 5;
    case kBsonTypeBinary:     return 6;
    case kBsonTypeObjectId:   return 7;
    case kBsonTypeBoolean:    return 8;
    case kBsonTypeDatetime:   return 9;
    case kBsonTypeTimestamp:  return 10;
    case kBsonTypeRegex:      return 11;
    case kBsonTypeMaxKey:     return 12;
    default:                  return 100;
  }
}

int32_t ReadLittleEndian32(const uint8_t* p) {
  int32_t result;
  memcpy(&result, p, sizeof(result));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  result = __builtin_bswap32(result);
#endif
  return result;
}

int64_t ReadLittleEndian64(const uint8_t* p) {
  int64_t result;
  memcpy(&result, p, sizeof(result));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  result = __builtin_bswap64(result);
#endif
  return result;
}

uint32_t ReadLittleEndianU32(const uint8_t* p) {
  uint32_t result;
  memcpy(&result, p, sizeof(result));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  result = __builtin_bswap32(result);
#endif
  return result;
}

double ReadLittleEndianDouble(const uint8_t* p) {
  double result;
  memcpy(&result, p, sizeof(result));
  return result;
}

// Returns the size in bytes of a BSON element's value (not including type byte and field name).
size_t BsonElementValueSize(uint8_t type, const uint8_t* data, const uint8_t* end) {
  switch (type) {
    case kBsonTypeDouble:
      return 8;
    case kBsonTypeString:
    case kBsonTypeSymbol:
      // int32 string_length + string bytes (including null terminator).
      if (data + 4 <= end) {
        return 4 + static_cast<size_t>(ReadLittleEndian32(data));
      }
      return end - data;
    case kBsonTypeDocument:
    case kBsonTypeArray:
      // Document size includes the size field itself.
      if (data + 4 <= end) {
        return static_cast<size_t>(ReadLittleEndian32(data));
      }
      return end - data;
    case kBsonTypeBinary:
      // int32 length + subtype byte + binary data.
      if (data + 4 <= end) {
        return 5 + static_cast<size_t>(ReadLittleEndian32(data));
      }
      return end - data;
    case kBsonTypeObjectId:
      return 12;
    case kBsonTypeBoolean:
      return 1;
    case kBsonTypeDatetime:
      return 8;
    case kBsonTypeNull:
    case kBsonTypeUndefined:
    case kBsonTypeMinKey:
    case kBsonTypeMaxKey:
      return 0;
    case kBsonTypeRegex: {
      // Two C strings (pattern + options).
      const uint8_t* p = data;
      while (p < end && *p) p++;
      if (p < end) p++;  // Skip null terminator of pattern.
      while (p < end && *p) p++;
      if (p < end) p++;  // Skip null terminator of options.
      return p - data;
    }
    case kBsonTypeInt32:
      return 4;
    case kBsonTypeTimestamp:
      return 8;
    case kBsonTypeInt64:
      return 8;
    case kBsonTypeDecimal128:
      return 16;
    default:
      return end - data;
  }
}

// Compares two BSON numeric values, handling cross-type comparison.
int CompareNumbers(uint8_t type_a, const uint8_t* data_a,
                   uint8_t type_b, const uint8_t* data_b) {
  // Same-type comparison for exact precision.
  if (type_a == type_b) {
    switch (type_a) {
      case kBsonTypeInt32: {
        int32_t a = ReadLittleEndian32(data_a);
        int32_t b = ReadLittleEndian32(data_b);
        return (a < b) ? -1 : (a > b) ? 1 : 0;
      }
      case kBsonTypeInt64: {
        int64_t a = ReadLittleEndian64(data_a);
        int64_t b = ReadLittleEndian64(data_b);
        return (a < b) ? -1 : (a > b) ? 1 : 0;
      }
      case kBsonTypeDouble: {
        double a = ReadLittleEndianDouble(data_a);
        double b = ReadLittleEndianDouble(data_b);
        // NaN handling: NaN is considered equal to NaN and less than all other numbers.
        if (std::isnan(a) && std::isnan(b)) return 0;
        if (std::isnan(a)) return -1;
        if (std::isnan(b)) return 1;
        return (a < b) ? -1 : (a > b) ? 1 : 0;
      }
      default:
        break;
    }
  }

  // Cross-type numeric comparison.
  // For int64 vs int32, compare as int64.
  if ((type_a == kBsonTypeInt64 || type_a == kBsonTypeInt32) &&
      (type_b == kBsonTypeInt64 || type_b == kBsonTypeInt32)) {
    int64_t a = (type_a == kBsonTypeInt64) ? ReadLittleEndian64(data_a)
                                           : static_cast<int64_t>(ReadLittleEndian32(data_a));
    int64_t b = (type_b == kBsonTypeInt64) ? ReadLittleEndian64(data_b)
                                           : static_cast<int64_t>(ReadLittleEndian32(data_b));
    return (a < b) ? -1 : (a > b) ? 1 : 0;
  }

  // For comparisons involving double, use double.
  // Note: this may lose precision for very large int64 values, matching MongoDB behavior.
  auto to_double = [](uint8_t type, const uint8_t* data) -> double {
    switch (type) {
      case kBsonTypeDouble: return ReadLittleEndianDouble(data);
      case kBsonTypeInt32: return static_cast<double>(ReadLittleEndian32(data));
      case kBsonTypeInt64: return static_cast<double>(ReadLittleEndian64(data));
      default: return 0.0;
    }
  };

  double a = to_double(type_a, data_a);
  double b = to_double(type_b, data_b);
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (std::isnan(a)) return -1;
  if (std::isnan(b)) return 1;
  return (a < b) ? -1 : (a > b) ? 1 : 0;
}

// Forward declaration for recursive comparison.
int CompareBsonDocuments(const uint8_t* a_data, const uint8_t* a_end,
                         const uint8_t* b_data, const uint8_t* b_end);

// Compares two BSON element values given their types and value data pointers.
int CompareBsonElementValues(uint8_t type_a, const uint8_t* data_a, const uint8_t* end_a,
                             uint8_t type_b, const uint8_t* data_b, const uint8_t* end_b) {
  int order_a = BsonTypeOrder(type_a);
  int order_b = BsonTypeOrder(type_b);

  if (order_a != order_b) {
    return (order_a < order_b) ? -1 : 1;
  }

  // Same canonical type - compare values.
  if (order_a == 2) {
    // Numeric types - use cross-type numeric comparison.
    return CompareNumbers(type_a, data_a, type_b, data_b);
  }

  // Use type_a for switch since both have the same canonical order.
  switch (order_a) {
    case -1:  // MinKey
    case 12:  // MaxKey
    case 1:   // Null/Undefined
      return 0;

    case 3: {  // String/Symbol
      int32_t len_a = ReadLittleEndian32(data_a);
      int32_t len_b = ReadLittleEndian32(data_b);
      // Length includes the null terminator. Compare the string content.
      int32_t min_content_len = std::min(len_a, len_b) - 1;
      if (min_content_len > 0) {
        int cmp = memcmp(data_a + 4, data_b + 4, min_content_len);
        if (cmp != 0) return (cmp < 0) ? -1 : 1;
      }
      return (len_a < len_b) ? -1 : (len_a > len_b) ? 1 : 0;
    }

    case 4:   // Document
    case 5: {  // Array (compared as document)
      int32_t size_a = ReadLittleEndian32(data_a);
      int32_t size_b = ReadLittleEndian32(data_b);
      return CompareBsonDocuments(data_a, data_a + size_a, data_b, data_b + size_b);
    }

    case 6: {  // Binary
      int32_t len_a = ReadLittleEndian32(data_a);
      int32_t len_b = ReadLittleEndian32(data_b);
      // Compare length first.
      if (len_a != len_b) return (len_a < len_b) ? -1 : 1;
      // Compare subtype.
      if (data_a[4] != data_b[4]) return (data_a[4] < data_b[4]) ? -1 : 1;
      // Compare binary data.
      if (len_a > 0) {
        int cmp = memcmp(data_a + 5, data_b + 5, len_a);
        if (cmp != 0) return (cmp < 0) ? -1 : 1;
      }
      return 0;
    }

    case 7: {  // ObjectId (12 bytes, big-endian comparison).
      int cmp = memcmp(data_a, data_b, 12);
      return (cmp < 0) ? -1 : (cmp > 0) ? 1 : 0;
    }

    case 8: {  // Boolean
      return (data_a[0] < data_b[0]) ? -1 : (data_a[0] > data_b[0]) ? 1 : 0;
    }

    case 9: {  // DateTime (signed int64 milliseconds since epoch).
      int64_t a = ReadLittleEndian64(data_a);
      int64_t b = ReadLittleEndian64(data_b);
      return (a < b) ? -1 : (a > b) ? 1 : 0;
    }

    case 10: {  // Timestamp (uint64: increment in low 32 bits, seconds in high 32 bits).
      // Compare seconds first (high 32 bits), then increment (low 32 bits).
      uint32_t a_secs = ReadLittleEndianU32(data_a + 4);
      uint32_t b_secs = ReadLittleEndianU32(data_b + 4);
      if (a_secs != b_secs) return (a_secs < b_secs) ? -1 : 1;
      uint32_t a_inc = ReadLittleEndianU32(data_a);
      uint32_t b_inc = ReadLittleEndianU32(data_b);
      return (a_inc < b_inc) ? -1 : (a_inc > b_inc) ? 1 : 0;
    }

    case 11: {  // Regex (two C strings: pattern + options).
      int cmp = strcmp(reinterpret_cast<const char*>(data_a),
                       reinterpret_cast<const char*>(data_b));
      if (cmp != 0) return (cmp < 0) ? -1 : 1;
      const char* opts_a = reinterpret_cast<const char*>(data_a) +
                           strlen(reinterpret_cast<const char*>(data_a)) + 1;
      const char* opts_b = reinterpret_cast<const char*>(data_b) +
                           strlen(reinterpret_cast<const char*>(data_b)) + 1;
      cmp = strcmp(opts_a, opts_b);
      return (cmp < 0) ? -1 : (cmp > 0) ? 1 : 0;
    }

    default:
      break;
  }

  return 0;
}

// Compares two BSON documents element by element.
int CompareBsonDocuments(const uint8_t* a_data, const uint8_t* a_end,
                         const uint8_t* b_data, const uint8_t* b_end) {
  // Skip the 4-byte document size.
  const uint8_t* ap = a_data + 4;
  const uint8_t* bp = b_data + 4;

  while (ap < a_end && bp < b_end) {
    // Check for document terminator (0x00).
    if (*ap == 0x00 && *bp == 0x00) return 0;
    if (*ap == 0x00) return -1;
    if (*bp == 0x00) return 1;

    // Read element types.
    uint8_t a_type = *ap++;
    uint8_t b_type = *bp++;

    // Read and compare field names (C strings).
    int name_cmp = strcmp(reinterpret_cast<const char*>(ap),
                          reinterpret_cast<const char*>(bp));

    // Skip past field names.
    while (ap < a_end && *ap) ap++;
    if (ap < a_end) ap++;
    while (bp < b_end && *bp) bp++;
    if (bp < b_end) bp++;

    // If field names differ, return the field name comparison.
    if (name_cmp != 0) {
      return (name_cmp < 0) ? -1 : 1;
    }

    // Compare element values.
    int cmp = CompareBsonElementValues(a_type, ap, a_end, b_type, bp, b_end);
    if (cmp != 0) return cmp;

    // Advance past values.
    ap += BsonElementValueSize(a_type, ap, a_end);
    bp += BsonElementValueSize(b_type, bp, b_end);
  }

  // Both exhausted = equal; otherwise, shorter document is less.
  if (ap >= a_end && bp >= b_end) return 0;
  return (ap >= a_end) ? -1 : 1;
}

}  // namespace

int CompareBson(Slice a, Slice b) {
  // Minimum valid BSON document is 5 bytes: int32 size + 0x00 terminator.
  if (a.size() < 5 || b.size() < 5) {
    return a.compare(b);
  }

  const auto* a_data = reinterpret_cast<const uint8_t*>(a.data());
  const auto* b_data = reinterpret_cast<const uint8_t*>(b.data());

  int32_t a_size = ReadLittleEndian32(a_data);
  int32_t b_size = ReadLittleEndian32(b_data);

  const uint8_t* a_end = a_data + std::min(static_cast<int32_t>(a.size()), a_size);
  const uint8_t* b_end = b_data + std::min(static_cast<int32_t>(b.size()), b_size);

  return CompareBsonDocuments(a_data, a_end, b_data, b_end);
}

// YB_TODO: Currently mapping BSON keys to string encoding, which is not correct. These functions
// need to be updated, so that it gets converted such that the binary representation matches the
// BSON sort order.

void BsonKeyToComparableBinary(Slice slice, KeyBuffer& dest) {
  ZeroEncodeAndAppendStrToKey(slice, dest);
}

void BsonKeyToComparableBinaryDescending(Slice slice, KeyBuffer& dest) {
  ComplementZeroEncodeAndAppendStrToKey(slice, dest);
}

Status BsonKeyFromComparableBinary(Slice* slice, std::string* result) {
  return DecodeZeroEncodedStr(slice, result);
}

Result<const char*> BsonKeyFromComparableBinary(
    const char* begin, const char* end, ValueBuffer* out) {
  return DecodeZeroEncodedStr(begin, end, out);
}

Result<const char*> SkipComparableBson(const char* begin, const char* end) {
  return SkipZeroEncodedStr(begin, end);
}

Status BsonKeyFromComparableBinaryDescending(Slice* slice, std::string* result) {
  return DecodeComplementZeroEncodedStr(slice, result);
}

Result<const char*> BsonKeyFromComparableBinaryDescending(
    const char* begin, const char* end, ValueBuffer* out) {
  return DecodeComplementZeroEncodedStr(begin, end, out);
}

Result<const char*> SkipComparableBsonDescending(const char* begin, const char* end) {
  return SkipComplementZeroEncodedStr(begin, end);
}

}  // namespace yb::dockv
