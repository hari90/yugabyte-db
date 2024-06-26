// tslint:disable
/**
 * Yugabyte Cloud
 * YugabyteDB as a Service
 *
 * The version of the OpenAPI document: v1
 * Contact: support@yugabyte.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


// eslint-disable-next-line no-duplicate-imports
import type { LiveQueryResponseYSQLQueryItem } from './LiveQueryResponseYSQLQueryItem';


/**
 * Schema for Live Query Response YSQL Data
 * @export
 * @interface LiveQueryResponseYSQLData
 */
export interface LiveQueryResponseYSQLData  {
  /**
   * Count of Errors
   * @type {number}
   * @memberof LiveQueryResponseYSQLData
   */
  error_count?: number;
  /**
   * 
   * @type {LiveQueryResponseYSQLQueryItem[]}
   * @memberof LiveQueryResponseYSQLData
   */
  queries?: LiveQueryResponseYSQLQueryItem[];
}



