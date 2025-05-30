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
import type { AnalysisIssueDetails } from './AnalysisIssueDetails';


/**
 * Sql obejcts type and count
 * @export
 * @interface SqlObjectsDetails
 */
export interface SqlObjectsDetails  {
  /**
   * 
   * @type {string}
   * @memberof SqlObjectsDetails
   */
  objectType?: string;
  /**
   * 
   * @type {number}
   * @memberof SqlObjectsDetails
   */
  totalCount?: number;
  /**
   * 
   * @type {number}
   * @memberof SqlObjectsDetails
   */
  invalidCount?: number;
  /**
   * 
   * @type {string}
   * @memberof SqlObjectsDetails
   */
  objectNames?: string;
  /**
   * 
   * @type {AnalysisIssueDetails[]}
   * @memberof SqlObjectsDetails
   */
  issues?: AnalysisIssueDetails[];
}



