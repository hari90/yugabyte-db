/*-------------------------------------------------------------------------
 *
 * ybrumwrite.c
 *    YugabyteDB storage integration for RUM index write operations.
 *    Follows the same pattern as ybginwrite.c and ybvectorwrite.c.
 *
 * Copyright (c) YugabyteDB, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "pg_documentdb_rum.h"

#include "access/genam.h"
#include "access/sysattr.h"
#include "access/yb_scan.h"
#include "catalog/index.h"
#include "catalog/pg_type.h"
#include "catalog/yb_type.h"
#include "commands/yb_cmds.h"
#include "executor/ybModifyTable.h"
#include "nodes/plannodes.h"
#include "pg_yb_utils.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/relcache.h"

typedef struct
{
	RumState	rumstate;
	double		indtuples;
	MemoryContext tmpCtx;
	uint64_t   *backfilltime;
} YbRumBuildState;


/*
 * Bind callback for RUM index writes to DocDB.
 * Based on ybginwrite.c doBindsForIdxWrite and ybvectorwrite.c doBindsForIdx.
 */
static void
doBindsForRumWrite(YbcPgStatement stmt,
				   void *indexstate,
				   Relation index,
				   Datum *values,
				   bool *isnull,
				   int n_bound_atts,
				   Datum ybbasectid,
				   bool ybctid_as_value)
{
	RumState   *rumstate = (RumState *) indexstate;
	TupleDesc	tupdesc = RelationGetDescr(index);

	for (AttrNumber attnum = 1; attnum <= n_bound_atts; ++attnum)
	{
		Oid			type_id = GetTypeId(attnum, tupdesc);
		Oid			collation_id =
			YBEncodingCollation(stmt, attnum,
								rumstate->supportCollation[attnum - 1]);

		YbBindDatumToColumn(stmt, attnum, type_id, collation_id,
							values[attnum - 1], isnull[attnum - 1],
							&YBCGinNullTypeEntity);
	}

	/* Bind the base table ybctid. */
	bool		is_null = false;

	YbBindDatumToColumn(stmt,
						YBIdxBaseTupleIdAttributeNumber,
						BYTEAOID,
						InvalidOid,
						ybbasectid,
						is_null,
						NULL /* null_type_entity */);
}


/*
 * Bind callback for RUM index deletes.
 */
static void
doBindsForRumDelete(YbcPgStatement stmt,
					void *indexstate,
					Relation index,
					Datum *values,
					bool *isnull,
					int n_bound_atts,
					Datum ybbasectid,
					bool ybctid_as_value)
{
	/* For deletes, bind the same way as writes. */
	doBindsForRumWrite(stmt, indexstate, index, values, isnull,
					   n_bound_atts, ybbasectid, ybctid_as_value);
}


/*
 * Extract RUM entries for a single tuple and write them to DocDB.
 */
static void
ybrumTupleWrite(RumState *rumstate, OffsetNumber attnum,
				Relation index, Datum value, bool isNull,
				Datum ybctid, uint64_t *backfilltime, bool isinsert)
{
	Datum	   *entries;
	int32		nentries;
	RumNullCategory *categories;
	Datum	   *addInfo;
	bool	   *addInfoIsNull;
	int			i;

	entries = rumExtractEntries(rumstate, attnum, value, isNull,
							   &nentries, &categories,
							   &addInfo, &addInfoIsNull);

	for (i = 0; i < nentries; i++)
	{
		/*
		 * Build the index tuple values. For a single-column RUM index, the
		 * tuple has: [key_value]. For multi-column: [attnum, key_value].
		 * We build based on the index tuple descriptor.
		 */
		int			natts = RelationGetNumberOfAttributes(index);
		Datum	   *idx_values = palloc0(sizeof(Datum) * natts);
		bool	   *idx_isnull = palloc0(sizeof(bool) * natts);

		if (rumstate->oneCol)
		{
			/* Single-column index: first att is the key */
			idx_values[0] = entries[i];
			idx_isnull[0] = (categories[i] != RUM_CAT_NORM_KEY);
		}
		else
		{
			/* Multi-column index: first att is column number, second is key */
			idx_values[0] = UInt16GetDatum(attnum);
			idx_isnull[0] = false;
			idx_values[1] = entries[i];
			idx_isnull[1] = (categories[i] != RUM_CAT_NORM_KEY);
		}

		if (isinsert)
			YBCExecuteInsertIndex(index, idx_values, idx_isnull, ybctid,
								  backfilltime,
								  doBindsForRumWrite, (void *) rumstate);
		else
			YBCExecuteDeleteIndex(index, idx_values, idx_isnull, ybctid,
								  doBindsForRumDelete, (void *) rumstate);

		pfree(idx_values);
		pfree(idx_isnull);
	}
}


/*
 * Callback for index build - processes each heap tuple.
 */
static void
ybrumBuildCallback(Relation index, Datum ybctid, Datum *values, bool *isnull,
				   bool tupleIsAlive, void *state)
{
	YbRumBuildState *buildstate = (YbRumBuildState *) state;
	MemoryContext oldCtx;
	OffsetNumber attnum;
	int			nkeys = buildstate->rumstate.oneCol ? 1 :
		buildstate->rumstate.origTupdesc->natts;

	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	for (attnum = 1; attnum <= nkeys; attnum++)
	{
		ybrumTupleWrite(&buildstate->rumstate, attnum, index,
						values[attnum - 1], isnull[attnum - 1],
						ybctid, buildstate->backfilltime, true /* isinsert */);
	}

	buildstate->indtuples += 1;

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}


/*
 * Common build code for both ybrumbuild and ybrumbackfill.
 */
static IndexBuildResult *
ybrumBuildCommon(Relation heap, Relation index, struct IndexInfo *indexInfo,
				 struct YbBackfillInfo *bfinfo,
				 struct YbPgExecOutParam *bfresult)
{
	IndexBuildResult *result;
	double		reltuples;
	YbRumBuildState buildstate;

	initRumState(&buildstate.rumstate, index);
	buildstate.rumstate.isBuild = true;
	buildstate.indtuples = 0;
	buildstate.backfilltime = bfinfo ? &bfinfo->read_time : NULL;
	buildstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											  "YbRum build temporary context",
											  ALLOCSET_DEFAULT_SIZES);

	if (!bfinfo)
		reltuples = yb_table_index_build_scan(heap, index, indexInfo, true,
											  ybrumBuildCallback,
											  (void *) &buildstate,
											  NULL /* HeapScanDesc */);
	else
		reltuples = IndexBackfillHeapRangeScan(heap, index, indexInfo,
											   ybrumBuildCallback,
											   (void *) &buildstate,
											   bfinfo,
											   bfresult);

	MemoryContextDelete(buildstate.tmpCtx);

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}


/*
 * Common write code for insert and delete.
 */
static void
ybrumWrite(Relation index, Datum *values, bool *isnull, Datum ybctid,
		   Relation heap, struct IndexInfo *indexInfo, bool isinsert)
{
	RumState   *rumstate = (RumState *) indexInfo->ii_AmCache;
	MemoryContext oldCtx;
	MemoryContext writeCtx;
	OffsetNumber attnum;
	int			nkeys;

	if (rumstate == NULL)
	{
		oldCtx = MemoryContextSwitchTo(indexInfo->ii_Context);
		rumstate = (RumState *) palloc(sizeof(RumState));
		initRumState(rumstate, index);
		indexInfo->ii_AmCache = (void *) rumstate;
		MemoryContextSwitchTo(oldCtx);
	}

	writeCtx = AllocSetContextCreate(CurrentMemoryContext,
									 "YbRum write temporary context",
									 ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(writeCtx);

	nkeys = rumstate->oneCol ? 1 : rumstate->origTupdesc->natts;

	for (attnum = 1; attnum <= nkeys; attnum++)
	{
		ybrumTupleWrite(rumstate, attnum, index,
						values[attnum - 1], isnull[attnum - 1],
						ybctid, NULL /* backfilltime */, isinsert);
	}

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(writeCtx);
}


/* Public API functions */

PGDLLEXPORT IndexBuildResult *
ybrumbuild(Relation heap, Relation index, struct IndexInfo *indexInfo)
{
	return ybrumBuildCommon(heap, index, indexInfo, NULL, NULL);
}

PGDLLEXPORT void
ybrumbuildempty(Relation index)
{
	elog(WARNING, "Unexpected building of empty unlogged RUM index");
}

PGDLLEXPORT IndexBulkDeleteResult *
ybrumbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				IndexBulkDeleteCallback callback, void *callback_state)
{
	elog(WARNING, "Unexpected bulk delete of RUM index via vacuum");
	return NULL;
}

PGDLLEXPORT IndexBulkDeleteResult *
ybrumvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	return stats;
}

PGDLLEXPORT bool
ybruminsert(Relation index, Datum *values, bool *isnull, Datum ybctid,
			Relation heap, IndexUniqueCheck checkUnique,
			struct IndexInfo *indexInfo, bool shared_insert)
{
	ybrumWrite(index, values, isnull, ybctid, heap, indexInfo,
			   true /* isinsert */);
	return false;
}

PGDLLEXPORT void
ybrumdelete(Relation index, Datum *values, bool *isnull, Datum ybctid,
			Relation heap, struct IndexInfo *indexInfo)
{
	ybrumWrite(index, values, isnull, ybctid, heap, indexInfo,
			   false /* isinsert */);
}

PGDLLEXPORT IndexBuildResult *
ybrumbackfill(Relation heap, Relation index, struct IndexInfo *indexInfo,
			  struct YbBackfillInfo *bfinfo,
			  struct YbPgExecOutParam *bfresult)
{
	return ybrumBuildCommon(heap, index, indexInfo, bfinfo, bfresult);
}

PGDLLEXPORT void
ybrumbindschema(YbcPgStatement handle,
				struct IndexInfo *indexInfo,
				TupleDesc indexTupleDesc,
				int16 *coloptions,
				Oid *opclassOids,
				Datum reloptions)
{
	YBCBindCreateIndexColumns(handle, indexInfo, indexTupleDesc,
							  coloptions, indexInfo->ii_NumIndexKeyAttrs);
}

PGDLLEXPORT bool
ybrummightrecheck(Scan *scan, Relation heapRelation, Relation indexRelation,
				  bool xs_want_itup, ScanKey keys, int nkeys)
{
	/* RUM indexes always require recheck */
	return true;
}
