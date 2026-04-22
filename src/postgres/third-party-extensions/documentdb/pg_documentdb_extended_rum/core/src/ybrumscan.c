/*-------------------------------------------------------------------------
 *
 * ybrumscan.c
 *    YugabyteDB scan routines for the RUM index access method.
 *    Follows the same pattern as ybginscan.c and ybginget.c.
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
#include "access/relscan.h"
#include "access/sysattr.h"
#include "access/yb_scan.h"
#include "miscadmin.h"
#include "pg_yb_utils.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "yb/yql/pggate/ybc_pggate.h"

/*
 * Extended opaque data for YB RUM scans.  Contains the base RumScanOpaqueData
 * plus YB-specific fields.  The base struct MUST be first so that pointers
 * can be cast between the two types.
 */
typedef struct YbRumScanOpaqueData
{
	RumScanOpaqueData rum_scan_opaque;
	YbcPgStatement handle;
	bool		is_exec_done;
} YbRumScanOpaqueData;

typedef YbRumScanOpaqueData *YbRumScanOpaque;


/*
 * Begin a YB RUM scan.  Allocates the extended opaque and initialises the
 * RUM state exactly as rumbeginscan() does.
 */
PGDLLEXPORT IndexScanDesc
ybrumbeginscan(Relation rel, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	RumScanOpaque so;
	MemoryContext prev = CurrentMemoryContext;

	scan = RelationGetIndexScan(rel, nkeys, norderbys);

	/* Allocate the extended YB opaque (contains RumScanOpaqueData at start). */
	so = (RumScanOpaque) palloc0(sizeof(YbRumScanOpaqueData));
	so->sortstate = NULL;
	so->keys = NULL;
	so->nkeys = 0;
	so->firstCall = true;
	so->totalentries = 0;
	so->sortedEntries = NULL;
	so->orderByScanData = NULL;
	so->scanLoops = 0;
	so->killedItems = NULL;
	so->numKilled = 0;
	so->killedItemsSkipped = 0;
	so->orderByKeyIndex = -1;
	so->orderScanDirection = ForwardScanDirection;
	so->tempCtx = RumContextCreate(CurrentMemoryContext,
								   "YbRum scan temporary context");
	so->keyCtx = RumContextCreate(CurrentMemoryContext,
								  "YbRum scan key context");
	so->rumStateCtx = RumContextCreate(CurrentMemoryContext,
									   "YbRum state context");

	MemoryContextSwitchTo(so->rumStateCtx);
	initRumState(&so->rumstate, scan->indexRelation);
	MemoryContextSwitchTo(prev);

	ItemPointerSetInvalid(&scan->xs_heaptid);
	scan->opaque = so;

	return scan;
}


/*
 * Rescan: set up RUM scan keys and create the YB select handle.
 */
PGDLLEXPORT void
ybrumrescan(IndexScanDesc scan, ScanKey scankey, int nscankeys,
			ScanKey orderbys, int norderbys)
{
	YbRumScanOpaque ybso = (YbRumScanOpaque) scan->opaque;

	/* Initialise the RUM scan key machinery (like rumrescan). */
	rumrescan(scan, scankey, nscankeys, orderbys, norderbys);

	/*
	 * Only create the YB select handle when heapRelation is available.
	 * During cost estimation or internal helper scans (e.g.
	 * RumGetMultiKeyStatusSlow), rescan may be called without a heap
	 * relation.  Skip the YB handle setup in that case.
	 */
	if (scan->heapRelation != NULL)
	{
		YbcPgPrepareParameters prepare_params;
		memset(&prepare_params, 0, sizeof(prepare_params));
		prepare_params.index_relfilenode_oid =
			YbGetRelfileNodeId(scan->indexRelation);
		prepare_params.index_only_scan = scan->xs_want_itup;
		prepare_params.embedded_idx =
			YbIsScanningEmbeddedIdx(scan->heapRelation,
									scan->indexRelation);

		ybso->handle = YbNewSelect(scan->heapRelation, &prepare_params);

		YbApplyPrimaryPushdown(ybso->handle, scan->yb_rel_pushdown);
		YbApplySecondaryIndexPushdown(ybso->handle, scan->yb_idx_pushdown);
	}

	ybso->is_exec_done = false;
}


/*
 * Set up targets for the YB select – we target all base table columns.
 */
static void
ybrumSetupTargets(IndexScanDesc scan)
{
	TupleDesc	tupdesc;
	YbRumScanOpaque ybso = (YbRumScanOpaque) scan->opaque;

	tupdesc = RelationGetDescr(scan->heapRelation);

	/* Always request the ybctid. */
	YbDmlAppendTargetSystem(YBTupleIdAttributeNumber, ybso->handle);

	/* Target all non-dropped columns of the base table. */
	for (AttrNumber attnum = 1; attnum <= tupdesc->natts; attnum++)
	{
		if (!TupleDescAttr(tupdesc, attnum - 1)->attisdropped)
			YbDmlAppendTargetRegular(tupdesc, attnum, ybso->handle);
	}
}


/*
 * Execute the YB select for the first time.
 */
static void
ybrumExecSelect(IndexScanDesc scan, ScanDirection dir)
{
	YbRumScanOpaque ybso = (YbRumScanOpaque) scan->opaque;

	Assert(!ScanDirectionIsBackward(dir));
	if (ScanDirectionIsForward(dir))
		HandleYBStatus(YBCPgSetForwardScan(ybso->handle, true));

	HandleYBStatus(YBCPgExecSelect(ybso->handle, NULL /* exec_params */));
}


/*
 * First-call setup: apply pushdowns, set targets, execute.
 * Returns false if the scan is void (unsatisfiable query).
 */
static bool
ybrumDoFirstExec(IndexScanDesc scan, ScanDirection dir)
{
	YbRumScanOpaque ybso = (YbRumScanOpaque) scan->opaque;

	/* targets */
	if (scan->yb_aggrefs != NIL)
		YbDmlAppendTargetsAggregate(scan->yb_aggrefs,
									NULL,
									RelationGetDescr(scan->indexRelation),
									scan->indexRelation,
									scan->xs_want_itup,
									ybso->handle);
	else
		ybrumSetupTargets(scan);

	YbSetCatalogCacheVersion(ybso->handle, YbGetCatalogCacheVersion());

	/* execute */
	ybrumExecSelect(scan, dir);

	return true;
}


/*
 * Fetch the next heap tuple from the YB result set.
 */
static HeapTuple
ybrumFetchNextHeapTuple(IndexScanDesc scan)
{
	bool		has_data = false;
	bool	   *nulls;
	Datum	   *values;
	HeapTuple	tuple = NULL;
	TupleDesc	tupdesc;
	YbcPgSysColumns syscols;
	YbRumScanOpaque ybso = (YbRumScanOpaque) scan->opaque;

	tupdesc = RelationGetDescr(scan->heapRelation);
	nulls = (bool *) palloc(tupdesc->natts * sizeof(bool));
	values = (Datum *) palloc0(tupdesc->natts * sizeof(Datum));

	HandleYBStatus(YBCPgDmlFetch(ybso->handle,
								 tupdesc->natts,
								 (uint64_t *) values,
								 nulls,
								 &syscols,
								 &has_data));
	if (has_data)
	{
		tuple = heap_form_tuple(tupdesc, values, nulls);
		tuple->t_tableOid = RelationGetRelid(scan->heapRelation);
		if (syscols.ybctid != NULL)
			HEAPTUPLE_YBCTID(tuple) = PointerGetDatum(syscols.ybctid);
	}
	pfree(values);
	pfree(nulls);

	return tuple;
}


/*
 * Get the next matching tuple.
 *
 * If the YB handle was not set (e.g. during cost estimation when
 * RumGetMultiKeyStatusSlow calls the scan path without a heap relation),
 * fall back to the vanilla rumgettuple.
 */
PGDLLEXPORT bool
ybrumgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HeapTuple	tup;
	YbRumScanOpaque ybso = (YbRumScanOpaque) scan->opaque;

	/*
	 * If no YB handle was set up (e.g. internal scan during cost estimation
	 * without a heap relation), return no results.  This is conservative but
	 * safe - it just means multi-key optimizations won't be applied.
	 */
	if (ybso->handle == NULL)
		return false;

	Assert(!ScanDirectionIsBackward(dir));

	if (!ybso->is_exec_done)
	{
		if (!ybrumDoFirstExec(scan, dir))
			return false;
		ybso->is_exec_done = true;
	}

	/* Aggregate pushdown path. */
	if (scan->yb_aggrefs)
	{
		scan->xs_recheck = true;
		return ybc_getnext_aggslot(scan, ybso->handle, scan->xs_want_itup);
	}

	while (HeapTupleIsValid(tup = ybrumFetchNextHeapTuple(scan)))
	{
		scan->xs_hitup = tup;
		scan->xs_hitupdesc = RelationGetDescr(scan->heapRelation);
		scan->xs_recheck = true;
		return true;
	}

	return false;
}


/*
 * End the scan.  Delegate to rumendscan for RUM-specific cleanup;
 * the YB handle does not need special cleanup.
 */
PGDLLEXPORT void
ybrumendscan(IndexScanDesc scan)
{
	rumendscan(scan);
}
