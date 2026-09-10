/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// sirius
#include <helper/logical_type.hpp>
#include <tae/tae_format.hpp>

// tae-scanner
#include "tae_filter.hpp"   // tae::PushedFilter
#include "tae_scanner.hpp"  // tae::TAEScanBindData

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/table_filter.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Canonical plan for a TAE scan.
 *
 * Built once by the GPU TAE scan operator from DuckDB planner inputs
 * (column_ids, projection_ids, returned_types, table_filters) and the
 * TAE bind data (all_col_names, all_col_mo_oids, sort_column_idx).
 *
 * All downstream consumers — operator-side AST filter translation,
 * global-state filter extraction, and the per-task @c compute_task hot path —
 * read from this single structure instead of repeatedly looking the same
 * data up out of bind_data and the operator's parallel vectors.
 *
 * Three index spaces appear throughout the scan pipeline (mirroring
 * @c sirius::op::scan::scan_plan for parquet):
 *   P = primary index       (TAE seqnum; index into the bind data's
 *                            @c all_col_names / @c all_col_mo_oids /
 *                            and into the operator's @c returned_types)
 *   C = column_ids position (index into the operator's @c column_ids list)
 *   D = batch-output position
 *
 * @note On the D space: the TAE converter today produces decoded columns
 *       in ascending @c column_chunk_info::column_idx order via
 *       @c std::map<uint16_t, ...>. @c column_idx is C-space (it stores
 *       @c col_ids_position from the projection). So the converter's batch
 *       order is "C-sorted ascending across the projected columns" — an
 *       emergent ordering, not an explicit D-space mapping. This plan
 *       deliberately does NOT introduce an explicit D-space; instead it
 *       preserves today's semantics by exposing a @c batch_column_map sized
 *       by C and ordered to match that emergent C-sorted batch.
 */
struct tae_scan_plan {
  //===-------------------------------------------------------------------===//
  // P-indexed schema metadata (copied from TAEScanBindData)
  //===-------------------------------------------------------------------===//
  std::vector<std::string> all_col_names;
  std::vector<std::uint8_t> all_col_mo_oids;
  std::int32_t sort_column_idx = -1;

  //===-------------------------------------------------------------------===//
  // Projected columns, in projection order.
  //
  // Replaces the per-task @c projected_seqnums / @c projected_col_ids_positions
  // vectors that @c compute_task used to rebuild on every invocation, and
  // pre-resolves the per-column metadata that the per-chunk loop used to look
  // up out of @c mo_oids and @c returned_types.
  //
  // @c col_ids_position is C-space — it is written verbatim into
  // @c host_tae_representation::column_chunk_info::column_idx today.
  //===-------------------------------------------------------------------===//
  struct projected_column {
    std::uint16_t seqnum;            ///< P — TAE seqnum (== primary_idx)
    std::uint16_t col_ids_position;  ///< C — written into chunk.column_idx
    tae::MOTypeOid type_oid;         ///< pre-resolved from all_col_mo_oids[seqnum]
    std::int32_t width = 0;          ///< decimal precision (0 if not decimal)
    std::int32_t scale = 0;          ///< decimal scale (0 if not decimal)
  };
  std::vector<projected_column> projected_columns;

  //===-------------------------------------------------------------------===//
  // Post-filter projection IDs.
  //
  // Indices into the converter's emergent C-sorted decoded @c columns[]
  // vector. Empty means "no post-filter projection step".
  //===-------------------------------------------------------------------===//
  std::vector<std::size_t> post_filter_projection_ids;

  //===-------------------------------------------------------------------===//
  // Filter pushdown side
  //===-------------------------------------------------------------------===//

  /// C → C-sorted-batch-position map, with @c idx_t(-1) sentinel for "not in
  /// batch" — same form as @c sirius::op::build_batch_column_map. Cached so
  /// the operator constructor does not call the free function and the global
  /// state need not recompute it for any future filter manipulation.
  std::vector<duckdb::idx_t> batch_column_map;

  /// Zone-map filters extracted from the DuckDB @c TableFilterSet, keyed by
  /// TAE seqnum. Replaces the global state's @c _pushed_filters field.
  std::vector<tae::PushedFilter> pushed_filters;

  /// Deduped seqnums appearing in @c pushed_filters, preserving first-seen
  /// order. Precomputed so @c compute_task does not rebuild it per task.
  std::vector<std::uint16_t> filter_seqnums;
};

/**
 * @brief Build a @c tae_scan_plan from operator inputs and TAE bind data.
 *
 * @param bind_data         TAE scan bind data (schema + object list).
 * @param column_ids        Column ids exposed by the table function.
 * @param projection_ids    Indices into column_ids that the planner projects.
 *                          When non-empty, the first @c output_types_size
 *                          entries are output columns; the remaining entries
 *                          are pure-filter columns. When empty, every
 *                          column_ids entry is read and emitted in order.
 * @param returned_types    Sirius logical types parallel to bind_data's
 *                          @c all_col_types (P-indexed).
 * @param output_types_size The scan operator's @c types.size() — used to split
 *                          @c projection_ids into output vs. pure-filter ranges.
 * @param table_filters     DuckDB filter set, or nullptr.
 *
 * @throws sirius::internal_exception on virtual columns appearing in output
 *         positions, on duplicate primary indices in the projection, or on
 *         narrowing failures (seqnum / column_ids position exceeding uint16).
 */
tae_scan_plan build_tae_scan_plan(const tae::TAEScanBindData& bind_data,
                                  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
                                  const duckdb::vector<duckdb::idx_t>& projection_ids,
                                  const duckdb::vector<sirius::logical_type>& returned_types,
                                  std::size_t output_types_size,
                                  duckdb::TableFilterSet* table_filters);

}  // namespace sirius::op::scan
