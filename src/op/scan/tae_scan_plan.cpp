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

#include <op/scan/tae_scan_plan.hpp>

// sirius
#include <log/logging.hpp>
#include <sirius/exception.hpp>

// standard library
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace sirius::op::scan {

namespace {

/// Build the C → C-sorted-batch-position map. Mirrors @c build_batch_column_map
/// in scan_utils.cpp so we can drop that call site from the operator ctor and
/// own the result on the plan.
std::vector<duckdb::idx_t> build_batch_column_map_local(
  const duckdb::vector<duckdb::idx_t>& projection_ids, std::size_t column_ids_count)
{
  constexpr auto NOT_PROJECTED = static_cast<duckdb::idx_t>(-1);
  std::vector<duckdb::idx_t> map(column_ids_count, NOT_PROJECTED);

  if (projection_ids.empty()) {
    for (std::size_t i = 0; i < column_ids_count; ++i) {
      map[i] = static_cast<duckdb::idx_t>(i);
    }
    return map;
  }

  std::vector<duckdb::idx_t> sorted(projection_ids.begin(), projection_ids.end());
  std::sort(sorted.begin(), sorted.end());

  for (std::size_t batch_pos = 0; batch_pos < sorted.size(); ++batch_pos) {
    auto c = sorted[batch_pos];
    if (c < column_ids_count) { map[c] = static_cast<duckdb::idx_t>(batch_pos); }
  }
  return map;
}

/// Narrowing helper: validates that a value fits in uint16_t before casting.
/// We hold seqnum / col_ids_position as uint16_t throughout the TAE path; this
/// matches the existing behavior in @c compute_task, but the prior code cast
/// silently. Centralizing here lets us fail loudly on pathological schemas.
std::uint16_t checked_uint16(std::size_t value, const char* what)
{
  if (value > std::numeric_limits<std::uint16_t>::max()) {
    throw sirius::internal_exception(
      "[tae_scan_plan] {} value {} exceeds uint16 range", what, value);
  }
  return static_cast<std::uint16_t>(value);
}

bool is_supported_mo_type(tae::MOTypeOid oid)
{
  switch (oid) {
    case tae::MO_T_bool:
    case tae::MO_T_bit:
    case tae::MO_T_int8:
    case tae::MO_T_int16:
    case tae::MO_T_int32:
    case tae::MO_T_int64:
    case tae::MO_T_uint8:
    case tae::MO_T_uint16:
    case tae::MO_T_uint32:
    case tae::MO_T_uint64:
    case tae::MO_T_float32:
    case tae::MO_T_float64:
    case tae::MO_T_decimal64:
    case tae::MO_T_decimal128:
    case tae::MO_T_date:
    case tae::MO_T_datetime:
    case tae::MO_T_timestamp:
    case tae::MO_T_year:
    case tae::MO_T_char:
    case tae::MO_T_varchar:
    case tae::MO_T_json:
    case tae::MO_T_enum:
    case tae::MO_T_text:
    case tae::MO_T_datalink: return true;
    default: return false;
  }
}

void resolve_projected_column(tae_scan_plan::projected_column& pc,
                              const tae_scan_plan& plan,
                              const duckdb::vector<sirius::logical_type>& returned_types)
{
  if (pc.seqnum >= plan.all_col_mo_oids.size() || pc.seqnum >= returned_types.size()) {
    throw sirius::internal_exception(
      "[tae_scan_plan] projected seqnum {} is outside the bound schema", pc.seqnum);
  }
  pc.type_oid = static_cast<tae::MOTypeOid>(plan.all_col_mo_oids[pc.seqnum]);
  if (!is_supported_mo_type(pc.type_oid)) {
    throw sirius::internal_exception(
      "[tae_scan_plan] MatrixOne type OID {} is not supported by the GPU decoder",
      static_cast<std::uint32_t>(pc.type_oid));
  }

  // Pre-resolve decimal width/scale once. compute_task used to do this per
  // chunk; for fixed-width decimal columns with hundreds of chunks per task
  // that's hundreds of redundant lookups per task.
  if (returned_types[pc.seqnum].is_decimal()) {
    pc.width = static_cast<std::int32_t>(returned_types[pc.seqnum].decimal_precision());
    pc.scale = static_cast<std::int32_t>(returned_types[pc.seqnum].decimal_scale());
  }
}

}  // namespace

tae_scan_plan build_tae_scan_plan(const tae::TAEScanBindData& bind_data,
                                  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
                                  const duckdb::vector<duckdb::idx_t>& projection_ids,
                                  const duckdb::vector<sirius::logical_type>& returned_types,
                                  std::size_t output_types_size,
                                  duckdb::TableFilterSet* table_filters)
{
  tae_scan_plan plan;

  // Copy P-indexed schema metadata. These are small (a handful of strings +
  // bytes); copying is simpler and safer than holding refs into bind_data.
  plan.all_col_names   = bind_data.all_col_names;
  plan.all_col_mo_oids = bind_data.all_col_mo_oids;
  plan.sort_column_idx = bind_data.sort_column_idx;

  // Resolve projected columns in projection order. Mirrors the per-task loop
  // in compute_task today (lines 269-282 of tae_scan_task.cpp pre-refactor)
  // but performed once at planning time.
  //
  // Defensive: skip virtual columns silently when they appear in pure-filter
  // positions (parquet does the same), but throw if they appear in output
  // positions — the TAE decode path has no story for materializing them.
  std::unordered_set<std::size_t> seen_primary_indices;

  auto handle_position = [&](std::size_t column_ids_pos, bool is_output) {
    auto const primary_idx = column_ids.at(column_ids_pos).GetPrimaryIndex();
    if (duckdb::IsVirtualColumn(primary_idx)) {
      if (is_output) {
        throw sirius::internal_exception(
          "[tae_scan_plan] virtual column at output position (column_ids index {}) "
          "is not supported by the TAE scan path",
          column_ids_pos);
      }
      // Filter-only virtual column: silently skip — the filter builder also
      // skips it via its OPTIONAL_FILTER / IS_NOT_NULL handling.
      return;
    }
    if (!seen_primary_indices.insert(primary_idx).second) {
      throw sirius::internal_exception(
        "[tae_scan_plan] duplicate primary index {} in projection — TAE scan "
        "expects each column to be projected at most once",
        primary_idx);
    }

    tae_scan_plan::projected_column pc{};
    pc.seqnum           = checked_uint16(primary_idx, "seqnum");
    pc.col_ids_position = checked_uint16(column_ids_pos, "col_ids_position");
    resolve_projected_column(pc, plan, returned_types);
    plan.projected_columns.push_back(pc);
  };

  if (projection_ids.empty()) {
    for (std::size_t c = 0; c < column_ids.size(); ++c) {
      handle_position(c, /* is_output */ true);
    }
  } else {
    for (std::size_t i = 0; i < projection_ids.size(); ++i) {
      bool const is_output = i < output_types_size;
      handle_position(projection_ids[i], is_output);
    }
  }

  // Build the C → C-sorted-batch-position map before resolving the output
  // projection below. `post_filter_projection_ids` is expressed in the
  // converter's batch space, so it must never observe the initial empty map.
  plan.batch_column_map = build_batch_column_map_local(projection_ids, column_ids.size());

  // Map output C positions to the converter's emergent C-sorted decoded
  // columns[] positions. `projection_ids` indexes column_ids, whereas the
  // converter emits only materialized columns; using the raw C position here
  // drops sparse projections (for example {1, 3}) or selects the wrong data.
  if (!projection_ids.empty()) {
    auto const n_output = std::min(projection_ids.size(), output_types_size);
    plan.post_filter_projection_ids.reserve(n_output);
    for (std::size_t i = 0; i < n_output; ++i) {
      auto const c = projection_ids[i];
      if (c >= plan.batch_column_map.size() ||
          plan.batch_column_map[c] == static_cast<duckdb::idx_t>(-1)) {
        throw sirius::internal_exception(
          "[tae_scan_plan] output projection position {} is not materialized", c);
      }
      plan.post_filter_projection_ids.push_back(static_cast<std::size_t>(plan.batch_column_map[c]));
    }
  }

  // Filter-pushdown side: extract zone-map filters once, and dedup their
  // seqnums once. compute_task previously rebuilt the seqnum dedup on every
  // task call.
  if (table_filters) {
    for (auto& [col_idx, filter] : table_filters->filters) {
      if (col_idx >= column_ids.size()) continue;
      auto const seqnum_p = column_ids[col_idx].GetPrimaryIndex();
      if (duckdb::IsVirtualColumn(seqnum_p)) continue;
      auto const seqnum = checked_uint16(seqnum_p, "filter seqnum");
      auto const c      = checked_uint16(col_idx, "filter column_ids index");
      std::uint8_t const mo_oid =
        (seqnum < plan.all_col_mo_oids.size()) ? plan.all_col_mo_oids[seqnum] : 0;
      tae::ExtractFilter(*filter, c, seqnum, mo_oid, plan.pushed_filters);
    }
    if (!plan.pushed_filters.empty()) {
      SIRIUS_LOG_INFO("[tae_scan_plan] extracted {} zone-map filters", plan.pushed_filters.size());
    }
  }

  // Dedup the filter seqnums (used by the per-task per-block zone-map
  // pruning loop). std::set keeps insert order out, so we use an unordered
  // set + first-seen-order vector to mirror the prior compute_task ordering.
  {
    std::unordered_set<std::uint16_t> seen;
    plan.filter_seqnums.reserve(plan.pushed_filters.size());
    for (auto& pf : plan.pushed_filters) {
      if (seen.insert(pf.seqnum).second) { plan.filter_seqnums.push_back(pf.seqnum); }
    }
  }

  SIRIUS_LOG_DEBUG(
    "[tae_scan_plan] built plan: {} projected col(s), {} post-filter projection id(s), "
    "{} pushed filter(s), {} unique filter seqnum(s)",
    plan.projected_columns.size(),
    plan.post_filter_projection_ids.size(),
    plan.pushed_filters.size(),
    plan.filter_seqnums.size());

  return plan;
}

}  // namespace sirius::op::scan
