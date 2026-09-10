/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <data/host_tae_representation.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/tae_scan_plan.hpp>

#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sirius::op {
class sirius_dynamic_filter_set;
}

namespace sirius::op::scan {

class tae_ingestible_table_info final : public ingestible_table_info {
 public:
  duckdb::unique_ptr<tae::TAEScanBindData> bind_data;
  duckdb::vector<sirius::logical_type> returned_types;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<sirius::logical_type> output_types;
  duckdb::vector<std::string> names;
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> sirius_dynamic_filters;
  duckdb::ClientContext* context = nullptr;

  [[nodiscard]] std::span<std::string const> column_names() const override { return names; }
  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    return bind_data ? std::span<std::string const>(object_paths_) : std::span<std::string const>{};
  }

  void refresh_object_paths();

 private:
  std::vector<std::string> object_paths_;
};

/// A scanner-produced compressed object ready for GPU LZ4 decode.
class tae_scan_info final : public scan_info {
 public:
  std::shared_ptr<sirius::pinned_host_buffer> host_data;
  std::vector<sirius::host_tae_representation::column_chunk_info> chunks;
  std::size_t rows               = 0;
  std::size_t compressed_bytes   = 0;
  std::size_t uncompressed_bytes = 0;

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override { return uncompressed_bytes; }
  [[nodiscard]] std::size_t estimated_working_set_bytes() const noexcept override
  {
    return uncompressed_bytes;
  }
};

/// Native TAE source for the current GPU_SCAN + scan-manager architecture.
class tae_gpu_ingestible final : public gpu_ingestible {
 public:
  explicit tae_gpu_ingestible(std::unique_ptr<tae_ingestible_table_info> info);
  ~tae_gpu_ingestible() override;

  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;
  [[nodiscard]] bool has_processed_all_metadata() const override;
  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override;

  filtered_table materialize_metadata_to_table(
    scan_info const& info,
    const cucascade::memory::memory_space& mem_space,
    rmm::cuda_stream_view stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    const cucascade::memory::memory_space& mem_space,
    rmm::cuda_stream_view stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
    std::unique_ptr<cudf::column>* survivors,
    std::span<std::size_t const> elided) override;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept override { return *_info; }
  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override;
  [[nodiscard]] bool has_row_filter() const noexcept override
  {
    return _filter_expression != nullptr;
  }
  [[nodiscard]] bool can_report_survivors() const noexcept override { return true; }

 private:
  [[nodiscard]] std::unique_ptr<tae_scan_info> load_object(
    std::size_t object_index, std::shared_ptr<io::sirius_ioctx> const& io_ctx) const;
  [[nodiscard]] std::unique_ptr<cudf::table> make_empty_table(
    const cucascade::memory::memory_space& mem_space, rmm::cuda_stream_view stream) const;

  std::unique_ptr<tae_ingestible_table_info> _info;
  tae_scan_plan _plan;
  std::shared_ptr<duckdb::Expression> _filter_expression;
  std::atomic<std::size_t> _next_object{0};
};

std::shared_ptr<tae_gpu_ingestible> make_ingestible(
  std::unique_ptr<tae_ingestible_table_info> info);

}  // namespace sirius::op::scan
