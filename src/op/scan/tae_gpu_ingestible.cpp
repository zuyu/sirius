/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <data/sirius_converter_registry.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <lz4.h>
#include <op/scan/owning_table_view.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/tae_gpu_ingestible.hpp>
#include <tae/tae_format.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {
namespace {

class passthrough_coalescer final : public batch_coalescer {
 public:
  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (info) { out.push_back(std::move(info)); }
    return out;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override { return {}; }
};

std::vector<std::uint8_t> decompress_metadata_lz4(const std::uint8_t* src,
                                                  std::uint32_t src_len,
                                                  std::uint32_t origin_size)
{
  if (src_len > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      origin_size > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("TAE metadata is too large for the LZ4 decoder");
  }
  std::vector<std::uint8_t> dst(origin_size);
  auto const decoded = LZ4_decompress_safe(reinterpret_cast<char const*>(src),
                                           reinterpret_cast<char*>(dst.data()),
                                           static_cast<int>(src_len),
                                           static_cast<int>(origin_size));
  if (decoded < 0 || static_cast<std::uint32_t>(decoded) != origin_size) {
    throw std::runtime_error("TAE metadata LZ4 decompression failed");
  }
  return dst;
}

void host_read_exact(io::sirius_datasource& source,
                     std::uint64_t offset,
                     std::size_t length,
                     std::uint8_t* destination)
{
  auto const read = source.host_read(offset, length, destination);
  if (read != length) { throw std::runtime_error("short TAE host read"); }
}

bool has_crc_wrapper(io::sirius_datasource& source)
{
  if (source.size() < 12) { throw std::runtime_error("TAE object is shorter than its header"); }
  std::uint8_t probe[12];
  host_read_exact(source, 0, sizeof(probe), probe);

  std::uint64_t magic_at_zero{};
  std::uint64_t magic_at_four{};
  std::memcpy(&magic_at_zero, probe, sizeof(magic_at_zero));
  std::memcpy(&magic_at_four, probe + 4, sizeof(magic_at_four));
  if (magic_at_zero == tae::OBJECT_MAGIC) { return false; }
  if (magic_at_four == tae::OBJECT_MAGIC) { return true; }
  throw std::runtime_error("invalid TAE magic: neither raw nor CRC-wrapped");
}

std::vector<std::uint8_t> read_logical_bytes(io::sirius_datasource& source,
                                             std::uint64_t logical_offset,
                                             std::uint64_t length,
                                             bool crc_wrapped)
{
  if (length == 0) { return {}; }
  if (logical_offset > std::numeric_limits<std::uint64_t>::max() - length) {
    throw std::runtime_error("TAE logical read range overflows");
  }
  if (!crc_wrapped) {
    if (logical_offset > source.size() || length > source.size() - logical_offset) {
      throw std::runtime_error("TAE logical read exceeds object size");
    }
    std::vector<std::uint8_t> result(length);
    host_read_exact(source, logical_offset, length, result.data());
    return result;
  }

  auto const first_block     = logical_offset / tae::CRC_CONTENT_SIZE;
  auto const last_block      = (logical_offset + length - 1) / tae::CRC_CONTENT_SIZE;
  auto const physical_offset = first_block * tae::CRC_BLOCK_SIZE;
  auto physical_end          = (last_block + 1) * tae::CRC_BLOCK_SIZE;
  physical_end               = std::min<std::uint64_t>(physical_end, source.size());

  std::vector<std::uint8_t> raw(physical_end - physical_offset);
  host_read_exact(source, physical_offset, raw.size(), raw.data());

  std::vector<std::uint8_t> stripped;
  stripped.reserve((raw.size() / tae::CRC_BLOCK_SIZE + 1) * tae::CRC_CONTENT_SIZE);
  for (std::size_t offset = 0; offset < raw.size(); offset += tae::CRC_BLOCK_SIZE) {
    auto const remaining = raw.size() - offset;
    if (remaining <= tae::CRC_SIZE) { break; }
    auto const content = std::min<std::size_t>(tae::CRC_CONTENT_SIZE, remaining - tae::CRC_SIZE);
    stripped.insert(stripped.end(),
                    raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE),
                    raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE + content));
  }

  auto const content_start = first_block * tae::CRC_CONTENT_SIZE;
  auto const local_offset  = logical_offset - content_start;
  if (local_offset + length > stripped.size()) {
    throw std::runtime_error("CRC TAE read exceeds stripped content");
  }
  return {stripped.begin() + static_cast<std::ptrdiff_t>(local_offset),
          stripped.begin() + static_cast<std::ptrdiff_t>(local_offset + length)};
}

struct planned_read {
  std::uint64_t offset{};
  std::uint32_t compressed_length{};
  std::uint32_t origin_size{};
  std::uint8_t algorithm{};
  std::uint16_t column_position{};
  std::uint32_t row_count{};
  std::uint32_t null_count{};
  tae::MOTypeOid type_oid{tae::MO_T_any};
  std::int32_t width{};
  std::int32_t scale{};
};

void read_chunks_coalesced(io::sirius_datasource& source,
                           bool crc_wrapped,
                           std::vector<planned_read>& reads,
                           pinned_host_buffer& destination)
{
  std::sort(reads.begin(), reads.end(), [](planned_read const& left, planned_read const& right) {
    return left.offset < right.offset;
  });

  for (auto const& read : reads) {
    if (read.compressed_length == 0 ||
        read.offset > std::numeric_limits<std::uint64_t>::max() - read.compressed_length) {
      throw std::runtime_error("invalid TAE column extent");
    }
    if (!crc_wrapped &&
        (read.offset > source.size() || read.compressed_length > source.size() - read.offset)) {
      throw std::runtime_error("TAE column extent exceeds object size");
    }
  }

  constexpr std::size_t max_group_reads = 32;
  std::size_t group_begin               = 0;
  std::size_t destination_offset        = 0;
  while (group_begin < reads.size()) {
    std::size_t group_end = group_begin + 1;
    while (group_end < reads.size() && group_end - group_begin < max_group_reads &&
           reads[group_end].offset ==
             reads[group_end - 1].offset + reads[group_end - 1].compressed_length) {
      ++group_end;
    }

    auto const& first      = reads[group_begin];
    auto const& last       = reads[group_end - 1];
    auto const logical_end = last.offset + last.compressed_length;
    if (!crc_wrapped) {
      host_read_exact(
        source, first.offset, logical_end - first.offset, destination.data() + destination_offset);
    } else {
      auto const first_block  = first.offset / tae::CRC_CONTENT_SIZE;
      auto const last_block   = (logical_end - 1) / tae::CRC_CONTENT_SIZE;
      auto const physical_beg = first_block * tae::CRC_BLOCK_SIZE;
      auto const physical_end =
        std::min<std::uint64_t>((last_block + 1) * tae::CRC_BLOCK_SIZE, source.size());
      std::vector<std::uint8_t> raw(physical_end - physical_beg);
      host_read_exact(source, physical_beg, raw.size(), raw.data());

      std::vector<std::uint8_t> stripped;
      stripped.reserve((raw.size() / tae::CRC_BLOCK_SIZE + 1) * tae::CRC_CONTENT_SIZE);
      for (std::size_t offset = 0; offset < raw.size(); offset += tae::CRC_BLOCK_SIZE) {
        auto const remaining = raw.size() - offset;
        if (remaining <= tae::CRC_SIZE) { break; }
        auto const content =
          std::min<std::size_t>(tae::CRC_CONTENT_SIZE, remaining - tae::CRC_SIZE);
        stripped.insert(
          stripped.end(),
          raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE),
          raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE + content));
      }

      auto const content_begin = first_block * tae::CRC_CONTENT_SIZE;
      std::size_t write_offset = destination_offset;
      for (std::size_t read_idx = group_begin; read_idx < group_end; ++read_idx) {
        auto const& read        = reads[read_idx];
        auto const local_offset = read.offset - content_begin;
        if (local_offset + read.compressed_length > stripped.size()) {
          throw std::runtime_error("CRC TAE coalesced read exceeds stripped content");
        }
        std::memcpy(destination.data() + write_offset,
                    stripped.data() + local_offset,
                    read.compressed_length);
        write_offset += read.compressed_length;
      }
    }

    for (std::size_t read_idx = group_begin; read_idx < group_end; ++read_idx) {
      destination_offset += reads[read_idx].compressed_length;
    }
    group_begin = group_end;
  }
}

std::vector<std::optional<std::size_t>> to_batch_positions(
  std::vector<duckdb::idx_t> const& positions)
{
  std::vector<std::optional<std::size_t>> out;
  out.reserve(positions.size());
  for (auto const position : positions) {
    if (position == static_cast<duckdb::idx_t>(-1)) {
      out.emplace_back(std::nullopt);
    } else {
      out.emplace_back(static_cast<std::size_t>(position));
    }
  }
  return out;
}

}  // namespace

void tae_ingestible_table_info::refresh_object_paths()
{
  object_paths_.clear();
  if (!bind_data) { return; }
  object_paths_.reserve(bind_data->objects.size());
  for (auto const& object : bind_data->objects) {
    object_paths_.push_back(bind_data->data_dir + "/" + object.file_path);
  }
}

tae_gpu_ingestible::tae_gpu_ingestible(std::unique_ptr<tae_ingestible_table_info> info)
  : _info(std::move(info))
{
  if (!_info || !_info->bind_data || _info->context == nullptr) {
    throw std::invalid_argument("TAE ingestible requires bind data and a client context");
  }
  _info->refresh_object_paths();
  _plan = build_tae_scan_plan(*_info->bind_data,
                              _info->column_ids,
                              _info->projection_ids,
                              _info->returned_types,
                              _info->output_types.size(),
                              _info->table_filters.get());

  if (_info->table_filters && !_info->table_filters->filters.empty()) {
    _filter_expression =
      op::convert_table_filters_to_expression(*_info->table_filters,
                                              _info->column_ids,
                                              _info->returned_types,
                                              to_batch_positions(_plan.batch_column_map));
  }
}

tae_gpu_ingestible::~tae_gpu_ingestible() = default;

std::unique_ptr<batch_coalescer> tae_gpu_ingestible::create_batch_coalescer() const
{
  return std::make_unique<passthrough_coalescer>();
}

bool tae_gpu_ingestible::has_processed_all_metadata() const
{
  return _next_object.load(std::memory_order_relaxed) >= _info->bind_data->objects.size();
}

gpu_ingestible::metadata_scan_task_t tae_gpu_ingestible::next_split_provider(
  io::ioctx_resolver resolve)
{
  if (!resolve) { throw std::runtime_error("TAE ingestible has no scan-manager I/O resolver"); }
  auto const object_index = _next_object.fetch_add(1, std::memory_order_relaxed);
  if (object_index >= _info->bind_data->objects.size()) { return nullptr; }
  auto const file_path =
    _info->bind_data->data_dir + "/" + _info->bind_data->objects[object_index].file_path;
  auto io_ctx = resolve(file_path);
  return
    [this, object_index, io_ctx = std::move(io_ctx)] { return load_object(object_index, io_ctx); };
}

std::unique_ptr<tae_scan_info> tae_gpu_ingestible::load_object(
  std::size_t object_index, std::shared_ptr<io::sirius_ioctx> const& io_ctx) const
{
  auto const& object   = _info->bind_data->objects.at(object_index);
  auto const file_path = _info->bind_data->data_dir + "/" + object.file_path;
  if (!io_ctx) { throw std::runtime_error("TAE ingestible received a null I/O context"); }
  if (!object.sort_key_zm.empty() && !_plan.pushed_filters.empty() && _plan.sort_column_idx >= 0 &&
      !tae::ZoneMapPassesFilters(_plan.pushed_filters,
                                 object.sort_key_zm.data(),
                                 static_cast<std::uint16_t>(_plan.sort_column_idx))) {
    return std::make_unique<tae_scan_info>();
  }

  auto source            = io_ctx->open_datasource(file_path, object.size_bytes);
  auto const crc_wrapped = has_crc_wrapper(*source);

  auto header = read_logical_bytes(*source, 0, tae::HEADER_SIZE, crc_wrapped);
  std::uint64_t magic{};
  std::memcpy(&magic, header.data(), sizeof(magic));
  if (magic != tae::OBJECT_MAGIC) {
    throw std::runtime_error("invalid TAE magic after removing CRC wrappers");
  }
  tae::Extent metadata_extent{};
  std::memcpy(
    &metadata_extent, header.data() + tae::HEADER_META_EXTENT_OFF, sizeof(metadata_extent));
  if (metadata_extent.alg > 1 || metadata_extent.length == 0 ||
      (metadata_extent.is_compressed() && metadata_extent.origin_size == 0)) {
    throw std::runtime_error("invalid TAE metadata extent");
  }
  auto metadata =
    read_logical_bytes(*source, metadata_extent.offset, metadata_extent.length, crc_wrapped);
  if (metadata_extent.is_compressed()) {
    metadata =
      decompress_metadata_lz4(metadata.data(), metadata_extent.length, metadata_extent.origin_size);
  }
  if (metadata.size() <= tae::IO_ENTRY_HEADER_LEN) {
    throw std::runtime_error("TAE object metadata is shorter than its IO entry header");
  }
  tae::IOEntryHeader metadata_header{};
  std::memcpy(&metadata_header, metadata.data(), sizeof(metadata_header));
  if (metadata_header.type != tae::IOET_ObjMeta) {
    throw std::runtime_error("TAE metadata extent has an unexpected IO entry type");
  }

  tae::ObjectMeta object_metadata;
  tae::ParseMetadata(metadata.data() + tae::IO_ENTRY_HEADER_LEN,
                     static_cast<std::uint32_t>(metadata.size() - tae::IO_ENTRY_HEADER_LEN),
                     object_metadata);

  std::vector<planned_read> reads;
  std::size_t selected_rows = 0;
  for (std::uint32_t block_index = 0; block_index < object_metadata.block_count; ++block_index) {
    auto const& block = object_metadata.blocks[block_index];
    bool passes       = true;
    for (auto const sequence : _plan.filter_seqnums) {
      if (sequence < block.columns.size() &&
          !tae::ZoneMapPassesFilters(
            _plan.pushed_filters, block.columns[sequence].zone_map, sequence)) {
        passes = false;
        break;
      }
    }
    if (!passes) { continue; }

    selected_rows += block.rows;
    for (auto const& projected : _plan.projected_columns) {
      if (projected.seqnum >= block.columns.size()) {
        throw std::runtime_error("TAE block is missing a projected column");
      }
      auto const& column = block.columns[projected.seqnum];
      if (column.data_type != static_cast<std::uint8_t>(projected.type_oid)) {
        throw std::runtime_error("TAE block column type does not match the manifest schema");
      }
      auto const& extent = column.location;
      if (extent.alg > 1 || extent.length == 0 ||
          (extent.is_compressed() && extent.origin_size == 0)) {
        throw std::runtime_error("invalid TAE column extent");
      }
      reads.push_back({extent.offset,
                       extent.length,
                       extent.origin_size,
                       extent.alg,
                       projected.col_ids_position,
                       block.rows,
                       column.null_cnt,
                       projected.type_oid,
                       projected.width,
                       projected.scale});
    }
  }

  auto result  = std::make_unique<tae_scan_info>();
  result->rows = selected_rows;
  if (reads.empty() || selected_rows == 0) { return result; }

  for (auto const& read : reads) {
    result->compressed_bytes += read.compressed_length;
    result->uncompressed_bytes += read.origin_size;
  }
  auto host               = std::make_shared<pinned_host_buffer>(result->compressed_bytes);
  std::size_t host_offset = 0;
  read_chunks_coalesced(*source, crc_wrapped, reads, *host);
  result->chunks.reserve(reads.size());
  for (auto const& read : reads) {
    host_tae_representation::column_chunk_info chunk;
    chunk.column_idx    = read.column_position;
    chunk.type_oid      = read.type_oid;
    chunk.width         = read.width;
    chunk.scale         = read.scale;
    chunk.extent        = tae::Extent{read.algorithm,
                               static_cast<std::uint32_t>(read.offset),
                               read.compressed_length,
                               read.origin_size};
    chunk.null_cnt      = read.null_count;
    chunk.row_count     = read.row_count;
    chunk.pinned_offset = host_offset;
    chunk.pinned_length = read.compressed_length;
    result->chunks.push_back(chunk);
    host_offset += read.compressed_length;
  }
  result->host_data = std::move(host);
  return result;
}

std::unique_ptr<cudf::table> tae_gpu_ingestible::make_empty_table(
  const cucascade::memory::memory_space&, rmm::cuda_stream_view) const
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(_info->output_types.size());
  for (auto const& logical : _info->output_types) {
    auto const native = sirius::try_get_cudf_type(logical);
    if (!native) { throw std::runtime_error("TAE empty result has an unsupported output type"); }
    columns.push_back(cudf::make_empty_column(*native));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

filtered_table tae_gpu_ingestible::materialize_metadata_to_table(
  scan_info const& generic_info,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream,
  bool,
  std::shared_ptr<const sirius::like_multiliteral_cache>)
{
  auto const& info = dynamic_cast<tae_scan_info const&>(generic_info);
  if (!info.host_data || info.chunks.empty()) {
    return {.table = owning_table_view{make_empty_table(mem_space, stream)},
            .state = filter_state::ROW_FILTERED_AND_PROJECTED};
  }

  host_tae_representation host{const_cast<cucascade::memory::memory_space*>(&mem_space),
                               info.host_data,
                               info.chunks,
                               info.rows,
                               info.compressed_bytes,
                               info.uncompressed_bytes};
  auto gpu = sirius::converter_registry::get().convert<cucascade::gpu_table_representation>(
    host, &mem_space, stream);
  return {.table = owning_table_view{gpu->release_table(stream)},
          .state = filter_state::UNFILTERED};
}

std::unique_ptr<cudf::table> tae_gpu_ingestible::post_filter_and_project(
  filtered_table&& input,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream,
  bool like_swar_fastpath,
  std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
  std::unique_ptr<cudf::column>* survivors,
  std::span<std::size_t const> elided)
{
  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto output_positions = _plan.post_filter_projection_ids;
  if (output_positions.empty() && !_info->output_types.empty()) {
    output_positions.resize(_info->output_types.size());
    std::iota(output_positions.begin(), output_positions.end(), std::size_t{0});
  }

  owning_table_view final_table;
  if (_filter_expression) {
    auto filter = sirius::ast::from_duckdb(*_filter_expression);
    if (!filter) { throw std::runtime_error("TAE scan could not lower its table filter"); }
    sirius::expression_evaluator evaluator(filter.get(),
                                           mr_ref,
                                           stream,
                                           sirius::strategy_from_config(),
                                           sirius::expression_evaluator::default_min_ast_size,
                                           like_swar_fastpath,
                                           std::move(like_cache));
    std::vector<cudf::size_type> cudf_output_positions;
    cudf_output_positions.reserve(output_positions.size());
    for (auto const position : output_positions) {
      cudf_output_positions.push_back(static_cast<cudf::size_type>(position));
    }
    if (survivors != nullptr && !output_positions.empty()) {
      final_table = owning_table_view{
        evaluator.select_with_survivors(input.table.view(), cudf_output_positions, *survivors)};
    } else if (output_positions.empty()) {
      final_table = owning_table_view{evaluator.select(input.table.view())};
    } else {
      final_table = owning_table_view{evaluator.select(input.table.view(), cudf_output_positions)};
    }
    // The select only enqueued its reads. Preserve the source table's reader
    // event before replacing the owner.
    input.table.record_reader_event(stream);
  } else {
    final_table = std::move(input.table);
  }

  if (!_filter_expression && !output_positions.empty()) {
    final_table.select_columns(output_positions);
  }
  if (!elided.empty() && elided.size() < final_table.view().num_columns()) {
    std::vector<std::size_t> kept;
    kept.reserve(final_table.view().num_columns() - elided.size());
    for (std::size_t position = 0;
         position < static_cast<std::size_t>(final_table.view().num_columns());
         ++position) {
      if (std::find(elided.begin(), elided.end(), position) == elided.end()) {
        kept.push_back(position);
      }
    }
    if (!kept.empty()) { final_table.select_columns(kept); }
  }
  return final_table.release(stream, mr_ref);
}

std::vector<std::size_t> tae_gpu_ingestible::materialized_column_order() const
{
  std::vector<std::size_t> order;
  order.reserve(_plan.projected_columns.size());
  for (auto const& column : _plan.projected_columns) {
    order.push_back(column.seqnum);
  }
  return order;
}

std::shared_ptr<tae_gpu_ingestible> make_ingestible(std::unique_ptr<tae_ingestible_table_info> info)
{
  return std::make_shared<tae_gpu_ingestible>(std::move(info));
}

}  // namespace sirius::op::scan
