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
//
// TAE metadata parser — reimplemented from tae-scanner for Sirius GPU pipeline.
// Pure C++ with no DuckDB dependencies.

#include <tae/tae_format.hpp>

#include <cstring>
#include <stdexcept>
#include <string>

namespace tae {

void ParseMetadata(const uint8_t* buf, uint32_t len, ObjectMeta& meta)
{
  if (len < META_V3_HEADER_LEN) { throw std::runtime_error("metadata too small for v3 header"); }

  // Read v3 header
  uint16_t data_meta_count;
  uint32_t data_meta_offset;
  memcpy(&data_meta_count, buf + MV3_DATA_COUNT_OFF, 2);
  memcpy(&data_meta_offset, buf + MV3_DATA_OFFSET_OFF, 4);

  if (data_meta_count == 0) {
    meta.block_count = 0;
    meta.blocks.clear();
    return;
  }

  if (data_meta_offset >= len) { throw std::runtime_error("data meta offset out of range"); }

  const uint8_t* dm = buf + data_meta_offset;
  uint32_t dm_len   = len - data_meta_offset;

  // DataMeta = objectDataMetaV1: starts with object-level BlockHeader (179B)
  if (dm_len < BLOCK_HEADER_SIZE) {
    throw std::runtime_error("data meta too small for block header");
  }

  // Object-level BlockHeader: extract metaColumnCount for the object
  uint16_t obj_meta_col_count;
  memcpy(&obj_meta_col_count, dm + BH_META_COL_CNT_OFF, 2);

  // Object-level data length = headerLen + metaColCount * colMetaLen
  if (obj_meta_col_count > (dm_len - BLOCK_HEADER_SIZE) / COL_META_LEN) {
    throw std::runtime_error("object column metadata exceeds buffer");
  }
  uint32_t obj_data_len = BLOCK_HEADER_SIZE + uint32_t(obj_meta_col_count) * COL_META_LEN;
  if (obj_data_len > dm_len) { throw std::runtime_error("object data meta exceeds buffer"); }

  // BlockIndex follows the object-level data
  const uint8_t* bi     = dm + obj_data_len;
  uint32_t bi_remaining = dm_len - obj_data_len;
  if (bi_remaining < BI_BLOCK_COUNT_LEN) { throw std::runtime_error("no room for block index"); }

  uint32_t block_count;
  memcpy(&block_count, bi, BI_BLOCK_COUNT_LEN);

  if (block_count > (bi_remaining - BI_BLOCK_COUNT_LEN) / BI_POS_LEN) {
    throw std::runtime_error("block index exceeds buffer");
  }
  meta.block_count = block_count;
  meta.blocks.resize(block_count);

  // Parse each block using the block index
  for (uint32_t b = 0; b < block_count; b++) {
    uint32_t pos_off = BI_BLOCK_COUNT_LEN + b * BI_POS_LEN;
    uint32_t blk_offset, blk_length;
    memcpy(&blk_offset, bi + pos_off, 4);
    memcpy(&blk_length, bi + pos_off + 4, 4);

    // blk_offset is relative to the start of DataMeta (dm)
    if (blk_offset > dm_len || blk_length > dm_len - blk_offset) {
      throw std::runtime_error("block " + std::to_string(b) + " out of range");
    }

    const uint8_t* blk = dm + blk_offset;
    if (blk_length < BLOCK_HEADER_SIZE) {
      throw std::runtime_error("block " + std::to_string(b) + " header too small");
    }

    auto& info = meta.blocks[b];
    memcpy(&info.rows, blk + BH_ROWS_OFF, 4);
    memcpy(&info.col_count, blk + BH_COL_COUNT_OFF, 2);
    memcpy(&info.meta_col_count, blk + BH_META_COL_CNT_OFF, 2);
    memcpy(&info.max_seqnum, blk + BH_MAX_SEQ_OFF, 2);

    // Parse column metadata (indexed by seqnum, not ordinal position)
    uint32_t num_cols = info.meta_col_count;
    if (num_cols > (blk_length - BLOCK_HEADER_SIZE) / COL_META_LEN) {
      throw std::runtime_error("block " + std::to_string(b) + " column metadata exceeds buffer");
    }

    info.columns.resize(num_cols);
    for (uint32_t c = 0; c < num_cols; c++) {
      const uint8_t* cm = blk + BLOCK_HEADER_SIZE + c * COL_META_LEN;
      auto& col         = info.columns[c];
      col.data_type     = cm[CM_DATA_TYPE_OFF];
      memcpy(&col.idx, cm + CM_IDX_OFF, 2);
      memcpy(&col.ndv, cm + CM_NDV_OFF, 4);
      memcpy(&col.null_cnt, cm + CM_NULL_CNT_OFF, 4);
      memcpy(&col.location, cm + CM_LOCATION_OFF, sizeof(Extent));
      memcpy(&col.checksum, cm + CM_CHECKSUM_OFF, 4);
      memcpy(col.zone_map, cm + CM_ZONEMAP_OFF, ZM_SIZE);
    }
  }
}

}  // namespace tae
