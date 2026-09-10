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

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>

namespace sirius::cuda::tae {

/**
 * @brief Decode a fixed-width TAE column from decompressed MO format to cuDF format.
 *
 * Caller must provide a pointer to the data section of the MO vector
 * (past IOEntryHeader + vector header). This kernel copies the data portion
 * directly (possibly with epoch adjustment for DATE/TIMESTAMP types).
 * The null bitmap is handled separately by invert_null_mask.
 *
 * @param d_src        Pointer to the data section of the MO vector
 * @param d_dst        Output buffer for cuDF column values
 * @param n_rows       Number of rows
 * @param elem_size    Element size in bytes (1,2,4,8,16)
 * @param epoch_adjust Value to subtract for epoch conversion (0 for non-temporal types,
 *                     MO_UNIX_EPOCH_DAYS for DATE, MO_UNIX_EPOCH_USEC for TIMESTAMP)
 * @param stream       CUDA stream
 */
void decode_fixed_width(const uint8_t* d_src,
                        uint8_t* d_dst,
                        uint32_t n_rows,
                        uint32_t elem_size,
                        int64_t epoch_adjust,
                        rmm::cuda_stream_view stream);

/**
 * @brief Compute cuDF string offsets from MO varlena column (pass 1 of 2).
 *
 * Reads each Varlena struct to compute per-row string lengths, then runs
 * CUB ExclusiveSum to produce offsets[0..n_rows]. offsets[n_rows] equals the
 * total character bytes (usable for chars buffer allocation).
 *
 * Call with d_temp_storage=nullptr first to query CUB temp buffer size.
 *
 * @param d_varlena_base  Pointer to the varlena struct array (data section)
 * @param d_area_base     Pointer to the area section (for big string reads)
 * @param d_offsets       Output: int32[n_rows+1] offsets array
 * @param d_temp_storage  CUB temporary storage (nullptr for size query)
 * @param temp_bytes      [in/out] Size of temp storage
 * @param n_rows          Number of rows
 * @param stream          CUDA stream
 */
void decode_varchar_offsets(const uint8_t* d_varlena_base,
                            const uint8_t* d_area_base,
                            int32_t* d_offsets,
                            void* d_temp_storage,
                            std::size_t& temp_bytes,
                            uint32_t n_rows,
                            bool is_constant,
                            rmm::cuda_stream_view stream);

/**
 * @brief Scatter string data using precomputed offsets (pass 2 of 2).
 *
 * Reads each Varlena struct and copies its string data to chars[offsets[i]].
 * Must be called after decode_varchar_offsets has produced the offsets array.
 *
 * @param d_varlena_base  Pointer to the varlena struct array (data section)
 * @param d_area_base     Pointer to the area section (for big string reads)
 * @param d_offsets       Input: precomputed int32[n_rows+1] offsets
 * @param d_chars         Output: character data buffer
 * @param n_rows          Number of rows
 * @param stream          CUDA stream
 */
void decode_varchar_scatter(const uint8_t* d_varlena_base,
                            const uint8_t* d_area_base,
                            const int32_t* d_offsets,
                            uint8_t* d_chars,
                            uint32_t n_rows,
                            bool is_constant,
                            rmm::cuda_stream_view stream);

/**
 * @brief Decode a varlena TAE column to cuDF string column format (convenience).
 *
 * Equivalent to calling decode_varchar_offsets + decode_varchar_scatter.
 *
 * @param d_varlena_base  Pointer to the varlena struct array (data section)
 * @param d_area_base     Pointer to the area section (for big string reads)
 * @param d_offsets       Output: int32[n_rows+1] offsets array
 * @param d_chars         Output: character data buffer
 * @param d_temp_storage  CUB temporary storage (nullptr on first call for size query)
 * @param temp_bytes      [in/out] Size of temp storage
 * @param n_rows          Number of rows
 * @param stream          CUDA stream
 */
void decode_varchar(const uint8_t* d_varlena_base,
                    const uint8_t* d_area_base,
                    int32_t* d_offsets,
                    uint8_t* d_chars,
                    void* d_temp_storage,
                    std::size_t& temp_bytes,
                    uint32_t n_rows,
                    bool is_constant,
                    rmm::cuda_stream_view stream);

/**
 * @brief Adjust offsets array by adding a constant base value.
 *
 * Used to make multi-block offsets globally monotonic for cuDF.
 * Each block's exclusive prefix-sum starts from 0; this adds the cumulative
 * character count from all previous blocks so cuDF sees strictly increasing offsets.
 *
 * @param d_offsets  Offsets array to adjust in-place
 * @param base       Constant to add to each element
 * @param count      Number of elements (typically n_rows + 1)
 * @param stream     CUDA stream
 */
void adjust_offsets(int32_t* d_offsets, int32_t base, uint32_t count, rmm::cuda_stream_view stream);

/**
 * @brief Invert a MO null bitmap to cuDF validity bitmask.
 *
 * MO: bit=1 means NULL.  cuDF: bit=1 means VALID.
 * Simply bitwise-NOT each 32-bit word.
 *
 * @param d_src       Pointer to the MO null bitmap words (in the nsp section,
 *                    after the 24-byte nsp header: count+len+dataSize)
 * @param d_dst       Output cuDF validity bitmask
 * @param n_rows      Number of rows
 * @param stream      CUDA stream
 */
void invert_null_mask(const uint8_t* d_src,
                      uint32_t* d_dst,
                      uint32_t n_rows,
                      rmm::cuda_stream_view stream);

// ---------------------------------------------------------------------------
// Batched decode descriptors — one per block (TAE object has many blocks).
// Used by 2D-grid batched kernels (blockIdx.y = descriptor index) to replace
// per-block kernel launches with a single launch per column.
// ---------------------------------------------------------------------------

/// Block descriptor for batched fixed-width decode.
struct BatchedFixedDesc {
  const uint8_t* src;   ///< Device pointer to data section (past VEC_HEADER).
  uint32_t n_rows;      ///< Number of rows in this block.
  uint32_t row_offset;  ///< Cumulative row offset in the output column.
  bool is_constant;     ///< Whether the serialized vector contains one repeated value.
  bool has_data;        ///< False for a constant-null vector with an empty data section.
};

/// Block descriptor for batched null mask inversion.
struct BatchedNullMaskDesc {
  const uint8_t* src;     ///< Device pointer to MO null bitmap.
  uint32_t n_rows;        ///< Number of rows in this block.
  uint32_t row_offset;    ///< Bit offset in the output cuDF validity mask.
  uint32_t bitmap_bytes;  ///< Serialized bytes available at @c src.
};

/**
 * @brief Batched fixed-width decode across all blocks of a column.
 *
 * Replaces per-block cudaMemcpyAsync / decode_fixed_width calls with a single
 * 2D-grid kernel launch (blockIdx.y = block descriptor index).
 *
 * @param d_descs         Device array of BatchedFixedDesc (one per block)
 * @param n_descs         Number of descriptors (blocks)
 * @param d_dst           Output buffer for the entire column
 * @param elem_size       Element size in bytes (1,2,4,8,16)
 * @param epoch_adjust    Epoch subtraction (0 for non-temporal types)
 * @param max_block_rows  Maximum n_rows across all descriptors (for grid sizing)
 * @param stream          CUDA stream
 */
void batched_decode_fixed_width(const BatchedFixedDesc* d_descs,
                                uint32_t n_descs,
                                uint8_t* d_dst,
                                uint32_t elem_size,
                                int64_t epoch_adjust,
                                uint32_t max_block_rows,
                                rmm::cuda_stream_view stream);

/**
 * @brief Batched null mask inversion across multiple blocks.
 *
 * Replaces per-block invert_null_mask calls with a single 2D-grid kernel.
 *
 * @param d_descs     Device array of BatchedNullMaskDesc
 * @param n_descs     Number of descriptors (blocks with nulls)
 * @param d_validity  Output cuDF validity bitmask (pre-initialized to ALL_VALID)
 * @param stream      CUDA stream
 */
void batched_invert_null_mask(const BatchedNullMaskDesc* d_descs,
                              uint32_t n_descs,
                              uint32_t* d_validity,
                              rmm::cuda_stream_view stream);

}  // namespace sirius::cuda::tae
