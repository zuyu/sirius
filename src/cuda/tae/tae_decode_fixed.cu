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

#include "cudf/cudf_compat.hpp"

#include <cudf/utilities/error.hpp>

#include <cuda/tae/tae_decode_kernels.hpp>
#include <cuda_runtime.h>

#include <cstdint>

namespace sirius::cuda::tae {

namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

// Fused copy + epoch adjustment kernel for 4-byte types (DATE: int32)
// Source may be unaligned (MO vector header is 29 bytes), destination is aligned.
__global__ void copy_and_adjust_i32_kernel(const uint8_t* __restrict__ src,
                                           int32_t* __restrict__ dst,
                                           uint32_t n_rows,
                                           int32_t adjust)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    int32_t val;
    memcpy(&val, src + i * sizeof(int32_t), sizeof(int32_t));
    dst[i] = val - adjust;
  }
}

// Fused copy + epoch adjustment kernel for 8-byte types (TIMESTAMP/DATETIME: int64)
// Source may be unaligned (MO vector header is 29 bytes), destination is aligned.
__global__ void copy_and_adjust_i64_kernel(const uint8_t* __restrict__ src,
                                           int64_t* __restrict__ dst,
                                           uint32_t n_rows,
                                           int64_t adjust)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    int64_t val;
    memcpy(&val, src + i * sizeof(int64_t), sizeof(int64_t));
    dst[i] = val - adjust;
  }
}

// ---------------------------------------------------------------------------
// Batched kernels — 2D grid: blockIdx.y selects descriptor, blockIdx.x tiles rows
// ---------------------------------------------------------------------------

// Byte-level batched memcpy (non-temporal fixed-width).
__global__ void batched_memcpy_kernel(const BatchedFixedDesc* __restrict__ descs,
                                      uint8_t* __restrict__ dst,
                                      uint32_t elem_size)
{
  auto const& desc = descs[blockIdx.y];
  uint32_t bytes   = desc.n_rows * elem_size;
  auto* src_base   = desc.src;
  auto* dst_base   = dst + static_cast<std::size_t>(desc.row_offset) * elem_size;

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += gridDim.x * blockDim.x) {
    dst_base[i] = !desc.has_data ? uint8_t{0} : src_base[desc.is_constant ? (i % elem_size) : i];
  }
}

// Batched copy + epoch adjust for 4-byte types (DATE: int32)
__global__ void batched_copy_adjust_i32_kernel(const BatchedFixedDesc* __restrict__ descs,
                                               int32_t* __restrict__ dst,
                                               int32_t adjust)
{
  auto const& desc = descs[blockIdx.y];
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < desc.n_rows;
       i += gridDim.x * blockDim.x) {
    int32_t val;
    if (!desc.has_data) {
      dst[desc.row_offset + i] = 0;
    } else {
      auto const src_row = desc.is_constant ? 0 : i;
      memcpy(&val, desc.src + src_row * sizeof(int32_t), sizeof(int32_t));
      dst[desc.row_offset + i] = val - adjust;
    }
  }
}

// Batched copy + epoch adjust for 8-byte types (TIMESTAMP/DATETIME: int64)
__global__ void batched_copy_adjust_i64_kernel(const BatchedFixedDesc* __restrict__ descs,
                                               int64_t* __restrict__ dst,
                                               int64_t adjust)
{
  auto const& desc = descs[blockIdx.y];
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < desc.n_rows;
       i += gridDim.x * blockDim.x) {
    int64_t val;
    if (!desc.has_data) {
      dst[desc.row_offset + i] = 0;
    } else {
      auto const src_row = desc.is_constant ? 0 : i;
      memcpy(&val, desc.src + src_row * sizeof(int64_t), sizeof(int64_t));
      dst[desc.row_offset + i] = val - adjust;
    }
  }
}

}  // anonymous namespace

void decode_fixed_width(const uint8_t* d_src,
                        uint8_t* d_dst,
                        uint32_t n_rows,
                        uint32_t elem_size,
                        int64_t epoch_adjust,
                        rmm::cuda_stream_view stream)
{
  if (n_rows == 0) return;

  if (epoch_adjust == 0) {
    // No adjustment — caller already uses cudaMemcpyAsync D2D for this case,
    // but handle it here as fallback.
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(d_dst, d_src, n_rows * elem_size, cudaMemcpyDeviceToDevice, stream.value()));
  } else {
    // Fused copy + epoch adjustment in a single pass (halves memory bandwidth)
    uint32_t row_blocks = (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (elem_size == 4) {
      copy_and_adjust_i32_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
        d_src, reinterpret_cast<int32_t*>(d_dst), n_rows, static_cast<int32_t>(epoch_adjust));
    } else if (elem_size == 8) {
      copy_and_adjust_i64_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
        d_src, reinterpret_cast<int64_t*>(d_dst), n_rows, epoch_adjust);
    } else {
      // Fallback for unusual element sizes with epoch adjustment — shouldn't happen
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_dst, d_src, n_rows * elem_size, cudaMemcpyDeviceToDevice, stream.value()));
    }
  }
}

void batched_decode_fixed_width(const BatchedFixedDesc* d_descs,
                                uint32_t n_descs,
                                uint8_t* d_dst,
                                uint32_t elem_size,
                                int64_t epoch_adjust,
                                uint32_t max_block_rows,
                                rmm::cuda_stream_view stream)
{
  if (n_descs == 0 || max_block_rows == 0) return;

  if (epoch_adjust == 0) {
    // Byte-level batched copy
    uint32_t max_bytes = max_block_rows * elem_size;
    uint32_t grid_x    = (max_bytes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(grid_x, n_descs);
    batched_memcpy_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.value()>>>(
      d_descs, d_dst, elem_size);
  } else if (elem_size == 4) {
    uint32_t grid_x = (max_block_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(grid_x, n_descs);
    batched_copy_adjust_i32_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.value()>>>(
      d_descs, reinterpret_cast<int32_t*>(d_dst), static_cast<int32_t>(epoch_adjust));
  } else if (elem_size == 8) {
    uint32_t grid_x = (max_block_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(grid_x, n_descs);
    batched_copy_adjust_i64_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.value()>>>(
      d_descs, reinterpret_cast<int64_t*>(d_dst), epoch_adjust);
  } else {
    // Fallback: byte-level copy
    uint32_t max_bytes = max_block_rows * elem_size;
    uint32_t grid_x    = (max_bytes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(grid_x, n_descs);
    batched_memcpy_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.value()>>>(
      d_descs, d_dst, elem_size);
  }
}

}  // namespace sirius::cuda::tae
