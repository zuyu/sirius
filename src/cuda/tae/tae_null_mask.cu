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

#include <cuda/tae/tae_decode_kernels.hpp>
#include <cuda_runtime.h>

#include <cstdint>

namespace sirius::cuda::tae {

namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

__global__ void invert_mask_kernel(const uint8_t* __restrict__ src,
                                   uint32_t* __restrict__ dst,
                                   uint32_t n_rows,
                                   uint32_t row_offset)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
       i += gridDim.x * blockDim.x) {
    if ((src[i / 8] & (uint8_t{1} << (i % 8))) != 0) {
      auto const output_row = row_offset + i;
      atomicAnd(dst + output_row / 32, ~(uint32_t{1} << (output_row % 32)));
    }
  }
}

// Batched null-mask conversion. Blocks can begin in the middle of an output
// word, so clear individual null bits atomically in an ALL_VALID destination.
__global__ void batched_invert_mask_kernel(const BatchedNullMaskDesc* __restrict__ descs,
                                           uint32_t* __restrict__ dst)
{
  auto const& desc = descs[blockIdx.y];
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < desc.n_rows;
       i += gridDim.x * blockDim.x) {
    if (i / 8 < desc.bitmap_bytes && (desc.src[i / 8] & (uint8_t{1} << (i % 8))) != 0) {
      auto const output_row = desc.row_offset + i;
      atomicAnd(dst + output_row / 32, ~(uint32_t{1} << (output_row % 32)));
    }
  }
}

}  // anonymous namespace

void invert_null_mask(const uint8_t* d_src,
                      uint32_t* d_dst,
                      uint32_t n_rows,
                      rmm::cuda_stream_view stream)
{
  if (n_rows == 0) return;

  uint32_t blocks = (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  invert_mask_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(d_src, d_dst, n_rows, 0);
}

void batched_invert_null_mask(const BatchedNullMaskDesc* d_descs,
                              uint32_t n_descs,
                              uint32_t* d_validity,
                              rmm::cuda_stream_view stream)
{
  if (n_descs == 0) return;

  // One x-block is enough for the current 8192-row MatrixOne block size;
  // grid-stride iteration also keeps this correct for larger blocks.
  uint32_t grid_x = 1;
  dim3 grid(grid_x, n_descs);
  batched_invert_mask_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.value()>>>(d_descs, d_validity);
}

}  // namespace sirius::cuda::tae
