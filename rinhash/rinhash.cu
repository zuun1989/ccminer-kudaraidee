#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept>

#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "sha3_256_device.cuh"
#include "blake3_device.cuh"

// Device constant for salt
__device__ __constant__ uint8_t kRinSalt[11] = {'R','i','n','C','o','i','n','S','a','l','t'};

// Kernel for single hash
__global__ void rinhash_cuda_kernel(
    const uint8_t* input, size_t input_len, uint8_t* output, block* argon2_mem, uint32_t m_cost
) {
    uint8_t blake3_out[32];
    light_hash_device(input, input_len, blake3_out);

    uint8_t argon2_out[32];
    device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, argon2_mem, kRinSalt, 11);

    sha3_256_device(argon2_out, 32, output);
}

// Kernel for batch hash
__global__ void rinhash_cuda_kernel_batch(
    const uint8_t* headers, size_t header_len, uint8_t* outputs, uint32_t num_blocks,
    block* argon2_mem, uint32_t m_cost
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    const uint8_t* input = headers + tid * header_len;
    uint8_t* output = outputs + tid * 32;

    uint8_t blake3_out[32];
    light_hash_device(input, header_len, blake3_out);

    block* memory = argon2_mem + tid * m_cost;
    uint8_t argon2_out[32];
    device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, kRinSalt, 11);

    sha3_256_device(argon2_out, 32, output);
}

// Host function for single hash
extern "C" void rinhash_cuda(const uint8_t* input, size_t input_len, uint8_t* output) {
    uint8_t *d_input = nullptr, *d_output = nullptr;
    block* d_argon2_mem = nullptr;
    const uint32_t m_cost = 64;
    cudaMalloc(&d_input, input_len);
    cudaMalloc(&d_output, 32);
    cudaMalloc(&d_argon2_mem, m_cost * sizeof(block));

    cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);

    rinhash_cuda_kernel<<<1, 1>>>(d_input, input_len, d_output, d_argon2_mem, m_cost);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_argon2_mem);
}

// Host function for batch hash
extern "C" void rinhash_cuda_batch(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    uint8_t *d_headers = nullptr, *d_outputs = nullptr;
    block* d_argon2_mem = nullptr;
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;
    const uint32_t m_cost = 64;
    cudaMalloc(&d_headers, headers_size);
    cudaMalloc(&d_outputs, outputs_size);
    cudaMalloc(&d_argon2_mem, num_blocks * m_cost * sizeof(block));

    cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);

    const int threads_per_block = 128;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    rinhash_cuda_kernel_batch<<<blocks, threads_per_block>>>(
        d_headers, block_header_len, d_outputs, num_blocks, d_argon2_mem, m_cost
    );
    cudaDeviceSynchronize();

    cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);

    cudaFree(d_headers);
    cudaFree(d_outputs);
    cudaFree(d_argon2_mem);
}

// Helper: build block header as bytes
extern "C" void blockheader_to_bytes(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output,
    size_t* output_len
) {
    size_t offset = 0;
    memcpy(output + offset, version, 4); offset += 4;
    memcpy(output + offset, prev_block, 32); offset += 32;
    memcpy(output + offset, merkle_root, 32); offset += 32;
    memcpy(output + offset, timestamp, 4); offset += 4;
    memcpy(output + offset, bits, 4); offset += 4;
    memcpy(output + offset, nonce, 4); offset += 4;
    *output_len = offset;
}

// Main entry for single hash
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
) {
    uint8_t block_header[80];
    size_t block_header_len;
    blockheader_to_bytes(
        version, prev_block, merkle_root, timestamp, bits, nonce,
        block_header, &block_header_len
    );
    rinhash_cuda(block_header, block_header_len, output);
}

bool is_better(uint8_t* hash1, uint8_t* hash2) {
    for (int i = 7; i >= 0; i--) {
        uint32_t h1 = ((uint32_t)hash1[i*4 + 0]) |
                      ((uint32_t)hash1[i*4 + 1] << 8) |
                      ((uint32_t)hash1[i*4 + 2] << 16) |
                      ((uint32_t)hash1[i*4 + 3] << 24);
        uint32_t h2 = ((uint32_t)hash2[i*4 + 0]) |
                      ((uint32_t)hash2[i*4 + 1] << 8) |
                      ((uint32_t)hash2[i*4 + 2] << 16) |
                      ((uint32_t)hash2[i*4 + 3] << 24);
        if (h1 < h2) return true;
        if (h1 > h2) return false;
    }
    return false;
}

// Mining function
extern "C" void RinHash_mine(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash
) {
    const size_t block_header_len = 80;
    uint8_t block_headers[80 * 1024];
    uint8_t hashes[32 * 1024];
    // Prepare block headers
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        uint32_t work_data_copy[20];
        memcpy(work_data_copy, work_data, 80);
        work_data_copy[nonce_offset] = current_nonce;
        uint8_t* header = block_headers + i * block_header_len;
        memcpy(header, work_data_copy, 80);
    }
    // Calculate hashes
    rinhash_cuda_batch(block_headers, block_header_len, hashes, num_nonces);
    // Find best
    memcpy(best_hash, hashes, 32);
    *found_nonce = start_nonce;
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint8_t* current_hash = hashes + i * 32;
        if (is_better(current_hash, best_hash)) {
            memcpy(best_hash, current_hash, 32);
            *found_nonce = start_nonce + i;
        }
    }
}
