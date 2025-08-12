#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept>

// Include shared device functions (chỉ include .cuh hoặc .h)
#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "sha3-256.cu"
#include "blake3_device.cuh"

// Số block tối đa cho batch (tùy GPU)
#define MAX_BATCH_BLOCKS 16384

// Kernel đơn: mỗi lần chỉ chạy 1 thread
extern "C" __global__ void rinhash_cuda_kernel(
    const uint8_t* input, 
    size_t input_len, 
    uint8_t* output,
    block* memory,      // bộ nhớ argon2 đã cấp phát trên host, truyền vào
    uint32_t m_cost
) {
    // Chỉ 1 thread xử lý
    if (threadIdx.x == 0) {
        // Step 1: BLAKE3 hash - now using light_hash_device
        light_hash_device(input, input_len, blake3_out);
        // Step 2: Argon2d hash
        uint32_t m_cost = 64; // Example
        size_t memory_size = m_cost * sizeof(block);
        block* d_memory = (block*)malloc(memory_size);
        uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
        uint8_t argon2_out[32];
        device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, salt, sizeof(salt));

        uint8_t sha3_out[32];
        sha3_256_device(argon2_out, 32, sha3_out);
    }
}

// Kernel batch: mỗi thread xử lý 1 block header
extern "C" __global__ void rinhash_cuda_kernel_batch(
    const uint8_t* headers,         // num_blocks * 80 bytes
    size_t header_len,              // = 80
    uint8_t* outputs,               // num_blocks * 32 bytes
    uint32_t num_blocks,
    block* memories,                // num_blocks * m_cost * sizeof(block)
    uint32_t m_cost
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    const uint8_t* input = headers + tid * header_len;
    uint8_t* output = outputs + tid * 32;
    block* memory = memories + tid * m_cost;

    uint8_t blake3_out[32];
    light_hash_device(input, header_len, blake3_out);

    uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    uint8_t argon2_out[32];
    device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, salt, sizeof(salt));

    sha3_256_device(argon2_out, 32, output);
}


// Helper: kiểm tra lỗi CUDA
inline void check_cuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error("CUDA error");
    }
}

// RinHash CUDA implementation (single)
extern "C" void rinhash_cuda(const uint8_t* input, size_t input_len, uint8_t* output) {
    uint8_t *d_input = nullptr;
    uint8_t *d_output = nullptr;
    block* d_memory = nullptr;
    uint32_t m_cost = 64;

    cudaError_t err;

    // Alloc device memory
    err = cudaMalloc(&d_input, input_len);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc input fail\n"); return; }

    err = cudaMalloc(&d_output, 32);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc output fail\n"); cudaFree(d_input); return; }

    err = cudaMalloc(&d_memory, m_cost * sizeof(block));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc argon2 memory fail\n"); cudaFree(d_input); cudaFree(d_output); return; }

    // Copy input
    err = cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy input fail\n"); cudaFree(d_input); cudaFree(d_output); cudaFree(d_memory); return; }

    // Launch kernel
    rinhash_cuda_kernel<<<256, 1024>>>(d_input, input_len, d_output, d_memory, m_cost);
    cudaDeviceSynchronize();
    check_cuda("rinhash_cuda_kernel");

    // Copy result
    err = cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy output fail\n"); }

    // Free
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_memory);
}

// Batch processing version for mining
extern "C" void rinhash_cuda_batch(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    if (num_blocks > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }

    uint8_t *d_headers = nullptr, *d_outputs = nullptr;
    block* d_memories = nullptr;
    uint32_t m_cost = 64;
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;
    size_t memories_size = num_blocks * m_cost * sizeof(block);

    cudaError_t err;
    err = cudaMalloc(&d_headers, headers_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc headers fail\n"); return; }
    err = cudaMalloc(&d_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc outputs fail\n"); cudaFree(d_headers); return; }
    err = cudaMalloc(&d_memories, memories_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc argon2 memories fail\n"); cudaFree(d_headers); cudaFree(d_outputs); return; }

    cudaMemset(d_outputs, 0xee, outputs_size);
    cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);

    const int threads_per_block = 128;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    rinhash_cuda_kernel_batch<<<blocks, threads_per_block>>>(
        d_headers, block_header_len, d_outputs, num_blocks, d_memories, m_cost
    );
    cudaDeviceSynchronize();
    check_cuda("rinhash_cuda_kernel_batch");

    err = cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy output batch fail\n"); }

    cudaFree(d_headers);
    cudaFree(d_outputs);
    cudaFree(d_memories);
}

// Helper function to convert a block header to bytes
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

// Main RinHash function that would be called from outside
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
) {
    uint8_t block_header[80]; // Standard block header size
    size_t block_header_len;
    blockheader_to_bytes(
        version,
        prev_block,
        merkle_root,
        timestamp,
        bits,
        nonce,
        block_header,
        &block_header_len
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
    return false; // equal
}

// Mining function that tries different nonces
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
    int headerbytes = block_header_len * num_nonces;
    int hashbytes = 32 * num_nonces;
    uint8_t block_headers[80 * 1024];
    uint8_t hashes[32 * 1024];
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024); // 128MB
    // Prepare block headers with different nonces
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        uint32_t work_data_copy[20];
        memcpy(work_data_copy, work_data, 80);
        work_data_copy[nonce_offset] = current_nonce;
        memcpy(&block_headers[i * block_header_len], work_data_copy, 80);
    }

    // Calculate hashes for all nonces
    rinhash_cuda_batch(block_headers.data(), block_header_len, hashes.data(), num_nonces);

    // Initialize best_hash with the first hash
    memcpy(best_hash, hashes.data(), 32);
    *found_nonce = start_nonce;
    for (uint32_t i = 1; i < num_nonces; i++) {
        uint8_t* current_hash = hashes.data() + i * 32;
        if (is_better(current_hash, best_hash)) {
            memcpy(best_hash, current_hash, 32);
            *found_nonce = start_nonce + i;
        }
    }
}
