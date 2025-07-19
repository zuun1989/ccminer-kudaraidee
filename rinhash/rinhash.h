#ifndef RINHASH_H
#define RINHASH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// RinHash v2 algorithm - BLAKE3 → Argon2d → SHA3-256
// Memory cost: 256MB for blocks >= 170,000
void rinhash_v2(const char* input, char* output, uint32_t input_len, uint32_t memory_cost);

// Legacy RinHash v1 - 64KB memory cost
void rinhash_v1(const char* input, char* output, uint32_t input_len);

// Main RinHash function with auto version detection
void rinhash(const char* input, char* output, uint32_t input_len, uint32_t block_height);

#ifdef __cplusplus
}
#endif

#endif // RINHASH_H
