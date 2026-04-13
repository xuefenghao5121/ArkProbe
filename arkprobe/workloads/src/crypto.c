/*
 * arkprobe_crypto — Cryptographic micro-kernel workload.
 *
 * Models key characteristics of cryptographic workloads:
 *   1. AES-like encryption: S-Box lookups, XOR operations, byte shuffling
 *   2. SHA-256 hashing: Bit rotations, message schedule, compression
 *   3. Modular arithmetic: Big integer operations for RSA-like workloads
 *
 * This workload produces:
 *   - High compute intensity (compute-bound)
 *   - Heavy bit manipulation and table lookups
 *   - Low branch misprediction (predictable patterns)
 *   - SIMD-friendly operations (if hardware crypto extensions available)
 *
 * Usage: arkprobe_crypto --threads N --duration S [--mode aes|sha256|mixed]
 * Output: prints throughput (MB/s) and operation statistics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define BLOCK_SIZE 16        /* AES block size */
#define SHA_BLOCK_SIZE 64    /* SHA-256 block size */
#define BUFFER_SIZE (16 * 1024 * 1024)  /* 16MB buffer */
#define TABLE_SIZE 256

/* AES S-Box (substitution box) */
static const uint8_t SBOX[TABLE_SIZE] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

/* Inverse S-Box for decryption */
static const uint8_t INV_SBOX[TABLE_SIZE] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

/* SHA-256 round constants */
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Worker statistics */
typedef struct {
    long operations;
    long bytes_processed;
    long aes_ops;
    long sha_ops;
} worker_stats_t;

/* Global state */
static uint8_t *g_buffer = NULL;
static volatile int g_running = 1;
static int g_mode = 0;  /* 0 = mixed, 1 = aes, 2 = sha256 */

/* Rotate right */
static inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* AES SubBytes transformation */
static void aes_sub_bytes(uint8_t *state) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        state[i] = SBOX[state[i]];
    }
}

/* AES ShiftRows transformation */
static void aes_shift_rows(uint8_t *state) {
    uint8_t temp;
    /* Row 1: shift left by 1 */
    temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;
    /* Row 2: shift left by 2 */
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;
    /* Row 3: shift left by 3 */
    temp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = temp;
}

/* AES MixColumns transformation (simplified) */
static void aes_mix_columns(uint8_t *state) {
    for (int i = 0; i < 4; i++) {
        uint8_t a = state[i * 4];
        uint8_t b = state[i * 4 + 1];
        uint8_t c = state[i * 4 + 2];
        uint8_t d = state[i * 4 + 3];

        state[i * 4]     = a ^ b ^ c ^ d;
        state[i * 4 + 1] = a ^ b ^ c ^ d;
        state[i * 4 + 2] = a ^ b ^ c ^ d;
        state[i * 4 + 3] = a ^ b ^ c ^ d;
    }
}

/* AES AddRoundKey (simplified - XOR with key derived from data) */
static void aes_add_round_key(uint8_t *state, const uint8_t *key) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        state[i] ^= key[i];
    }
}

/* Simulate AES encryption round */
static void aes_encrypt_block(uint8_t *block, const uint8_t *key) {
    /* Initial round key */
    aes_add_round_key(block, key);

    /* 10 rounds for AES-128 */
    for (int round = 0; round < 10; round++) {
        aes_sub_bytes(block);
        aes_shift_rows(block);
        aes_mix_columns(block);
        /* Derive round key from block position */
        uint8_t round_key[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            round_key[i] = key[i] ^ (round * 0x11);
        }
        aes_add_round_key(block, round_key);
    }

    /* Final round (no MixColumns) */
    aes_sub_bytes(block);
    aes_shift_rows(block);
    aes_add_round_key(block, key);
}

/* SHA-256 compression function (simplified) */
static void sha256_compress(uint32_t *state, const uint8_t *block) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

    /* Prepare message schedule */
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    /* Initialize working variables */
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    /* Main loop */
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        t1 = h + S1 + ch + K[i] + W[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        t2 = S0 + maj;

        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    /* Add compressed chunk to current hash value */
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* SHA-256 hash computation */
static void sha256_hash(const uint8_t *data, size_t len, uint8_t *digest) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    /* Process complete blocks */
    size_t blocks = len / SHA_BLOCK_SIZE;
    for (size_t i = 0; i < blocks; i++) {
        sha256_compress(state, data + i * SHA_BLOCK_SIZE);
    }

    /* Output digest */
    for (int i = 0; i < 8; i++) {
        digest[i * 4]     = (state[i] >> 24) & 0xFF;
        digest[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        digest[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        digest[i * 4 + 3] = state[i] & 0xFF;
    }
}

/* Do cryptographic operation */
static void do_crypto_op(worker_stats_t *stats) {
    size_t offset = rand() % (BUFFER_SIZE - BLOCK_SIZE * 2);
    uint8_t *block = g_buffer + offset;

    int op = g_mode;
    if (op == 0) {
        op = (rand() % 2) + 1;  /* Random: AES or SHA-256 */
    }

    if (op == 1) {
        /* AES encryption */
        uint8_t key[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            key[i] = rand() & 0xFF;
        }
        aes_encrypt_block(block, key);
        stats->aes_ops++;
        stats->bytes_processed += BLOCK_SIZE;
    } else {
        /* SHA-256 hashing */
        uint8_t digest[32];
        sha256_hash(block, SHA_BLOCK_SIZE, digest);
        stats->sha_ops++;
        stats->bytes_processed += SHA_BLOCK_SIZE;
    }

    stats->operations++;
}

static void *worker(void *arg) {
    worker_stats_t *stats = (worker_stats_t *)arg;

    while (g_running) {
        do_crypto_op(stats);
    }

    return NULL;
}

static void *timer_thread(void *arg) {
    int seconds = *(int *)arg;
    struct timespec ts = {seconds, 0};
    nanosleep(&ts, NULL);
    g_running = 0;
    return NULL;
}

int main(int argc, char *argv[]) {
    int threads = 1;
    int duration = 60;
    int mode = 0;  /* 0 = mixed, 1 = aes, 2 = sha256 */

    static struct option long_opts[] = {
        {"threads",  required_argument, 0, 't'},
        {"duration", required_argument, 0, 'd'},
        {"mode",     required_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:m:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
            case 'm':
                if (strcmp(optarg, "aes") == 0) mode = 1;
                else if (strcmp(optarg, "sha256") == 0) mode = 2;
                else mode = 0;
                break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;

    g_mode = mode;

    /* Initialize buffer with random data */
    g_buffer = malloc(BUFFER_SIZE);
    for (size_t i = 0; i < BUFFER_SIZE; i++) {
        g_buffer[i] = rand() & 0xFF;
    }

    /* Start workers */
    worker_stats_t *stats = calloc(threads, sizeof(worker_stats_t));
    pthread_t *tids = malloc(sizeof(pthread_t) * (threads + 1));

    pthread_create(&tids[threads], NULL, timer_thread, &duration);

    for (int i = 0; i < threads; i++) {
        pthread_create(&tids[i], NULL, worker, &stats[i]);
    }

    for (int i = 0; i <= threads; i++) {
        pthread_join(tids[i], NULL);
    }

    /* Aggregate results */
    long total_ops = 0, total_bytes = 0, total_aes = 0, total_sha = 0;
    for (int i = 0; i < threads; i++) {
        total_ops += stats[i].operations;
        total_bytes += stats[i].bytes_processed;
        total_aes += stats[i].aes_ops;
        total_sha += stats[i].sha_ops;
    }

    double throughput_mbps = (double)total_bytes / duration / (1024 * 1024);
    double ops_per_sec = (double)total_ops / duration;

    printf("Throughput: %.2f MB/s\n", throughput_mbps);
    printf("Operations: %.2f ops/sec\n", ops_per_sec);
    printf("AES ops: %ld\n", total_aes);
    printf("SHA-256 ops: %ld\n", total_sha);

    /* Cleanup */
    free(g_buffer);
    free(stats);
    free(tids);

    return 0;
}
