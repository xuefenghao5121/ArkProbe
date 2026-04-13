/*
 * arkprobe_compress — Compression micro-kernel workload.
 *
 * Models key characteristics of compression workloads (LZ77/ZSTD/LZ4 style):
 *   1. Sliding window: Hash table lookups for match finding
 *   2. Literal copying: Memory copy operations for unmatched data
 *   3. Match encoding: Variable-length encoding of (offset, length) pairs
 *   4. Hash chain traversal: Pointer chasing for finding best matches
 *
 * This workload produces:
 *   - Mixed compute/memory patterns
 *   - Hash table lookups (cache misses)
 *   - Sequential and random memory access
 *   - Branch-heavy code paths (match/no-match decisions)
 *
 * Usage: arkprobe_compress --threads N --duration S [--level L]
 * Output: prints throughput (MB/s) and compression ratio.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define INPUT_SIZE (8 * 1024 * 1024)   /* 8MB input buffer */
#define OUTPUT_SIZE (16 * 1024 * 1024) /* 16MB output buffer */
#define WINDOW_SIZE (64 * 1024)        /* 64KB sliding window */
#define HASH_SIZE 16384                /* Hash table size */
#define MIN_MATCH 4                    /* Minimum match length */
#define MAX_MATCH 256                  /* Maximum match length */
#define HASH_CHAIN_LEN 128             /* Hash chain depth */

/* Hash function (FNV-1a variant) */
static inline uint32_t hash4(const uint8_t *p) {
    return ((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24)) * 2654435761U;
}

/* Compression state */
typedef struct {
    const uint8_t *input;
    size_t input_size;
    uint8_t *output;
    size_t output_pos;

    /* Hash table: points to position in input */
    uint32_t hash_table[HASH_SIZE];

    /* Hash chains for collision resolution */
    uint32_t *hash_chain;
    uint32_t hash_chain_head[HASH_SIZE];
} compress_state_t;

/* Worker statistics */
typedef struct {
    long bytes_in;
    long bytes_out;
    long matches;
    long literals;
    long hash_lookups;
    long hash_probes;
} worker_stats_t;

/* Global state */
static uint8_t *g_input = NULL;
static uint8_t *g_output = NULL;
static volatile int g_running = 1;
static int g_level = 3;  /* Compression level 1-9 */

/* Initialize compression state */
static void compress_init(compress_state_t *state, const uint8_t *input, size_t input_size,
                          uint8_t *output) {
    state->input = input;
    state->input_size = input_size;
    state->output = output;
    state->output_pos = 0;

    memset(state->hash_table, 0xFF, sizeof(state->hash_table));
    state->hash_chain = calloc(input_size, sizeof(uint32_t));
    memset(state->hash_chain_head, 0xFF, sizeof(state->hash_chain_head));
}

/* Find best match using hash chains */
static int find_match(compress_state_t *state, size_t pos, worker_stats_t *stats,
                      uint32_t *match_offset) {
    if (pos + MIN_MATCH > state->input_size) {
        return 0;
    }

    uint32_t h = hash4(state->input + pos) % HASH_SIZE;
    uint32_t chain_pos = state->hash_table[h];

    int best_len = 0;
    uint32_t best_offset = 0;
    int probes = 0;
    int max_probes = HASH_CHAIN_LEN * (g_level / 3 + 1);

    stats->hash_lookups++;

    while (chain_pos != 0xFFFFFFFF && probes < max_probes) {
        stats->hash_probes++;
        probes++;

        /* Check if position is within window */
        if (pos - chain_pos > WINDOW_SIZE) {
            break;
        }

        /* Compare bytes */
        const uint8_t *match = state->input + chain_pos;
        const uint8_t *cur = state->input + pos;

        int len = 0;
        int max_len = state->input_size - pos;
        if (max_len > MAX_MATCH) max_len = MAX_MATCH;

        /* Fast match check */
        if (match[0] == cur[0] && match[1] == cur[1] &&
            match[2] == cur[2] && match[3] == cur[3]) {
            /* Full match */
            len = MIN_MATCH;
            while (len < max_len && match[len] == cur[len]) {
                len++;
            }

            if (len > best_len) {
                best_len = len;
                best_offset = pos - chain_pos;

                /* Early exit for good matches */
                if (len >= MAX_MATCH / 2) {
                    break;
                }
            }
        }

        /* Follow hash chain */
        chain_pos = state->hash_chain[chain_pos];
    }

    if (best_len >= MIN_MATCH) {
        *match_offset = best_offset;
        return best_len;
    }

    return 0;
}

/* Update hash table */
static void update_hash(compress_state_t *state, size_t pos) {
    if (pos + MIN_MATCH > state->input_size) {
        return;
    }

    uint32_t h = hash4(state->input + pos) % HASH_SIZE;

    /* Insert at head of hash chain */
    state->hash_chain[pos] = state->hash_table[h];
    state->hash_table[h] = pos;
}

/* Encode a literal byte */
static void emit_literal(compress_state_t *state, uint8_t byte) {
    /* Simple encoding: 0x00 + byte for literal */
    state->output[state->output_pos++] = 0x00;
    state->output[state->output_pos++] = byte;
}

/* Encode a match */
static void emit_match(compress_state_t *state, uint32_t offset, int length) {
    /* Simple encoding: 0x01 + offset (2 bytes) + length (1 byte) */
    state->output[state->output_pos++] = 0x01;
    state->output[state->output_pos++] = (offset >> 8) & 0xFF;
    state->output[state->output_pos++] = offset & 0xFF;
    state->output[state->output_pos++] = length & 0xFF;
}

/* Compress a chunk */
static size_t compress_chunk(compress_state_t *state, worker_stats_t *stats) {
    size_t pos = 0;

    while (pos < state->input_size && g_running) {
        uint32_t offset;
        int match_len = find_match(state, pos, stats, &offset);

        if (match_len >= MIN_MATCH) {
            /* Emit match */
            emit_match(state, offset, match_len);
            stats->matches++;
            stats->bytes_out += 4;

            /* Update hash for matched bytes */
            for (int i = 0; i < match_len; i++) {
                update_hash(state, pos + i);
            }

            pos += match_len;
            stats->bytes_in += match_len;
        } else {
            /* Emit literal */
            emit_literal(state, state->input[pos]);
            update_hash(state, pos);
            stats->literals++;
            stats->bytes_in++;
            stats->bytes_out += 2;
            pos++;
        }
    }

    return state->output_pos;
}

/* Decompress for verification (simplified) */
static void decompress_chunk(const uint8_t *compressed, size_t comp_size,
                             uint8_t *decompressed, size_t *decomp_size) {
    size_t in_pos = 0;
    size_t out_pos = 0;

    while (in_pos < comp_size) {
        if (compressed[in_pos] == 0x00) {
            /* Literal */
            decompressed[out_pos++] = compressed[in_pos + 1];
            in_pos += 2;
        } else if (compressed[in_pos] == 0x01) {
            /* Match */
            uint32_t offset = (compressed[in_pos + 1] << 8) | compressed[in_pos + 2];
            int length = compressed[in_pos + 3];

            /* Copy from previous output */
            for (int i = 0; i < length; i++) {
                decompressed[out_pos] = decompressed[out_pos - offset];
                out_pos++;
            }
            in_pos += 4;
        } else {
            break;
        }
    }

    *decomp_size = out_pos;
}

/* Do compression work */
static void do_compress(worker_stats_t *stats) {
    compress_state_t state;
    size_t chunk_size = INPUT_SIZE / 4;  /* Process 1/4 of buffer per iteration */
    size_t offset = rand() % (INPUT_SIZE - chunk_size);

    compress_init(&state, g_input + offset, chunk_size, g_output);
    compress_chunk(&state, stats);

    free(state.hash_chain);
}

static void *worker(void *arg) {
    worker_stats_t *stats = (worker_stats_t *)arg;

    while (g_running) {
        do_compress(stats);
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

/* Generate compressible data */
static void generate_data(uint8_t *buf, size_t size, int entropy) {
    /* Create data with varying compressibility */
    size_t pos = 0;

    while (pos < size) {
        int pattern_len = 16 + (rand() % 256);
        int repeat = 1 + (rand() % (entropy + 1));

        if (pos + pattern_len * repeat > size) {
            pattern_len = (size - pos) / (repeat + 1);
            if (pattern_len < 1) pattern_len = 1;
        }

        /* Generate pattern */
        for (int i = 0; i < pattern_len && pos + i < size; i++) {
            buf[pos + i] = rand() % 256;
        }

        /* Repeat pattern */
        for (int r = 1; r < repeat; r++) {
            for (int i = 0; i < pattern_len && pos + r * pattern_len + i < size; i++) {
                buf[pos + r * pattern_len + i] = buf[pos + i];
            }
        }

        pos += pattern_len * repeat;
    }
}

int main(int argc, char *argv[]) {
    int threads = 1;
    int duration = 60;
    int level = 3;

    static struct option long_opts[] = {
        {"threads",  required_argument, 0, 't'},
        {"duration", required_argument, 0, 'd'},
        {"level",    required_argument, 0, 'l'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:l:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
            case 'l': level = atoi(optarg); break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;
    if (level < 1) level = 1;
    if (level > 9) level = 9;

    g_level = level;

    /* Initialize buffers */
    g_input = malloc(INPUT_SIZE);
    g_output = malloc(OUTPUT_SIZE);

    /* Generate compressible test data */
    generate_data(g_input, INPUT_SIZE, level);

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
    long total_in = 0, total_out = 0, total_matches = 0, total_literals = 0;
    long total_lookups = 0, total_probes = 0;

    for (int i = 0; i < threads; i++) {
        total_in += stats[i].bytes_in;
        total_out += stats[i].bytes_out;
        total_matches += stats[i].matches;
        total_literals += stats[i].literals;
        total_lookups += stats[i].hash_lookups;
        total_probes += stats[i].hash_probes;
    }

    double throughput_mbps = (double)total_in / duration / (1024 * 1024);
    double ratio = total_out > 0 ? (double)total_in / total_out : 1.0;
    double avg_probes = total_lookups > 0 ? (double)total_probes / total_lookups : 0;

    printf("Throughput: %.2f MB/s\n", throughput_mbps);
    printf("Compression ratio: %.2fx\n", ratio);
    printf("Matches: %ld\n", total_matches);
    printf("Literals: %ld\n", total_literals);
    printf("Avg hash probes: %.2f\n", avg_probes);

    /* Cleanup */
    free(g_input);
    free(g_output);
    free(stats);
    free(tids);

    return 0;
}
