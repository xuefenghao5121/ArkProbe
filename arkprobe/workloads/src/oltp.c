/*
 * arkprobe_oltp — Database OLTP micro-kernel workload.
 *
 * Models key characteristics of database OLTP workloads:
 *   1. Lock contention: Row-level mutex simulation with random hold times
 *   2. B+tree index traversal: Random tree walks with cache-unfriendly access
 *   3. WAL writes: Sequential append-only writes (log buffer)
 *   4. Buffer pool: Mixed hot/cold page access patterns
 *
 * This workload produces:
 *   - High lock contention (futex waits, context switches)
 *   - Random cache misses (B+tree traversal)
 *   - Sequential write bandwidth (WAL)
 *   - Mixed instruction mix (compare, branch, memory access)
 *
 * Usage: arkprobe_oltp --threads N --duration S [--rows R] [--hot-ratio H]
 * Output: prints TPS (transactions/sec) and lock wait statistics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define DEFAULT_ROWS 100000
#define DEFAULT_HOT_RATIO 0.20  /* 20% hot rows get 80% access */
#define LOCK_TABLE_SIZE 65536   /* Lock hash table size */
#define WAL_BUFFER_SIZE (16 * 1024 * 1024)  /* 16MB WAL buffer */
#define PAGE_SIZE 8192

/* Simulated row with lock */
typedef struct {
    pthread_mutex_t lock;
    uint64_t version;
    char data[64];  /* Row data */
} row_t;

/* B+tree node (simplified) */
typedef struct btnode {
    int level;
    int nkeys;
    uint64_t keys[16];
    struct btnode *children[17];
    row_t *rows[16];  /* Leaf nodes point to rows */
} btnode_t;

/* WAL buffer */
typedef struct {
    char *buffer;
    size_t size;
    size_t head;
    pthread_mutex_t lock;
} wal_t;

/* Worker statistics */
typedef struct {
    long transactions;
    long lock_waits;
    long lock_wait_ns;
    long cache_misses;
    long wal_bytes;
} worker_stats_t;

/* Global state */
static row_t *g_rows = NULL;
static int g_nrows = DEFAULT_ROWS;
static double g_hot_ratio = DEFAULT_HOT_RATIO;
static wal_t g_wal;
static volatile int g_running = 1;

/* Lock hash table for row-level locking */
static pthread_mutex_t g_lock_table[LOCK_TABLE_SIZE];

static inline uint32_t hash_row(uint64_t rowid) {
    return (uint32_t)((rowid * 2654435761U) % LOCK_TABLE_SIZE);
}

/* Initialize a B+tree with random structure */
static btnode_t *create_btree(int nrows, row_t *rows, int level) {
    if (nrows <= 16 || level > 4) {
        /* Leaf node */
        btnode_t *node = calloc(1, sizeof(btnode_t));
        node->level = 0;
        node->nkeys = nrows < 16 ? nrows : 16;
        for (int i = 0; i < node->nkeys; i++) {
            node->keys[i] = rows[i].version;
            node->rows[i] = &rows[i];
        }
        return node;
    }

    btnode_t *node = calloc(1, sizeof(btnode_t));
    node->level = level;
    node->nkeys = 4;

    /* Create children */
    int chunk = nrows / 5;
    for (int i = 0; i <= node->nkeys; i++) {
        int start = i * chunk;
        int count = (i == node->nkeys) ? (nrows - start) : chunk;
        if (count > 0) {
            node->children[i] = create_btree(count, rows + start, level + 1);
            if (node->children[i] && node->children[i]->nkeys > 0) {
                node->keys[i] = node->children[i]->keys[0];
            }
        }
    }
    return node;
}

/* Traverse B+tree to find a row (cache-unfriendly) */
static row_t *btree_lookup(btnode_t *root, uint64_t key) {
    if (!root) return NULL;

    btnode_t *node = root;
    while (node->level > 0) {
        int i;
        for (i = 0; i < node->nkeys && key >= node->keys[i]; i++);
        if (i > 16) i = 16;
        node = node->children[i];
        if (!node) return NULL;
    }

    /* Leaf node - linear search (realistic for small leaf) */
    for (int i = 0; i < node->nkeys; i++) {
        if (node->rows[i] && node->rows[i]->version == key) {
            return node->rows[i];
        }
    }
    return node->nkeys > 0 ? node->rows[0] : NULL;
}

/* Simulate WAL write */
static void wal_append(const void *data, size_t len) {
    pthread_mutex_lock(&g_wal.lock);
    size_t space = g_wal.size - g_wal.head;
    if (len > space) {
        /* Wrap around (simulate flush) */
        g_wal.head = 0;
    }
    memcpy(g_wal.buffer + g_wal.head, data, len);
    g_wal.head += len;
    pthread_mutex_unlock(&g_wal.lock);
}

/* Simulate a database transaction */
static void do_transaction(worker_stats_t *stats, btnode_t *btree, uint64_t *hot_rows, int nhot) {
    struct timespec ts1, ts2;

    /* 1. Choose row to access (zipfian: hot rows get most access) */
    uint64_t rowid;
    if ((double)rand() / RAND_MAX < 0.8 && nhot > 0) {
        /* 80% access hot rows */
        rowid = hot_rows[rand() % nhot];
    } else {
        /* 20% access random rows */
        rowid = (uint64_t)rand() % g_nrows;
    }

    /* 2. B+tree index lookup (causes random cache misses) */
    row_t *row = btree_lookup(btree, rowid);
    if (!row) row = &g_rows[rowid % g_nrows];

    /* 3. Acquire row lock (simulate lock contention) */
    uint32_t lock_slot = hash_row(rowid);
    pthread_mutex_t *lock = &g_lock_table[lock_slot];

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    pthread_mutex_lock(lock);
    clock_gettime(CLOCK_MONOTONIC, &ts2);

    long wait_ns = (ts2.tv_sec - ts1.tv_sec) * 1000000000L +
                   (ts2.tv_nsec - ts1.tv_nsec);
    stats->lock_wait_ns += wait_ns;
    if (wait_ns > 1000) stats->lock_waits++;

    /* 4. Read/modify row */
    row->version++;
    volatile char sum = 0;
    for (int i = 0; i < 64; i++) {
        sum += row->data[i];
    }

    /* 5. Write WAL record */
    char wal_record[128];
    memset(wal_record, rowid & 0xFF, sizeof(wal_record));
    wal_append(wal_record, sizeof(wal_record));
    stats->wal_bytes += sizeof(wal_record);

    /* 6. Release lock */
    pthread_mutex_unlock(lock);

    stats->transactions++;
}

static void *worker(void *arg) {
    worker_stats_t *stats = (worker_stats_t *)arg;

    /* Create local B+tree for this worker */
    btnode_t *btree = create_btree(g_nrows, g_rows, 1);

    /* Identify hot rows (20% of rows get 80% of access) */
    int nhot = (int)(g_nrows * g_hot_ratio);
    uint64_t *hot_rows = malloc(nhot * sizeof(uint64_t));
    for (int i = 0; i < nhot; i++) {
        hot_rows[i] = (uint64_t)rand() % g_nrows;
    }

    while (g_running) {
        do_transaction(stats, btree, hot_rows, nhot);
    }

    free(hot_rows);
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
    int nrows = DEFAULT_ROWS;
    double hot_ratio = DEFAULT_HOT_RATIO;

    static struct option long_opts[] = {
        {"threads",    required_argument, 0, 't'},
        {"duration",   required_argument, 0, 'd'},
        {"rows",       required_argument, 0, 'r'},
        {"hot-ratio",  required_argument, 0, 'H'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:r:H:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
            case 'r': nrows = atoi(optarg); break;
            case 'H': hot_ratio = atof(optarg); break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;
    if (nrows < 1000) nrows = 1000;
    if (hot_ratio < 0.01) hot_ratio = 0.01;
    if (hot_ratio > 0.5) hot_ratio = 0.5;

    g_nrows = nrows;
    g_hot_ratio = hot_ratio;

    /* Initialize rows */
    g_rows = calloc(g_nrows, sizeof(row_t));
    for (int i = 0; i < g_nrows; i++) {
        pthread_mutex_init(&g_rows[i].lock, NULL);
        g_rows[i].version = i;
        memset(g_rows[i].data, i & 0xFF, sizeof(g_rows[i].data));
    }

    /* Initialize lock table */
    for (int i = 0; i < LOCK_TABLE_SIZE; i++) {
        pthread_mutex_init(&g_lock_table[i], NULL);
    }

    /* Initialize WAL */
    g_wal.buffer = malloc(WAL_BUFFER_SIZE);
    g_wal.size = WAL_BUFFER_SIZE;
    g_wal.head = 0;
    pthread_mutex_init(&g_wal.lock, NULL);

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
    long total_tx = 0, total_waits = 0, total_wait_ns = 0, total_wal = 0;
    for (int i = 0; i < threads; i++) {
        total_tx += stats[i].transactions;
        total_waits += stats[i].lock_waits;
        total_wait_ns += stats[i].lock_wait_ns;
        total_wal += stats[i].wal_bytes;
    }

    double tps = (double)total_tx / duration;
    double avg_wait_us = total_waits > 0 ? (double)total_wait_ns / total_waits / 1000.0 : 0;
    double wal_mbps = (double)total_wal / duration / (1024 * 1024);

    printf("TPS: %.2f tx/sec\n", tps);
    printf("Lock waits: %ld\n", total_waits);
    printf("Avg lock wait: %.2f us\n", avg_wait_us);
    printf("WAL throughput: %.2f MB/s\n", wal_mbps);

    /* Cleanup */
    for (int i = 0; i < g_nrows; i++) {
        pthread_mutex_destroy(&g_rows[i].lock);
    }
    for (int i = 0; i < LOCK_TABLE_SIZE; i++) {
        pthread_mutex_destroy(&g_lock_table[i]);
    }
    pthread_mutex_destroy(&g_wal.lock);
    free(g_rows);
    free(g_wal.buffer);
    free(stats);
    free(tids);

    return 0;
}
