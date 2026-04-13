/*
 * arkprobe_kvstore — Key-Value Store micro-kernel workload.
 *
 * Models key characteristics of in-memory KV stores (Redis/Memcached style):
 *   1. Hash table lookup: Random access with high collision rate
 *   2. GET/SET operations: Mixed read/write with varying value sizes
 *   3. LRU eviction simulation: Linked list traversal for cache policy
 *   4. Expiration checking: Time-based key expiry scanning
 *
 * This workload produces:
 *   - Random cache misses (hash table lookups)
 *   - Mixed read/write patterns (GET/SET ratio)
 *   - Pointer chasing (LRU list traversal)
 *   - Branch mispredictions (hash collisions, key comparisons)
 *
 * Usage: arkprobe_kvstore --threads N --duration S [--keys K] [--ratio R]
 * Output: prints OPS (operations/sec) and hit rate statistics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define DEFAULT_KEYS 100000
#define DEFAULT_SET_RATIO 0.30  /* 30% SET, 70% GET */
#define HASH_BUCKETS 65536
#define MAX_KEY_LEN 16
#define MIN_VALUE_LEN 64
#define MAX_VALUE_LEN 1024
#define LRU_SIZE 10000

/* KV entry */
typedef struct kv_entry {
    char key[MAX_KEY_LEN];
    char *value;
    size_t value_len;
    uint64_t expire_time;
    struct kv_entry *hash_next;  /* Hash chain */
    struct kv_entry *lru_prev;   /* LRU list */
    struct kv_entry *lru_next;
} kv_entry_t;

/* Hash bucket */
typedef struct {
    kv_entry_t *head;
    pthread_mutex_t lock;
} hash_bucket_t;

/* LRU cache */
typedef struct {
    kv_entry_t *head;  /* Most recently used */
    kv_entry_t *tail;  /* Least recently used */
    int count;
    pthread_mutex_t lock;
} lru_cache_t;

/* Worker statistics */
typedef struct {
    long operations;
    long gets;
    long sets;
    long hits;
    long misses;
    long evictions;
    long expired;
} worker_stats_t;

/* Global state */
static kv_entry_t *g_entries = NULL;
static int g_nkeys = DEFAULT_KEYS;
static double g_set_ratio = DEFAULT_SET_RATIO;
static hash_bucket_t g_hash[HASH_BUCKETS];
static lru_cache_t g_lru;
static volatile int g_running = 1;

/* Simple hash function (djb2) */
static inline uint32_t hash_key(const char *key) {
    uint32_t hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_BUCKETS;
}

/* Generate random key */
static void gen_key(char *buf, int n) {
    for (int i = 0; i < n - 1; i++) {
        buf[i] = 'a' + (rand() % 26);
    }
    buf[n - 1] = '\0';
}

/* Initialize KV store */
static void kv_init(void) {
    g_entries = calloc(g_nkeys, sizeof(kv_entry_t));

    for (int i = 0; i < HASH_BUCKETS; i++) {
        g_hash[i].head = NULL;
        pthread_mutex_init(&g_hash[i].lock, NULL);
    }

    g_lru.head = g_lru.tail = NULL;
    g_lru.count = 0;
    pthread_mutex_init(&g_lru.lock, NULL);

    /* Pre-populate some keys */
    for (int i = 0; i < g_nkeys; i++) {
        gen_key(g_entries[i].key, MAX_KEY_LEN);
        g_entries[i].value_len = MIN_VALUE_LEN + (rand() % (MAX_VALUE_LEN - MIN_VALUE_LEN));
        g_entries[i].value = malloc(g_entries[i].value_len);
        memset(g_entries[i].value, 'v', g_entries[i].value_len);
        g_entries[i].expire_time = 0;  /* No expiration initially */
        g_entries[i].hash_next = NULL;
        g_entries[i].lru_prev = NULL;
        g_entries[i].lru_next = NULL;

        /* Insert into hash table */
        uint32_t bucket = hash_key(g_entries[i].key);
        g_entries[i].hash_next = g_hash[bucket].head;
        g_hash[bucket].head = &g_entries[i];
    }
}

/* Move entry to LRU head (most recently used) */
static void lru_touch(kv_entry_t *entry) {
    pthread_mutex_lock(&g_lru.lock);

    /* Remove from current position */
    if (entry->lru_prev) {
        entry->lru_prev->lru_next = entry->lru_next;
    }
    if (entry->lru_next) {
        entry->lru_next->lru_prev = entry->lru_prev;
    }
    if (entry == g_lru.tail) {
        g_lru.tail = entry->lru_prev;
    }

    /* Insert at head */
    entry->lru_prev = NULL;
    entry->lru_next = g_lru.head;
    if (g_lru.head) {
        g_lru.head->lru_prev = entry;
    }
    g_lru.head = entry;

    if (!g_lru.tail) {
        g_lru.tail = entry;
    }

    if (g_lru.count < LRU_SIZE) {
        g_lru.count++;
    }

    pthread_mutex_unlock(&g_lru.lock);
}

/* Evict LRU tail */
static kv_entry_t *lru_evict(void) {
    pthread_mutex_lock(&g_lru.lock);

    kv_entry_t *victim = g_lru.tail;
    if (victim) {
        if (victim->lru_prev) {
            victim->lru_prev->lru_next = NULL;
        }
        g_lru.tail = victim->lru_prev;
        victim->lru_prev = NULL;
        victim->lru_next = NULL;
        g_lru.count--;
    }

    pthread_mutex_unlock(&g_lru.lock);
    return victim;
}

/* GET operation */
static int kv_get(const char *key, worker_stats_t *stats) {
    uint32_t bucket = hash_key(key);

    pthread_mutex_lock(&g_hash[bucket].lock);

    kv_entry_t *entry = g_hash[bucket].head;
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            /* Check expiration */
            if (entry->expire_time > 0 && entry->expire_time < time(NULL)) {
                pthread_mutex_unlock(&g_hash[bucket].lock);
                stats->expired++;
                stats->misses++;
                return 0;
            }

            /* Read value (simulate cache access) */
            volatile char sum = 0;
            for (size_t i = 0; i < entry->value_len; i++) {
                sum += entry->value[i];
            }

            lru_touch(entry);
            stats->hits++;
            pthread_mutex_unlock(&g_hash[bucket].lock);
            return 1;
        }
        entry = entry->hash_next;
    }

    stats->misses++;
    pthread_mutex_unlock(&g_hash[bucket].lock);
    return 0;
}

/* SET operation */
static void kv_set(const char *key, worker_stats_t *stats) {
    uint32_t bucket = hash_key(key);

    pthread_mutex_lock(&g_hash[bucket].lock);

    /* Find existing or create new */
    kv_entry_t *entry = g_hash[bucket].head;
    while (entry && strcmp(entry->key, key) != 0) {
        entry = entry->hash_next;
    }

    if (!entry) {
        /* Create new entry (from pre-allocated pool) */
        int idx = rand() % g_nkeys;
        entry = &g_entries[idx];
        strncpy(entry->key, key, MAX_KEY_LEN - 1);

        /* Insert into hash chain */
        entry->hash_next = g_hash[bucket].head;
        g_hash[bucket].head = entry;
    }

    /* Update value */
    entry->value_len = MIN_VALUE_LEN + (rand() % (MAX_VALUE_LEN - MIN_VALUE_LEN));
    memset(entry->value, 'v', entry->value_len);

    /* Set expiration (10% chance to have TTL) */
    if (rand() % 10 == 0) {
        entry->expire_time = time(NULL) + (rand() % 300);
    } else {
        entry->expire_time = 0;
    }

    lru_touch(entry);

    /* Simulate eviction if LRU is full */
    if (g_lru.count > LRU_SIZE) {
        kv_entry_t *victim = lru_evict();
        if (victim) {
            stats->evictions++;
        }
    }

    pthread_mutex_unlock(&g_hash[bucket].lock);
}

/* Simulate a KV operation */
static void do_operation(worker_stats_t *stats) {
    /* Choose random key */
    int idx = rand() % g_nkeys;
    char key[MAX_KEY_LEN];
    strncpy(key, g_entries[idx].key, MAX_KEY_LEN);

    /* Decide GET or SET based on ratio */
    if ((double)rand() / RAND_MAX < g_set_ratio) {
        kv_set(key, stats);
        stats->sets++;
    } else {
        kv_get(key, stats);
        stats->gets++;
    }

    stats->operations++;
}

static void *worker(void *arg) {
    worker_stats_t *stats = (worker_stats_t *)arg;

    while (g_running) {
        do_operation(stats);
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
    int nkeys = DEFAULT_KEYS;
    double set_ratio = DEFAULT_SET_RATIO;

    static struct option long_opts[] = {
        {"threads",    required_argument, 0, 't'},
        {"duration",   required_argument, 0, 'd'},
        {"keys",       required_argument, 0, 'k'},
        {"ratio",      required_argument, 0, 'r'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:k:r:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
            case 'k': nkeys = atoi(optarg); break;
            case 'r': set_ratio = atof(optarg); break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;
    if (nkeys < 1000) nkeys = 1000;
    if (set_ratio < 0.0) set_ratio = 0.0;
    if (set_ratio > 1.0) set_ratio = 1.0;

    g_nkeys = nkeys;
    g_set_ratio = set_ratio;

    /* Initialize KV store */
    kv_init();

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
    long total_ops = 0, total_gets = 0, total_sets = 0;
    long total_hits = 0, total_misses = 0, total_evictions = 0, total_expired = 0;

    for (int i = 0; i < threads; i++) {
        total_ops += stats[i].operations;
        total_gets += stats[i].gets;
        total_sets += stats[i].sets;
        total_hits += stats[i].hits;
        total_misses += stats[i].misses;
        total_evictions += stats[i].evictions;
        total_expired += stats[i].expired;
    }

    double ops_per_sec = (double)total_ops / duration;
    double hit_rate = total_gets > 0 ? (double)total_hits / total_gets * 100.0 : 0;

    printf("OPS: %.2f ops/sec\n", ops_per_sec);
    printf("GET/SET ratio: %ld/%ld\n", total_gets, total_sets);
    printf("Hit rate: %.2f%%\n", hit_rate);
    printf("Evictions: %ld\n", total_evictions);
    printf("Expired: %ld\n", total_expired);

    /* Cleanup */
    for (int i = 0; i < g_nkeys; i++) {
        free(g_entries[i].value);
    }
    for (int i = 0; i < HASH_BUCKETS; i++) {
        pthread_mutex_destroy(&g_hash[i].lock);
    }
    pthread_mutex_destroy(&g_lru.lock);
    free(g_entries);
    free(stats);
    free(tids);

    return 0;
}
