/*
 * arkprobe_webserver — Web Server micro-kernel workload.
 *
 * Models key characteristics of high-performance web servers (Nginx/Envoy style):
 *   1. Connection handling: Accept/dispatch pattern with event loops
 *   2. HTTP parsing: Header parsing with string operations
 *   3. Routing: Hash-based URL routing with pattern matching
 *   4. Response generation: Template rendering with variable substitution
 *
 * This workload produces:
 *   - High branch misprediction (parsing, routing)
 *   - String operations (memcpy, strcmp)
 *   - Random memory access (hash table lookups)
 *   - Mixed compute/memory patterns
 *
 * Usage: arkprobe_webserver --threads N --duration S [--connections C]
 * Output: prints RPS (requests/sec) and latency statistics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define DEFAULT_CONNECTIONS 10000
#define ROUTE_TABLE_SIZE 1024
#define MAX_HEADER_SIZE 4096
#define MAX_BODY_SIZE 16384
#define MAX_ROUTES 256
#define MAX_ROUTE_LEN 128

/* HTTP method */
typedef enum {
    METHOD_GET,
    METHOD_POST,
    METHOD_PUT,
    METHOD_DELETE,
    METHOD_HEAD,
    METHOD_COUNT
} http_method_t;

/* HTTP request */
typedef struct {
    http_method_t method;
    char path[MAX_ROUTE_LEN];
    char headers[MAX_HEADER_SIZE];
    size_t header_len;
    char body[MAX_BODY_SIZE];
    size_t body_len;
} http_request_t;

/* HTTP response */
typedef struct {
    int status;
    char headers[MAX_HEADER_SIZE];
    size_t header_len;
    char body[MAX_BODY_SIZE];
    size_t body_len;
} http_response_t;

/* Route handler */
typedef void (*route_handler_t)(http_request_t *req, http_response_t *resp);

/* Route entry */
typedef struct route_entry {
    char pattern[MAX_ROUTE_LEN];
    http_method_t method;
    route_handler_t handler;
    struct route_entry *next;
} route_entry_t;

/* Route table */
typedef struct {
    route_entry_t *buckets[ROUTE_TABLE_SIZE];
    pthread_mutex_t locks[ROUTE_TABLE_SIZE];
} route_table_t;

/* Connection state */
typedef struct {
    int id;
    int active;
    http_request_t request;
    http_response_t response;
} connection_t;

/* Worker statistics */
typedef struct {
    long requests;
    long bytes_in;
    long bytes_out;
    long parse_errors;
    long route_misses;
    long latency_ns;
} worker_stats_t;

/* Global state */
static connection_t *g_connections = NULL;
static int g_nconnections = DEFAULT_CONNECTIONS;
static route_table_t g_routes;
static volatile int g_running = 1;

/* Predefined response templates */
static const char *STATUS_TEXT[] = {
    [200] = "OK",
    [201] = "Created",
    [204] = "No Content",
    [301] = "Moved Permanently",
    [304] = "Not Modified",
    [400] = "Bad Request",
    [401] = "Unauthorized",
    [403] = "Forbidden",
    [404] = "Not Found",
    [405] = "Method Not Allowed",
    [500] = "Internal Server Error",
    [502] = "Bad Gateway",
    [503] = "Service Unavailable",
};

static const char *HTTP_METHOD_STR[] = {
    [METHOD_GET] = "GET",
    [METHOD_POST] = "POST",
    [METHOD_PUT] = "PUT",
    [METHOD_DELETE] = "DELETE",
    [METHOD_HEAD] = "HEAD",
};

/* Hash function for routes */
static inline uint32_t hash_route(const char *path) {
    uint32_t hash = 5381;
    int c;
    while ((c = *path++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % ROUTE_TABLE_SIZE;
}

/* Route handlers */
static void handle_api_json(http_request_t *req, http_response_t *resp) {
    resp->status = 200;
    const char *json = "{\"status\":\"ok\",\"data\":{\"value\":42}}";
    resp->body_len = strlen(json);
    memcpy(resp->body, json, resp->body_len);
}

static void handle_api_echo(http_request_t *req, http_response_t *resp) {
    resp->status = 200;
    resp->body_len = req->body_len < MAX_BODY_SIZE ? req->body_len : MAX_BODY_SIZE;
    memcpy(resp->body, req->body, resp->body_len);
}

static void handle_static(http_request_t *req, http_response_t *resp) {
    resp->status = 200;
    /* Simulate static file serving */
    memset(resp->body, 'X', 4096);
    resp->body_len = 4096;
}

static void handle_health(http_request_t *req, http_response_t *resp) {
    resp->status = 200;
    const char *health = "OK";
    resp->body_len = 2;
    memcpy(resp->body, health, resp->body_len);
}

static void handle_redirect(http_request_t *req, http_response_t *resp) {
    resp->status = 301;
    const char *loc = "/new-location";
    resp->body_len = strlen(loc);
    memcpy(resp->body, loc, resp->body_len);
}

static void handle_not_found(http_request_t *req, http_response_t *resp) {
    resp->status = 404;
    const char *msg = "Not Found";
    resp->body_len = 9;
    memcpy(resp->body, msg, resp->body_len);
}

/* Register route */
static void route_register(const char *pattern, http_method_t method, route_handler_t handler) {
    uint32_t bucket = hash_route(pattern);

    route_entry_t *entry = malloc(sizeof(route_entry_t));
    strncpy(entry->pattern, pattern, MAX_ROUTE_LEN - 1);
    entry->method = method;
    entry->handler = handler;
    entry->next = g_routes.buckets[bucket];
    g_routes.buckets[bucket] = entry;
}

/* Initialize routes */
static void routes_init(void) {
    for (int i = 0; i < ROUTE_TABLE_SIZE; i++) {
        g_routes.buckets[i] = NULL;
        pthread_mutex_init(&g_routes.locks[i], NULL);
    }

    /* Register common routes */
    route_register("/api/json", METHOD_GET, handle_api_json);
    route_register("/api/echo", METHOD_POST, handle_api_echo);
    route_register("/static/", METHOD_GET, handle_static);
    route_register("/health", METHOD_GET, handle_health);
    route_register("/redirect", METHOD_GET, handle_redirect);

    /* Add more routes for realistic routing table */
    for (int i = 0; i < 50; i++) {
        char pattern[64];
        snprintf(pattern, sizeof(pattern), "/api/v%d/resource", i % 10);
        route_register(pattern, METHOD_GET, handle_api_json);
    }
}

/* Parse HTTP request (simplified) */
static int parse_request(const char *raw, size_t len, http_request_t *req) {
    /* Parse method */
    if (strncmp(raw, "GET ", 4) == 0) {
        req->method = METHOD_GET;
        raw += 4;
    } else if (strncmp(raw, "POST ", 5) == 0) {
        req->method = METHOD_POST;
        raw += 5;
    } else if (strncmp(raw, "PUT ", 4) == 0) {
        req->method = METHOD_PUT;
        raw += 4;
    } else if (strncmp(raw, "DELETE ", 7) == 0) {
        req->method = METHOD_DELETE;
        raw += 7;
    } else if (strncmp(raw, "HEAD ", 5) == 0) {
        req->method = METHOD_HEAD;
        raw += 5;
    } else {
        return -1;
    }

    /* Parse path */
    char *space = strchr(raw, ' ');
    if (!space) return -1;

    size_t path_len = space - raw;
    if (path_len >= MAX_ROUTE_LEN) path_len = MAX_ROUTE_LEN - 1;
    strncpy(req->path, raw, path_len);
    req->path[path_len] = '\0';

    /* Skip HTTP version line */
    char *eol = strchr(space, '\n');
    if (!eol) return -1;

    /* Parse headers (simplified - just copy) */
    char *body_start = strstr(eol, "\r\n\r\n");
    if (body_start) {
        req->header_len = body_start - eol;
        if (req->header_len >= MAX_HEADER_SIZE) req->header_len = MAX_HEADER_SIZE - 1;
        memcpy(req->headers, eol, req->header_len);

        body_start += 4;
        req->body_len = len - (body_start - raw);
        if (req->body_len >= MAX_BODY_SIZE) req->body_len = MAX_BODY_SIZE - 1;
        memcpy(req->body, body_start, req->body_len);
    }

    return 0;
}

/* Build response headers */
static void build_response_headers(http_response_t *resp) {
    const char *status_text = STATUS_TEXT[resp->status] ? STATUS_TEXT[resp->status] : "Unknown";

    resp->header_len = snprintf(resp->headers, MAX_HEADER_SIZE,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: keep-alive\r\n"
        "Server: arkprobe_webserver\r\n"
        "\r\n",
        resp->status, status_text, resp->body_len);
}

/* Route lookup */
static route_entry_t *route_lookup(const char *path, http_method_t method) {
    uint32_t bucket = hash_route(path);

    pthread_mutex_lock(&g_routes.locks[bucket]);

    route_entry_t *entry = g_routes.buckets[bucket];
    while (entry) {
        /* Simple prefix match for static routes */
        if (strncmp(entry->pattern, path, strlen(entry->pattern)) == 0) {
            if (entry->method == method) {
                pthread_mutex_unlock(&g_routes.locks[bucket]);
                return entry;
            }
        }
        /* Exact match */
        if (strcmp(entry->pattern, path) == 0 && entry->method == method) {
            pthread_mutex_unlock(&g_routes.locks[bucket]);
            return entry;
        }
        entry = entry->next;
    }

    pthread_mutex_unlock(&g_routes.locks[bucket]);
    return NULL;
}

/* Generate random request */
static void gen_random_request(char *buf, size_t *len) {
    const char *paths[] = {
        "/api/json",
        "/api/echo",
        "/static/index.html",
        "/health",
        "/redirect",
        "/api/v1/users",
        "/api/v2/orders",
        "/api/v3/products",
        "/unknown/path",
        "/another/random/path"
    };
    const char *methods[] = {"GET", "POST", "PUT", "DELETE", "HEAD"};
    const char *headers = "Host: localhost\r\nUser-Agent: arkprobe/1.0\r\nAccept: */*\r\n";

    int path_idx = rand() % 10;
    int method_idx = rand() % 5;

    *len = snprintf(buf, MAX_HEADER_SIZE + MAX_BODY_SIZE,
        "%s %s HTTP/1.1\r\n%s\r\n{\"test\":\"data\"}\r\n",
        methods[method_idx], paths[path_idx], headers);
}

/* Process a request */
static void process_request(worker_stats_t *stats) {
    struct timespec ts1, ts2;
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* Get a connection */
    int conn_id = rand() % g_nconnections;
    connection_t *conn = &g_connections[conn_id];

    /* Generate request */
    char raw_request[MAX_HEADER_SIZE + MAX_BODY_SIZE];
    size_t raw_len;
    gen_random_request(raw_request, &raw_len);
    stats->bytes_in += raw_len;

    /* Parse request */
    if (parse_request(raw_request, raw_len, &conn->request) != 0) {
        stats->parse_errors++;
        return;
    }

    /* Route lookup */
    route_entry_t *route = route_lookup(conn->request.path, conn->request.method);

    if (route) {
        route->handler(&conn->request, &conn->response);
    } else {
        handle_not_found(&conn->request, &conn->response);
        stats->route_misses++;
    }

    /* Build response */
    build_response_headers(&conn->response);
    stats->bytes_out += conn->response.header_len + conn->response.body_len;

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    long latency = (ts2.tv_sec - ts1.tv_sec) * 1000000000L +
                   (ts2.tv_nsec - ts1.tv_nsec);
    stats->latency_ns += latency;
    stats->requests++;
}

static void *worker(void *arg) {
    worker_stats_t *stats = (worker_stats_t *)arg;

    while (g_running) {
        process_request(stats);
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
    int nconnections = DEFAULT_CONNECTIONS;

    static struct option long_opts[] = {
        {"threads",     required_argument, 0, 't'},
        {"duration",    required_argument, 0, 'd'},
        {"connections", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:c:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
            case 'c': nconnections = atoi(optarg); break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;
    if (nconnections < 100) nconnections = 100;

    g_nconnections = nconnections;

    /* Initialize connections */
    g_connections = calloc(g_nconnections, sizeof(connection_t));
    for (int i = 0; i < g_nconnections; i++) {
        g_connections[i].id = i;
        g_connections[i].active = 0;
    }

    /* Initialize routes */
    routes_init();

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
    long total_reqs = 0, total_bytes_in = 0, total_bytes_out = 0;
    long total_errors = 0, total_misses = 0, total_latency_ns = 0;

    for (int i = 0; i < threads; i++) {
        total_reqs += stats[i].requests;
        total_bytes_in += stats[i].bytes_in;
        total_bytes_out += stats[i].bytes_out;
        total_errors += stats[i].parse_errors;
        total_misses += stats[i].route_misses;
        total_latency_ns += stats[i].latency_ns;
    }

    double rps = (double)total_reqs / duration;
    double avg_latency_us = total_reqs > 0 ? (double)total_latency_ns / total_reqs / 1000.0 : 0;
    double throughput_mbps = (double)(total_bytes_in + total_bytes_out) / duration / (1024 * 1024);

    printf("RPS: %.2f req/sec\n", rps);
    printf("Avg latency: %.2f us\n", avg_latency_us);
    printf("Throughput: %.2f MB/s\n", throughput_mbps);
    printf("Parse errors: %ld\n", total_errors);
    printf("Route misses: %ld\n", total_misses);

    /* Cleanup */
    for (int i = 0; i < ROUTE_TABLE_SIZE; i++) {
        route_entry_t *entry = g_routes.buckets[i];
        while (entry) {
            route_entry_t *next = entry->next;
            free(entry);
            entry = next;
        }
        pthread_mutex_destroy(&g_routes.locks[i]);
    }
    free(g_connections);
    free(stats);
    free(tids);

    return 0;
}
