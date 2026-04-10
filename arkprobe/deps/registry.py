"""Centralized mapping from binary names to install hints."""

INSTALL_HINTS: dict[str, str] = {
    # Database
    "sysbench": "yum install sysbench -y",
    "mysqld": "yum install mysql-server -y",
    "redis-server": "yum install redis -y",
    "redis-benchmark": "yum install redis -y",
    "pgbench": "yum install postgresql-contrib -y",
    "psql": "yum install postgresql -y",
    # Big Data
    "spark-submit": "See https://spark.apache.org/downloads.html",
    "hdfs": "See https://hadoop.apache.org/releases.html",
    # Codec
    "ffmpeg": "yum install ffmpeg -y",
    # Microservice
    "nginx": "yum install nginx -y",
    "wrk": "yum install wrk -y  # or build from https://github.com/wg/wrk",
    "envoy": "See https://www.envoyproxy.io/docs/envoy/latest/start/install",
    "nighthawk_client": "See https://github.com/envoyproxy/nighthawk",
    "docker-compose": "yum install docker-compose -y",
    # Search
    "elasticsearch": "See https://www.elastic.co/downloads/elasticsearch",
    # System tools
    "perf": "yum install perf -y",
    "bpftrace": "yum install bpftrace -y",
    # BCC tools (io_latency, lock_contention, offcpu, cache_stats, tcp_latency)
    "biolatency-bpfcc": "yum install bcc-tools -y  # or: apt install bpfcc-tools",
    "cachestat-bpfcc": "yum install bcc-tools -y",
    "tcprtt-bpfcc": "yum install bcc-tools -y",
    "tcpconnlat-bpfcc": "yum install bcc-tools -y",
    "offcputime-bpfcc": "yum install bcc-tools -y",
    "gcc": "yum install gcc -y",
}


def get_install_hint(binary: str) -> str:
    """Return install hint for a binary, or a generic message."""
    return INSTALL_HINTS.get(binary, f"Install '{binary}' using your package manager")
