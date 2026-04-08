# ArkProbe 方舟探针

> 面向鲲鹏芯片设计空间探索的统一负载特征分析框架

ArkProbe 是一个针对 ARM 鲲鹏处理器（920/930）的负载特征建模与硬件设计空间探索工具。它通过 perf + eBPF 混合采集，将数据库、大数据、编解码、搜推广、微服务等典型场景统一建模为特征向量，为芯片架构师提供设计参数敏感度分析和优化建议。

## 核心能力

```
场景YAML → 数据采集(perf/eBPF/system) → 统一特征向量 → 分析引擎 → HTML报告
```

- **统一特征建模** — 8个维度（计算/Cache/分支/内存/IO/网络/并发/扩展性）的 Pydantic 模型，跨场景可比
- **鲲鹏专用采集** — ARM PMU 事件精确分组（6计数器约束），支持 Uncore PMU（DDR带宽/L3 cache）
- **TopDown 瓶颈分析** — ARM TopDown L1/L2 方法论，自动识别 Frontend/Backend/BadSpec 瓶颈
- **设计空间探索** — 13个硬件参数（cache/BW/发射宽度/ROB/BTB/SIMD等）的敏感度评分与跨场景矩阵
- **交互式报告** — Plotly 图表的自包含 HTML，含执行摘要、场景深度分析、跨场景对比、设计建议

## 快速开始

### 安装

```bash
git clone https://github.com/xuefenghao5121/ArkProbe.git
cd ArkProbe
pip install -e ".[dev]"
```

### 基本用法

```bash
# 查看预置场景
arkprobe list

# 在鲲鹏服务器上采集（需要 root 权限执行 eBPF）
arkprobe collect -s database_oltp -t 60

# 分析采集数据，提取特征向量
arkprobe analyze -i ./data

# 生成 HTML 报告
arkprobe report -f data/*_features.json -o report.html

# 查看设计参数敏感度矩阵
arkprobe sensitivity -f data/*_features.json

# 一键全流程：采集 + 分析 + 报告
arkprobe full-run -s all -o report.html
```

### 跳过 eBPF（非 root / 非鲲鹏环境）

```bash
arkprobe collect -s database_oltp --skip-ebpf
```

## 预置场景

| 场景 | 类型 | 典型负载 | 关键关注指标 |
|------|------|---------|-------------|
| MySQL OLTP | database_oltp | sysbench oltp_read_write | L3 MPKI, 锁竞争, IO延迟 |
| PostgreSQL OLAP | database_olap | TPC-H | 内存带宽, SIMD利用率 |
| Redis | database_kv | redis-benchmark | L1D MPKI, 网络延迟 |
| Spark Batch | bigdata_batch | TPC-DS / WordCount | 内存带宽, L3 MPKI |
| Spark Streaming | bigdata_streaming | Structured Streaming | 网络吞吐, 上下文切换 |
| Hadoop MR | bigdata_batch | WordCount | IO吞吐, 磁盘IOPS |
| H.264 编码 | codec_video | ffmpeg libx264 | SIMD利用率, IPC |
| H.265 编码 | codec_video | ffmpeg libx265 | SIMD, 核扩展性 |
| AV1 编码 | codec_video | SVT-AV1 | SIMD, 发射宽度 |
| 搜索推荐 | search_recommend | Elasticsearch | 分支MPKI, L3 MPKI |
| API 网关 | microservice | Nginx + wrk | 网络PPS, TCP延迟 |
| 服务网格 | microservice | Envoy sidecar | 网络延迟, 锁竞争 |

添加自定义场景：复制 `arkprobe/scenarios/templates/scenario_template.yaml` 到 `configs/` 目录并填入配置。

## 统一特征向量

所有场景统一建模为 `WorkloadFeatureVector`，核心维度：

```
ComputeCharacteristics    — IPC, CPI, 指令mix, SIMD利用率, TopDown L1/L2
CacheHierarchy            — L1I/L1D/L2/L3 MPKI & miss rate, 局部性评分
BranchBehavior            — MPKI, 误预测率, 间接分支占比
MemorySubsystem           — 读写带宽, 利用率, 延迟, NUMA本地比, TLB MPKI
IOCharacteristics         — IOPS, 吞吐, 延迟直方图, 读写比
NetworkCharacteristics    — PPS, 带宽, TCP延迟, 连接速率
ConcurrencyProfile        — 线程数, 上下文切换, 锁竞争, futex等待
ScalabilityProfile        — 核数-吞吐曲线, Amdahl串行分数, 最优核数
```

## 设计空间探索

ArkProbe 的核心交付物是**场景 × 硬件参数**的敏感度矩阵：

| 参数 | 评分依据 |
|------|---------|
| L1D/L1I/L2/L3 Cache Size | 对应级别 MPKI + 局部性 |
| L3 Associativity | miss rate vs MPKI (conflict miss 判定) |
| 内存带宽 / 通道数 | 带宽利用率 + backend_memory_bound |
| 发射宽度 | IPC/当前宽度 + retiring fraction |
| ROB 深度 | backend_bound × L3 MPKI |
| BTB 大小 | branch MPKI + 间接分支占比 |
| Prefetch 激进度 | 空间局部性 × L2 MPKI |
| 核数 | Amdahl 串行分数 / 扩展效率 |
| SIMD 宽度 | SIMD 利用率 + vector 指令占比 |

输出示例（`arkprobe sensitivity`）：

```
Top Design Recommendations:
  1. btb_entries     (priority=0.370, cost=low)
  2. l1d_cache_size  (priority=0.294, cost=medium)
  3. core_count      (priority=0.253, cost=high)
```

## 项目结构

```
arkprobe/
├── model/           # 统一特征向量 Pydantic 模型
├── collectors/      # perf / eBPF / system 数据采集器
├── scenarios/       # 12个预置场景 YAML + 加载器
├── analysis/        # 特征提取 / TopDown / 对比 / 扩展性 / 设计空间
├── reports/         # Plotly 图表 + HTML 报告生成
├── utils/           # 平台检测 / 子进程 / 单位换算
└── cli.py           # Click CLI 入口
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest -v

# 代码检查
ruff check arkprobe/ tests/
```

## 依赖

- Python >= 3.10
- Linux perf（需 `perf` 命令可用）
- BCC tools / bpftrace（eBPF 采集，可选）
- 主要 Python 包：click, pydantic, pandas, numpy, scipy, plotly, scikit-learn, jinja2, rich

## 目标平台

| 型号 | 核心 | 发射宽度 | PMU 计数器 | SIMD |
|------|------|---------|-----------|------|
| 鲲鹏 920 | TSV110 | 4-wide | 6 + 1 fixed | NEON 128-bit |
| 鲲鹏 930 | TaiShan V200 | 8-wide | 6 + 1 fixed | NEON + SVE |

## License

MIT
