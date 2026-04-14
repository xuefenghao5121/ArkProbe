# ArkProbe 方舟探针

> 面向鲲鹏芯片设计空间探索的统一负载特征分析框架

ArkProbe 是一个针对 ARM 鲲鹏处理器（920/930）的负载特征建模与硬件设计空间探索工具。它通过 perf + eBPF 混合采集，将数据库、大数据、编解码、搜推广、微服务等典型场景统一建模为特征向量，为芯片架构师提供设计参数敏感度分析和优化建议。

## 核心能力

```
场景YAML → 数据采集(perf/eBPF/system) → 统一特征向量 → 分析引擎 → HTML报告
```

- **统一特征建模** — 9个维度（计算/Cache/分支/内存/IO/网络/并发/功耗温度/扩展性）的 Pydantic 模型，跨场景可比
- **鲲鹏专用采集** — ARM PMU 事件精确分组（6计数器约束），支持 Uncore PMU（DDR带宽/L3 cache）
- **TopDown 瓶颈分析** — ARM TopDown L1/L2 方法论，自动识别 Frontend/Backend/BadSpec 瓶颈
- **设计空间探索** — 13个硬件参数（cache/BW/发射宽度/ROB/BTB/SIMD等）的敏感度评分与跨场景矩阵
- **平台优化建议** — 22条鲲鹏调优规则，覆盖OS（大页/THP/governor/NUMA/IO调度器）、BIOS（SMT/预取/C-states/电源）、驱动（NIC offload/ring buffer/IRQ/文件系统），按场景差异化推荐，输出可执行命令
- **交互式报告** — Plotly 图表的自包含 HTML，含执行摘要、场景深度分析、跨场景对比、设计建议、平台优化
- **硬件调节实验** — 8个预置配置，自动调节 CPU 频率/SMT/C-state/NUMA 并对比性能
- **gem5 微架构仿真** — 探索真实硬件无法调节的参数（Cache 大小/ROB 深度/发射宽度/BTB 大小）

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

# 查看平台优化建议（OS/BIOS/驱动调优）
arkprobe optimize -f data/*_features.json

# 硬件调节实验
arkprobe tune-configs                  # 列出调节配置
arkprobe tune -s compute -c performance -c power

# gem5 微架构仿真（需要安装 gem5）
arkprobe gem5-configs                  # 列出 gem5 配置
arkprobe simulate -s compute -c default -c large_cache

# 一键全流程：采集 + 分析 + 报告
arkprobe full-run -s all -o report.html
```

### 跳过 eBPF（非 root / 非鲲鹏环境）

```bash
arkprobe collect -s database_oltp --skip-ebpf
```

## 预置场景

### 内置场景（零依赖）

| 场景 | 类型 | 描述 | 关键关注指标 |
|------|------|------|-------------|
| Compute Intensive | compute_bound | 矩阵乘法，高 IPC | IPC, retiring, SIMD |
| Memory Intensive | memory_bound | 流式拷贝 + 指针追踪 | L3 MPKI, backend_bound |
| Mixed Workload | mixed | 计算+访存交替 | IPC, cache miss rate |
| STREAM Benchmark | memory_bound | 标准 COPY/SCALE/ADD/TRIAD | 内存带宽利用率 |
| Random Access | memory_bound | 随机指针追踪 | 内存延迟, TLB MPKI |
| Database OLTP | database_oltp | 锁竞争/B+tree/WAL 微内核 | 锁等待, L3 MPKI, branch MPKI |
| Key-Value Store | database_kv | 哈希表/LRU/GET/SET 微内核 | L1D MPKI, TLB MPKI |
| Web Server | microservice | HTTP解析/路由/响应 微内核 | branch MPKI, IPC |
| Cryptographic Ops | compute_bound | AES加密/SHA-256哈希 微内核 | IPC, retiring, SIMD |
| Compression | mixed | LZ77/哈希链/匹配编码 微内核 | branch MPKI, L3 MPKI |
| Video Encoding | compute_bound | DCT/运动估计/量化 微内核 | FP ratio, SIMD, IPC |
| ML Inference | compute_bound | GEMM/卷积/激活函数 微内核 | IPC, FP ratio, retiring |

### 外部场景（需依赖工具）

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
PowerThermal              — CPU/DRAM功耗, 温度, C-state停留分布, 频率统计
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

## 平台优化建议

ArkProbe 自动采集当前平台配置，与场景最优配置对比，输出可执行的调优命令：

| 层级 | 调优项 | 场景差异化示例 |
|------|--------|---------------|
| **OS** | vm.nr_hugepages, THP, vm.swappiness, dirty_ratio, CPU governor, NUMA balancing, IO scheduler, net backlog/somaxconn, sched_granularity | 数据库: swappiness=1, mq-deadline; 编解码: THP=always; 微服务: backlog=65536 |
| **BIOS** | NUMA, HW Prefetcher, SMT, Power Profile, C-states | 数据库: 关SMT降锁竞争; 编解码: 关C-states降延迟抖动 |
| **驱动** | NIC TSO/GRO, ring buffer, irqbalance, noatime, RPS/RFS | 微服务: 最大ring buffer+开RPS; 数据库: 关irqbalance+noatime |

输出示例（`arkprobe optimize`）：

```
MySQL OLTP (50/100)
  OS (5 gaps)
    Parameter   Current  Recommended  Impact  Command
    Swappiness  60       1            70%     sysctl -w vm.swappiness=1
    ...

Universal Recommendations (benefit all workloads):
  1. CPU Frequency Governor → cpupower frequency-set -g performance
  2. NIC Offload (TSO/GRO) → ethtool -K eth0 tso on gro on
```

## 项目结构

```
arkprobe/
├── model/           # 统一特征向量 Pydantic 模型
├── collectors/      # perf / eBPF / system 数据采集器
├── scenarios/       # 12个预置场景 YAML + 加载器
├── analysis/        # 特征提取 / TopDown / 对比 / 扩展性 / 设计空间 / 平台优化
├── tuner/           # 硬件调节 + gem5 仿真
│   ├── hardware_tuner.py  # CPU频率/SMT/C-state/NUMA 调节
│   ├── comparator.py      # 配置对比分析器
│   └── gem5_tuner.py      # gem5 微架构仿真
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

## 更新日志

### v0.2.9 (2026-04-14)

**New Feature: gem5 微架构仿真**
- 新增 `arkprobe simulate` 命令，在 gem5 中仿真不同微架构配置下的负载性能
- 新增 `arkprobe gem5-configs` 命令列出可用配置
- 支持探索真实硬件无法调节的参数：Cache 大小/关联度、ROB 深度、发射宽度、BTB 大小
- 7 个预置配置：default、small_cache、large_cache、wide_issue、deep_rob、big_btb、kunpeng_like
- 自动生成 gem5 Python 配置脚本，解析 stats.txt 输出
- 多配置对比分析：IPC 变化、MPKI 变化、瓶颈迁移

**新增模块**
- `arkprobe/tuner/gem5_tuner.py` — gem5 仿真器
  - `CacheConfig`: Cache 配置（size/assoc/latency/MSHRs）
  - `O3CPUConfig`: O3 CPU 参数（发射宽度/ROB/IQ/LSQ/BTB/RAS）
  - `Gem5Config`: 完整仿真配置
  - `Gem5Stats`: 解析仿真统计
  - `Gem5Tuner`: 生成配置脚本、运行仿真、解析结果
- 25 个新增单元测试

**gem5 支持的微架构参数**
- Cache 层级：L1I/L1D/L2/L3 的 size、associativity、latency、MSHRs
- CPU 流水线：fetch/decode/rename/issue/wb/commit 宽度
- Buffer 大小：ROB/IQ/LSQ entries
- 分支预测：BTB/RAS 大小
- 内存子系统：DDR 类型、频率、容量

### v0.2.8 (2026-04-13)

**New Feature: 真实硬件调节能力**
- 新增 `arkprobe tune` 命令，支持在不同硬件配置下运行负载并对比性能
- 可调节参数：CPU 频率/governor、SMT 开关、C-state 限制、NUMA 策略、CPU 亲和性、THP 设置
- 8 个预置配置：default、performance、performance_no_smt、latency、power、database、compute、memory
- 新增 `arkprobe tune-configs` 命令列出可用配置
- 自动对比分析：IPC 变化、MPKI 变化、瓶颈迁移、整体改进评分

**新增模块**
- `arkprobe/tuner/hardware_tuner.py` — 硬件参数调节器
- `arkprobe/tuner/comparator.py` — 配置对比分析器
- 25 个新增单元测试

### v0.2.7 (2026-04-13)

**New Feature: 4 个新内置负载**
- 新增加密微内核（`crypto.c`）：AES 加密、SHA-256 哈希、S-Box 查找
- 新增压缩微内核（`compress.c`）：LZ77 风格、哈希链匹配、滑动窗口
- 新增视频编码微内核（`videoenc.c`）：DCT 变换、运动估计、帧内预测
- 新增 ML 推理微内核（`ml_inference.c`）：GEMM、卷积、激活函数、池化
- 内置场景从 8 个扩展到 12 个

**报告展示优化**
- 改进瓶颈分析器分类准确性（Memory-Bound vs Core-Bound）
- 执行摘要新增 L3 MPKI、内存带宽列和 Quick Insights
- 图表新增严重程度颜色编码和阈值参考线
- 场景深度分析新增瓶颈严重程度可视化

### v0.2.6 (2026-04-13)

**New Feature: 真实场景微内核负载**
- 新增 Database OLTP 微内核（`oltp.c`）：模拟锁竞争、B+tree 遍历、WAL 写入
- 新增 Key-Value Store 微内核（`kvstore.c`）：模拟哈希表查找、LRU 淘汰、GET/SET 操作
- 新增 Web Server 微内核（`webserver.c`）：模拟 HTTP 解析、路由匹配、响应生成
- 内置场景从 5 个扩展到 8 个

**微内核设计特点**
- 基于真实负载特征设计，提取关键行为模式
- 锁竞争模拟：Row-level mutex + lock hash table
- B+tree 遍历：随机树遍历，cache-unfriendly 访问
- 哈希表查找：随机访问 + 高碰撞率
- HTTP 解析：Header 解析 + 字符串操作
- 路由匹配：Hash-based URL 路由

### v0.2.5 (2026-04-13)

**New Feature: Uncore PMU 采集**
- 新增 `PerfCollector.collect_uncore()` 方法采集 DDR 控制器带宽和 L3 uncore 统计
- 使用鲲鹏 Uncore PMU 事件（hisi_sccl_ddrc/hisi_sccl_l3c）获取实际内存带宽数据
- `MemorySubsystem.bandwidth_read_gbps/write_gbps/utilization` 现已填充实际测量值

**New Feature: 内置负载扩展**
- 新增 STREAM 基准测试（COPY/SCALE/ADD/TRIAD）用于标准内存带宽测量
- 新增随机访问负载用于内存延迟测量（指针追踪）
- 内置场景从 3 个扩展到 5 个

**采集流程优化**
- 采集流程新增 Phase 4: Uncore PMU 采集
- Uncore 采集为可选，失败不影响整体采集

### v0.2.4 (2026-04-11)

**New Feature: 功耗温度分析维度**
- 新增 `PowerThermal` 特征维度：功耗、温度、C-state 停留分布、频率统计
- 数据源：`/sys/class/hwmon`（功耗/温度）、`/sys/class/thermal`（温度）、`/sys/devices/system/cpu/cpu*/cpuidle`（C-state）、`/sys/devices/system/cpu/cpu*/cpufreq`（频率）
- 统一特征向量扩展为 9 个维度

**Security Fix**
- 修复 `perf_collector.py` 命令注入漏洞，新增 `validate_command_safety()` 函数检测危险 shell 元字符

### v0.2.0 (2026-04-08)

**New Feature: 平台优化建议**
- 新增 `arkprobe optimize` 命令，输出 OS/BIOS/驱动调优建议
- 22 条鲲鹏调优规则知识库，按场景差异化推荐（数据库 vs 编解码 vs 微服务等）
- 自动采集当前 OS/BIOS/驱动配置，与推荐配置对比生成 gap 分析
- 跨场景通用建议 + 冲突参数检测
- HTML 报告新增第 5 节：优化得分、gap 分析表、受益矩阵热力图、可复制调优脚本
- 统一特征向量新增 `platform_config` 字段（向后兼容）

### v0.1.1 (2026-04-08)

**Bug Fixes**
- 修复 `perf stat`/`perf record` 中 `sleep` 命令前缺少 `--` 分隔符导致参数解析错误
- 修复 eBPF bpftrace 临时文件泄漏（`delete=False` 未清理）
- 修复 P99 延迟计算索引越界风险
- 修复 `collector_orchestrator` 中 `dict.get()` 无默认值导致 `None` 参与算术运算
- 修复磁盘设备正则匹配不完整（`sd|vd` → `sd[a-z]|vd[a-z]`）
- 修复设计空间分析中枚举比较使用字符串而非 `AccessPattern.RANDOM`
- 修复 Amdahl 加速比归一化公式多乘了 `base_cores`
- 修复 `futex_wait_time` 检查将 `0` 误判为 falsy
- 修复瓶颈分析中硬编码 `dispatch_width=4`，改为可配置参数
- 修复 PCA 图 legend 去重逻辑对重复 dict 失效
- 修复带宽-延迟散点图中 `latency=0` 被 `or` 运算符误判

### v0.1.0 (2026-04-08)

- 初始发布：统一特征建模框架 + 12 个预置场景 + 完整分析引擎 + HTML 报告

## License

MIT
