# ArkProbe 开发计划

## 当前版本
**v0.3.1** (commit: pending)

---

## P0 — 必须解决

### 1. [ ] JVM 分析实机验证
- [ ] 在真实 JVM 应用上跑通 JFR 采集→特征提取→瓶颈检测→调优建议全链路
- [ ] 930 服务器安装 JDK，测试 `benchmarks/jvm_load_test.sh`
- [ ] 验证 GC 算法识别、GC 暂停率、线程数、JIT 编译数等指标准确性

### 2. [ ] 优化规则校准闭环
- [ ] 用户逐个测试 13 个 builtin 场景，反馈优化建议实际效果
- [ ] 迭代修正 base_impact / prerequisites / recommended_values
- [ ] 验证 memory 场景不再推荐 HugePages/THP 等无关参数

---

## P1 — 应尽快解决

### 3. [ ] 930 Uncore PMU 验证
- [ ] DDR 带宽 / l3d uncore 采集在 930 上数据为 0
- [ ] 排查 PMU 事件名是否正确，paranoid 权限是否足够
- [ ] 如无法采集，记录为 known limitation

### 4. [ ] gem5 实机验证
- [ ] gem5_tuner.py 已实现，未在真实 gem5 环境跑通
- [ ] 在有 gem5 的环境中运行仿真测试

### 5. [ ] Hardware Tuner 实机验证
- [ ] hardware_tuner.py 已实现，需 root + 鲲鹏服务器
- [ ] 验证 CPU Governor / C-state / THP 切换是否生效

### 6. [ ] JVM 外部场景增强
- [ ] 更新 bigdata_spark_batch 等含 `target_process=java` 的外部场景
- [ ] 自动启用 JFR 采集

---

## P2 — 重要但非紧急

### 7. [ ] 外部场景端到端验证
- [ ] 12 个外部场景未做完整集成验证
- [ ] 逐个场景运行，验证数据准确性

### 8. [ ] 报告交互增强
- [ ] 更多图表类型（并行坐标图、桑基图等）
- [ ] 筛选/钻取能力
- [ ] 导出功能（PDF、Excel）

### 9. [ ] JDK 8 JMX 回退
- [ ] 当前 JDK 8 用 jstat/jstack
- [ ] JMX (jconsole/JMXConnector) 可提供更丰富数据

---

## P3 — 规划中

### 10. [ ] Windows GUI（Electron + React）
> 状态：Phase 1 完成，Phase 2-3 开发中

#### Phase 1: GUI 基础架构 ✅
- [x] 创建 `arkprobe-gui/` 项目（独立仓库）
- [x] Electron + React + TypeScript 项目架子
- [x] Dashboard 页面：最新采集结果、关键指标卡片、快速入口
- [ ] SSH 隧道管理：连接远程 930 服务器

#### Phase 2: 采集分析页面 🚧
- [x] Collect 页面：场景选择、JVM PID 输入、采集参数配置、实时进度
- [x] Analyze 页面：特征向量展示（TopDown 分解、缓存命中率、JVM 指标）
- [x] Optimize 页面：分层调优建议（OS/BIOS/Driver/JVM）

#### Phase 3: 热点 C++ 化集成 🚧
- [x] Hotspot 页面：热点方法列表、C++ 代码预览、编译/加载状态
- [ ] 集成 ArkProbe Server 的 hotspot 模块

---

### 11. [ ] Java 热点 C++ 化（Velox 风格）
> 状态：Phase 1-2 完成，Phase 3 开发中

#### Phase 1: 热点分析核心 ✅
- [x] HotspotProfiler: JFR `jdk.ExecutionSample` + `jdk.ProfiledMethod` 采集扩展
- [x] BytecodeExtractor: jcmd HSDB / class histogram 字节码提取
- [x] PatternMatcher: 热点方法模式识别（向量表达式优先）

#### Phase 2: C++ 代码生成 ✅
- [x] CppGenerator: Jinja2 模板 → C++ 源码 + JNI wrapper
- [x] vector_expr.cpp.j2 模板：SIMD 向量表达式
- [x] jni_bridge.cpp.j2 模板：JNI 封装
- [x] CMakeLists.txt.j2 模板

#### Phase 3: 编译工具链 ✅
- [x] JFR 热点采集 (jdk.ExecutionSample)
- [x] 字节码提取 (jcmd HSDB)
- [x] PatternMatcher: 模式识别（向量表达式/字符串/数学/访存）
- [x] CppGenerator: Jinja2 模板 → C++ 源码 + JNI wrapper
- [x] UMF 模板 (umf_template.cpp.j2): array_copy / array_scale / array_add / matrix_mul
- [x] Velox 风格模板 (vector_expr.cpp.j2): vector_map / vector_reduce / vector_filter
- [x] Math 模板 (math.cpp.j2): sigmoid / relu / gemm / sin / cos / sqrt
- [x] String 模板 (string.cpp.j2): split / replace / parse / search
- [x] Compiler: GCC/Clang 编译 .so
- [x] JNI Loader: 运行时加载 .so
- [x] BenchmarkRunner: 性能对比

#### Phase 4: 性能验证 ✅
- [x] 集成测试：Java vs C++ 性能对比（目标 3-5x 加速）
- [x] 37 个 hotspot 测试用例通过
- [x] 270 个全量测试通过

#### Bug 修复（2026-04-23）
- [x] CRITICAL: JNI 名称修饰双重 Java_ 前缀 — cpp_generator.py 不再预加 Java_，由模板统一添加
- [x] CRITICAL: 实现完整 JNI 规范修饰（_→_1, ;→_2, [→_3, $→_00024, Unicode→_0xxxx）
- [x] CRITICAL: string.cpp.j2 空 target 无限循环 — find("") 返回 0 导致死循环
- [x] CRITICAL: string.cpp.j2 NewStringUTF 空指针 OOM 崩溃
- [x] CRITICAL: jni_bridge.cpp.j2 RegisterNatives 失败未返回 JNI_ERR
- [x] HIGH: pattern_matcher.py _has_simd_opcodes 搜索 ARM NEON 硬件指令码（Java 字节码里不存在），改为 JVM 字节码模式
- [x] HIGH: estimate_simd_potential 误判 0x0a 为 aload_0（实际是 lload_0），改为搜索 JVM 循环跳转指令
- [x] HIGH: jni_loader.py _detect_jdk_version 异常吞没（except Exception: pass → 捕获具体异常）
- [x] HIGH: jni_loader.py _load_via_agent_jdk8 无论成功失败都返回 True → 检查 result.ok
- [x] HIGH: benchmark_runner.py _invoke_java_method 纯 sleep 占位 → 加 jcmd JFR 尝试
- [x] MEDIUM: accelerator.py target_arch 默认值 "armv8-a+simd" → None（跨平台）
- [x] MEDIUM: CMakeLists.txt.j2 硬编码 stdc++ → 移除（macOS 不需要）
- [x] MEDIUM: string.cpp.j2 target_arch=None 时模板崩溃（"x86" in None → TypeError）
- [x] MEDIUM: 未使用 import 清理（os/shutil/Optional/Any/resolve_jfr_events/HotspotProfile）

---

## 完成项

### v0.3.1 — Java 热点 C++ 化（2026-04-22）
- [x] JFR 热点采集 (jdk.ExecutionSample)
- [x] 字节码提取 (jcmd HSDB / javap)
- [x] PatternMatcher: 模式识别（向量表达式/字符串/数学/访存）
- [x] CppGenerator: 5 个 Jinja2 模板
- [x] UMF 模板: array_copy / array_scale / array_add / matrix_mul
- [x] Velox 模板: vector_map / vector_reduce / vector_filter
- [x] Math 模板: sigmoid / relu / gemm / sin / cos / sqrt
- [x] String 模板: split / replace / parse / search
- [x] Compiler: GCC/Clang 编译 .so
- [x] JNILoader: 运行时加载
- [x] BenchmarkRunner: 性能对比
- [x] 端到端集成测试
- [x] docs/hotspot_acceleration.md 用户指南

### v0.3.0 — JVM 应用分析（2026-04-22）
- [x] JfrCollector（JDK 11+ JFR + JDK 8 jstat/jstack 回退）
- [x] JvmCharacteristics 模型（GCMetrics / JITMetrics / JVMThreadMetrics）
- [x] FeatureExtractor._extract_jvm()
- [x] BottleneckAnalyzer._analyze_jvm_bottlenecks()
- [x] 8 条 JVM 调优规则
- [x] jvm_general 内置场景
- [x] CLI --jfr / --jvm-pid / --jfr-events
- [x] benchmarks/jvm_load_test.sh JVM 测试脚本
- [x] JVM review fixes（10 个问题修复）

### v0.2.9 — gem5 集成（2026-04-14）
- [x] gem5_tuner.py 仿真对接
- [x] 8 个 gem5 预置配置（default / small_cache / large_cache / wide_issue / deep_rob / big_btb / kunpeng_like）
- [x] 微架构参数探索（Cache 层级 / CPU 流水线 / 内存子系统）
- [x] 鲲鹏 930 PMU 兼容性修复

### v0.2.8 — Hardware Tuning（2026-04-10）
- [x] hardware_tuner.py：CPU 频率 / SMT / C-state / NUMA / THP 调节
- [x] 8 个预置配置（default / performance / performance_no_smt / latency / power / database / compute / memory）
- [x] comparator.py 配置对比分析器
- [x] 数据验证工具

### v0.2.7 — 真实场景微内核负载（2026-04-08）
- [x] 4 个新内置负载：crypto / compress / videoenc / ml_inference
- [x] 共 12 个 builtin 场景

### v0.2.6 — OLTP/KV/Web（2026-04-06）
- [x] 真实微内核 workload：OLTP / KV Store / Web Server
- [x] 报告显示优化

### v0.2.5 — Uncore PMU（2026-04-04）
- [x] DDR 带宽 / L3 uncore 事件采集
- [x] STREAM / Random Access 内置 workload

### v0.2.4 — PowerThermal（2026-04-02）
- [x] PowerThermal 分析维度
- [x] perf collector 命令注入修复

### v0.2.2 — 实机验证（2026-03-28）
- [x] 零依赖内置负载（compute / memory / mixed）
- [x] 按需场景加载
- [x] 依赖检测系统
- [x] perf 兼容性修复（CSV 格式）

---

## 项目里程碑

| 版本 | 目标 | 状态 |
|------|------|------|
| v0.3.0 | JVM 应用分析 | ✅ 完成 |
| v0.3.1 | 热点 C++ 化 + Bug 修复 | ✅ 完成 |
| v0.4.0 | Windows GUI + 容器级采集 | 📋 规划中 |
