# Java 热点 C++ 化用户指南

ArkProbe 可以自动识别 Java 应用的热点方法，并生成高度优化的 C++ 实现（通过 JNI），将 Java JIT 热点手动替换为性能更高的本地代码。

参考实现：Meta **Velox** 向量表达式引擎 + Intel **UMF (Unified Memory Framework)** 访存优化库。

---

## 快速开始

```bash
# 1. 启动目标 Java 应用（例如一个计算密集型服务）
java -jar myapp.jar &

# 2. 获取 JVM PID
jps -l

# 3. 运行热点分析（30秒采集）
arkprobe hotspot --jvm-pid <PID> --output ./hotspot_output

# 4. 查看生成的 C++ 代码
ls ./hotspot_output/codegen/

# 5. 查看性能对比（如果启用了 benchmark）
cat ./hotspot_output/acceleration_result.json
```

---

## 完整命令

```bash
arkprobe hotspot [OPTIONS]

选项：
  --jvm-pid INTEGER       JVM 进程 PID（必需）
  --output DIR           输出目录（默认：./hotspot_output）
  --duration INTEGER     JFR 采集时长（秒，默认：30）
  --min-cpu FLOAT        最小 CPU 百分比阈值（默认：2.0）
  --min-speedup FLOAT    最小加速比阈值（默认：1.5）
  --no-benchmark         跳过性能基准测试
  --arch TEXT            目标架构（armv8-a+simd / x86-64-v2）
  --opt-level TEXT       优化级别（O1/O2/O3，默认：O3）
```

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. JFR 热点采集                                            │
│     jcmd <pid> JFR.start events=jdk.ExecutionSample        │
│     ↓ 30秒采样                                               │
│     → 热点方法列表（CPU 时间分布）                          │
├─────────────────────────────────────────────────────────────┤
│  2. 模式匹配                                                │
│     PatternMatcher 根据方法签名/类名分类：                  │
│       • vector_expr → 向量表达式（Stream/map/reduce）      │
│       • math        → 数学函数（sin/cos/sigmoid/gemm）     │
│       • string      → 字符串处理（split/replace/parse）    │
│       • memory_bandwidth → 访存密集型（arraycopy/scale）  │
├─────────────────────────────────────────────────────────────┤
│  3. 字节码提取（可选）                                      │
│     javap / jcmd HSDB → 原始 Java bytecode 供参考          │
├─────────────────────────────────────────────────────────────┤
│  4. C++ 代码生成                                            │
│     Jinja2 模板 → NEON SIMD / AVX2 / UMF 优化              │
├─────────────────────────────────────────────────────────────┤
│  5. CMake 编译                                               │
│     g++ -fPIC -shared -O3 -march=armv8-a+simd → lib.so     │
├─────────────────────────────────────────────────────────────┤
│  6. 性能验证                                                 │
│     BenchmarkRunner：Java vs C++ 对比                       │
│     → 加速比 ≥ min_speedup（默认 1.5x）才推荐             │
└─────────────────────────────────────────────────────────────┘
```

---

## 模式模板对照表

| 模式分类 | 模式子类 | Java 方法示例 | C++ 优化技术 | 参考 |
|---------|---------|--------------|------------|------|
| **vector_expr** | `vector_map` | `stream().map(x → x*x)` | NEON vmulq_f32 | Velox |
| | `vector_reduce` | `stream().sum()` | NEON vaddq_f32 tree-reduce | Velox |
| | `vector_filter` | `stream().filter(x > 0)` | NEON vcgeq_f32 + compact | Velox |
| **math** | `math_sigmoid` | `MathUtils.sigmoid(x)` | NEON vrecpe 近似 | SLEEF |
| | `math_relu` | `activation.relu(x)` | NEON vmaxq_f32 | cuDNN |
| | `math_gemm` | `matrix.multiply(A,B)` | Cache-blocked GEMM | OpenBLAS |
| | `sin/cos/sqrt` | `Math.sin(x)` | 直接调用 libm | glibc |
| **string** | `string_split` | `str.split(",")` | NEON memchr 批量扫描 | re2 |
| | `string_replace` | `str.replace("a","b")` | 批量 memcpy | simdstr |
| | `string_parse` | `Integer.parseInt()` | SIMD 数字提取 | Google dense_hash |
| **memory_bandwidth** | `array_copy` | `System.arraycopy()` | _mm256_stream_store (NTSTORE) | Intel UMF |
| | `array_scale` | `Arrays.fill(a, v)` | _mm256_mul_ps + 非临时存储 | UMF |
| | `array_add` | `a[i] += b[i]` | _mm256_add_ps + FMA | UMF |
| | `matrix_mul` | `for(i)for(j)for(k)` | 64×64 分块 + NEON | OpenBLAS |
| | `prefetch` | 手动预取 | _mm_prefetch(_MM_HINT_T0) | UMF |

---

## 输出目录结构

```bash
hotspot_output/
├── profiler/
│   └── hotspot-<pid>.jfr              # 原始 JFR 文件
├── bytecode/
│   └── <class>_<method>.bytecode      # 提取的字节码（文本）
├── codegen/
│   ├── templates/                     # Jinja2 模板
│   ├── <Class>_<method>.cpp          # 单个方法的 C++ 实现
│   ├── jni_bridge.cpp                # JNI_OnLoad + 注册表
│   └── CMakeLists.txt                # CMake 构建配置
├── build/
│   └── libarkprobe_hotspot.so        # 编译后的共享库
└── acceleration_result.json          # 性能对比结果
```

---

## acceleration_result.json 格式

```json
{
  "methods_analyzed": 156,
  "methods_classified": 23,
  "methods_accelerated": 18,
  "generated_files": [
    "hotspot_output/codegen/com_example_FastMath_compute.cpp",
    "..."
  ],
  "compiled_libraries": [
    "hotspot_output/build/libarkprobe_hotspot.so"
  ],
  "benchmark_results": [
    {
      "method_name": "com.example.FastMath.compute",
      "java_time_ms": 12.45,
      "cpp_time_ms": 3.82,
      "speedup_factor": 3.26,
      "iterations": 1000
    }
  ],
  "recommended_methods": [
    {
      "method": "com.example.FastMath.compute",
      "speedup": 3.26,
      "java_time_ms": 12.45,
      "cpp_time_ms": 3.82
    }
  ]
}
```

---

## 使用生成的 C++ 库

### 自动加载（推荐）

`HotspotAccelerator` 会调用 `JNILoader` 使用 `jcmd VM.loadLibrary` 自动加载 .so 到目标 JVM：

```python
from arkprobe.hotspot import accelerate

result = accelerate(jvm_pid=12345, output_dir="./out")
# 编译后的 libarkprobe_hotspot.so 已自动加载到 JVM
```

### 手动加载

```bash
# 方式1：jcmd（JDK 9+）
jcmd <pid> VM.loadLibrary /path/to/libarkprobe_hotspot.so

# 方式2：Java Attach API（创建临时 agent）
java -cp $JAVA_HOME/lib/tools.jar:. LoadAgent <pid> /path/to/lib.so
```

---

## 性能 Benchmark

### 工具集

| 文件 | 说明 |
|------|------|
| `arkprobe/benchmarks/java/HotspotBench.java` | 自包含 Java benchmark，11 个方法（4 类模式），自动检测 native 库 |
| `arkprobe/benchmarks/java/arkprobe_hotspot.cpp` | C++ JNI 实现，ARM NEON / x86 AVX2 / scalar 三路分支 |
| `arkprobe/benchmarks/hotspot_benchmark.sh` | 一键全流程：编译 → 基线 → 加速 → 对比 |
| `arkprobe/benchmarks/compare_results.py` | JSON 结果解析 + 速度比计算 + 表格输出 |

### 快速运行

```bash
cd arkprobe/benchmarks

# 一键全流程（编译 Java → 基线 → 编译 C++ → 加速 → 对比）
./hotspot_benchmark.sh

# 仅 Java 基线（不编译 C++）
./hotspot_benchmark.sh --java-only

# 跳过 JFR profiling，直接使用预置 C++ 实现
./hotspot_benchmark.sh --skip-profiling
```

### Benchmark 方法说明

| 方法 | 模式分类 | Java 实现 | C++ 优化技术 |
|------|---------|----------|-------------|
| `vector_map` | vector_expr | float[] 逐元素乘法 | NEON vmulq_f32 / AVX2 _mm256_mul_ps |
| `vector_reduce` | vector_expr | float[] 求和 | NEON vaddq_f32 tree-reduce |
| `vector_filter` | vector_expr | float[] 条件过滤 | NEON vcgeq_f32 + compact |
| `math_sigmoid` | math | 1/(1+exp(-x)) | NEON vrecpe 近似 / libm |
| `math_relu` | math | max(0, x) | NEON vmaxq_f32 / AVX2 _mm256_max_ps |
| `matmul` | math | 64x64 矩阵乘法 | Cache-blocked GEMM + SIMD |
| `array_copy` | memory_bandwidth | System.arraycopy | memcpy / SIMD stream load-store |
| `array_scale` | memory_bandwidth | 逐元素乘标量 | NEON vmulq_n_f32 |
| `prefetch` | memory_bandwidth | 顺序访问 + 软预取 | __builtin_prefetch / _mm_prefetch |
| `string_parse` | string | Integer.parseInt 批量 | sscanf / strtol |
| `string_search` | string | String.indexOf 批量 | memchr / SIMD strstr |

### x86 实测结果（3 次平均）

| 方法 | Java (ms) | C++ (ms) | 加速比 | 结论 |
|------|-----------|----------|--------|------|
| `vector_reduce` | 79.8 | 14.3 | **5.6x** | SIMD tree-reduce 大幅领先 |
| `vector_map` | 83.9 | 58.7 | **1.5x** | SIMD 有效但 JNI 开销吃掉部分收益 |
| `math_relu` | 46.7 | 33.5 | **1.4x** | NEON vmax 优于标量 |
| `math_sigmoid` | 48.2 | 41.0 | **1.2x** | exp() 调用开销为主 |
| `vector_filter` | 91.5 | 82.3 | **1.1x** | JNI 数组拷贝抵消 SIMD 收益 |
| `matmul` | 104.6 | 95.1 | **1.1x** | 小矩阵 JNI 开销显著 |
| `array_copy` | 13.5 | 13.4 | **1.0x** | 纯内存带宽瓶颈 |
| `array_scale` | 56.1 | 55.8 | **1.0x** | 内存带宽受限 |
| `prefetch` | 72.9 | 72.5 | **1.0x** | 预取收益被 JNI 抵消 |
| `string_parse` | 85.2 | 209.1 | **0.4x** | JNI GetStringUTFChars 逐串转换开销 |
| `string_search` | 42.3 | 263.8 | **0.16x** | JNI 字符串创建/销毁循环开销 |

### 关键发现

1. **数值型向量方法加速显著**：`vector_reduce` 达 5.6x，`vector_map` 1.5x — SIMD + GetPrimitiveArrayCritical 免拷贝
2. **内存密集型方法收益有限**：`array_copy/scale/prefetch` ~1.0x — 瓶颈在内存带宽而非计算
3. **字符串方法 JNI 开销致命**：每条字符串都需要 `GetStringUTFChars`/`ReleaseStringUTFChars`/`NewStringUTF`，JNI 边界转换成本远超 C++ 计算本身。这是 JNI 架构限制，非代码优化可解
4. **小矩阵 JNI 边界成本高**：64x64 matmul 仅 1.1x — 建议仅对 256x256+ 矩阵使用 C++ 加速

### 推荐加速策略

| 场景 | 推荐策略 | 预期加速 |
|------|---------|---------|
| 数值向量 map/reduce/filter | C++ SIMD + CriticalArray | 1.5-5.6x |
| 数学函数（sigmoid/relu） | C++ libm + SIMD | 1.2-1.4x |
| 大矩阵乘法 (≥256x256) | C++ cache-blocked GEMM | 2-4x |
| 内存带宽密集型 | 不推荐 C++ 化 | ~1.0x |
| 字符串处理 | 不推荐 JNI C++ 化 | <1.0x |
| 小矩阵乘法 (<128x128) | 不推荐 JNI C++ 化 | ~1.1x |

---

## 性能验证

```bash
# 一键全流程 benchmark
cd arkprobe/benchmarks && ./hotspot_benchmark.sh

# 或通过 Python API
from arkprobe.hotspot.runtime.jni_loader import BenchmarkRunner

runner = BenchmarkRunner(java_bin="java", jni_lib_path="/path/to/libarkprobe_hotspot.so")
results = runner.benchmark_external(
    class_path="arkprobe/benchmarks/java",
    class_name="HotspotBench",
    iterations=100,
)
for name, r in results.items():
    print(f"{name}: Java={r.java_time_ms:.2f}ms, C++={r.cpp_time_ms:.2f}ms, speedup={r.speedup:.2f}x")
```

---

## 模式分类规则详解

### vector_expr（向量表达式）

**触发条件：**
- 类名匹配：`.*Stream$`, `.*Vector$`, `.*ArrayList`
- 方法名：`forEach`, `map`, `filter`, `reduce`, `collect`, `sum`, `average`
- 签名含：`Ljava/util/function/` 或 `[D`, `[I`, `[J` 数组参数

**模板：** `vector_expr.cpp.j2` (NEON SIMD)

```cpp
// 示例：vector_reduce → 数组求和
float32x4_t acc = vdupq_n_f32(0.0f);
for (int i = 0; i < count; i += 4) {
    float32x4_t vec = vld1q_f32(&arr[i]);
    acc = vaddq_f32(acc, vec);
}
// Horizontal sum
float sum = vgetq_lane_f32(acc,0) + vgetq_lane_f32(acc,1) +
            vgetq_lane_f32(acc,2) + vgetq_lane_f32(acc,3);
```

### math（数学函数）

**触发条件：**
- 类名：`.*Math`, `.*FastMath`, `.*NumericUtils`, `.*Matrix`
- 方法名：`sin`, `cos`, `exp`, `log`, `sqrt`, `sigmoid`, `relu`, `gemm`

**模板：** `math.cpp.j2`（直接 libm 调用）

```cpp
// sigmoid: f(x) = 1 / (1 + exp(-x))
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_Math_sigmoid(JNIEnv* env, jobject thiz, jdouble x) {
    return 1.0 / (1.0 + std::exp(-x));
}
```

### string（字符串处理）

**触发条件：**
- 类名：`.*String`, `.*Regex`, `.*Pattern`, `.*Parser`
- 方法名：`split`, `replace`, `matches`, `parse`, `toString`

**模板：** `string.cpp.j2`（SIMD memchr + NEON）

### memory_bandwidth（访存密集型）

**触发条件：**
- 类名：`.*ByteBuffer`, `.*DirectBuffer`, `.*Arrays`, `.*ArrayUtil`
- 方法名：`arraycopy`, `copyOf`, `fill`, `scale`, `add`, `put`, `get`

**模板：** `umf_template.cpp.j2`（Intel UMF 风格）

```cpp
// array_copy：非临时存储（Non-Temporal Store）
for (int i = 0; i < avx_count; i += 8) {
    __m256 vec = _mm256_stream_load_ps(&src[i]);   // 不污染缓存
    _mm256_stream_store_ps(&dst[i], vec);           // NTSTORE 直写内存
}
```

---

## Python API

```python
from pathlib import Path
from arkprobe.hotspot import (
    HotspotAccelerator,
    AccelerationConfig,
    CppGenerator,
    PatternMatcher,
    Compiler,
)

# 端到端运行
result = accelerate(
    jvm_pid=12345,
    output_dir="./hotspot_out",
    profiling_duration=30,
    run_benchmark=True,
)

print(f"分析了 {result.methods_analyzed} 个方法")
print(f"分类了 {result.methods_classified} 个方法")
print(f"生成了 {len(result.generated_files)} 个 C++ 文件")

for rec in result.recommended_methods:
    print(f"  {rec['method']}: {rec['speedup']:.2f}x")

# 单独使用各模块
config = AccelerationConfig(output_dir=Path("./out"))
accelerator = HotspotAccelerator(jvm_pid=12345, config=config)
result = accelerator.run(profiling_duration=30, run_benchmark=True)

report = accelerator.generate_report(result)
print(report)
```

---

## 测试

```bash
# 运行 hotspot 测试（37 个）
pytest tests/test_hotspot/ -v

# 包含集成测试
pytest tests/test_hotspot/test_e2e_integration.py -v
```

---

## 已知限制

1. **字节码提取**：需要目标 Java 应用的 `.class` 文件可访问（通过 javap）或 HSDB 支持
2. **编译**：需要 GCC/Clang + CMake 安装，JNI 头文件（JAVA_HOME 配置）
3. **运行时加载**：`jcmd VM.loadLibrary` 需要 JDK 9+ 且 attach 机制可用
4. **字符串 JNI 瓶颈**：逐字符串 JNI 转换开销远超 C++ 计算本身，string 模式不推荐用于 JNI 加速
5. **小数据集 JNI 边界成本**：数据量小时 JNI 调用开销占比高，建议数组长度 ≥ 10K 再考虑 C++ 化
6. **内存带宽受限场景**：纯拷贝/缩放等访存密集操作，C++ 无额外收益

---

## 变更记录

### 2026-04-23 Benchmark + 性能验证

- **NEW**: `HotspotBench.java` — 11 方法自包含 Java benchmark（vector/math/memory/string 4 类模式）
- **NEW**: `arkprobe_hotspot.cpp` — C++ JNI 实现，ARM NEON / x86 AVX2 / scalar 三路分支
- **NEW**: `hotspot_benchmark.sh` — 一键全流程 benchmark 脚本
- **NEW**: `compare_results.py` — JSON 结果对比 + 表格输出
- **NEW**: `BenchmarkRunner.benchmark_external()` — 两次运行对比法（纯 Java vs 加载 .so）
- **FINDING**: 数值向量方法 1.1-5.6x 加速，字符串方法 JNI 开销导致 0.16-0.4x 减速
- **FINDING**: 内存带宽受限方法 ~1.0x，无额外收益
- **OPT**: `GetPrimitiveArrayCritical` 替代 `GetFloatArrayElements`，消除数组拷贝开销
- **FIX**: `extern "C"` 包裹 JNI 函数，防止 C++ name mangling 导致 UnsatisfiedLinkError

- **CRITICAL**: JNI 名称修饰双重 `Java_` 前缀 — 修复 `cpp_generator.py` 和 `jni_bridge.cpp.j2` 的配合
- **CRITICAL**: 实现完整 JNI 规范修饰（`_`→`_1`, `;`→`_2`, `[`→`_3`, `$`→`_00024`, Unicode→`_0xxxx`）
- **CRITICAL**: `string.cpp.j2` 空 target 无限循环 — `find("")` 返回 0 导致死循环
- **CRITICAL**: `string.cpp.j2` NewStringUTF 返回值未检查 OOM 空指针
- **CRITICAL**: `jni_bridge.cpp.j2` RegisterNatives 失败后仅打日志未返回 JNI_ERR
- **HIGH**: `pattern_matcher.py` SIMD 检测搜索 ARM NEON 硬件指令码（Java 字节码里不存在），改为 JVM 字节码模式
- **HIGH**: `estimate_simd_potential` 误判 `0x0a` 为循环指示，改为搜索 JVM 循环跳转指令
- **HIGH**: `jni_loader.py` 异常吞没 + JDK8 agent 加载始终返回 True
- **HIGH**: `benchmark_runner.py` 纯 sleep 占位 → 加 jcmd JFR 尝试
- **MEDIUM**: `target_arch` 默认值改为 None（跨平台自动检测）
- **MEDIUM**: CMakeLists.txt.j2 移除硬编码 `stdc++`
- **MEDIUM**: 模板 `target_arch=None` 时崩溃修复
- **MEDIUM**: 未使用 import 清理

---

## 真实 Workload 性能验证

### 概述

在 HotspotBench 微基准测试之外，使用 5 个真实计算密集型 Java 算法验证 C++ 加速效果。这些是科学计算、信号处理和机器学习中常见的真实算法，而非人为构造的微内核。

### 工具集

| 文件 | 说明 |
|------|------|
| `arkprobe/benchmarks/java/RealWorkloadBench.java` | 自包含 Java benchmark，5 个真实算法，自动检测 native 库 |
| `arkprobe/benchmarks/java/arkprobe_realworkload.cpp` | C++ JNI 实现，ARM NEON / x86 AVX2 / scalar 三路分支 |
| `arkprobe/benchmarks/real_workload_benchmark.sh` | 一键全流程：编译 → 基线 → 加速 → 对比 |

### 快速运行

```bash
cd arkprobe/benchmarks

# 一键全流程
./real_workload_benchmark.sh

# 仅 Java 基线
./real_workload_benchmark.sh --java-only
```

### 5 个真实算法

| 方法 | 算法 | 数据规模 | C++ 优化技术 |
|------|------|---------|-------------|
| `fftTransform` | Cooley-Tukey radix-2 1D FFT | 2^20 (1M) 点, 50 次 | 蝶形运算 SIMD 展开 |
| `sorIteration` | 红黑 SOR 迭代（PDE 求解器） | 500×500, 50 迭代 | 标量循环（红黑模式不规则访问） |
| `sparseMatvec` | CSR 稀疏矩阵向量乘 | 100K×100K, 10 nnz/行, 200 次 | 逐行 SIMD 点积 + 展开 |
| `luDecompose` | 部分主元 LU 分解 | 512×512, 10 次 | SIMD 行消去（NEON vmlsq / AVX2） |
| `kmeansAssign` | KMeans 距离计算 | 100K 点 × 32 维 × 64 质心, 50 次 | SIMD 距离计算（dx²+dy²+...） |

### x86 实测结果

| 方法 | Java (ms) | C++ (ms) | 加速比 | 分析 |
|------|-----------|----------|--------|------|
| `luDecompose` | 270 | 134 | **2.0x** | 行消去是连续内存 SIMD 操作，收益最大 |
| `sparseMatvec` | 210 | 156 | **1.3x** | 逐行 SIMD 点积有效，但 gather-scatter 访存限制收益 |
| `kmeansAssign` | 4850 | 3710 | **1.3x** | 距离计算是完美 SIMD 场景，但 n×k 循环结构限制了向量化比例 |
| `sorIteration` | 143 | 118 | **1.2x** | 红黑模式跳步访问，SIMD 无效但 C++ 编译优化仍有收益 |
| `fftTransform` | 2000 | 1950 | **1.0x** | 蝶形运算数据依赖强，twiddle 旋转串行，SIMD 收益被 JNI 开销抵消 |

### 与微基准测试对比

| 类别 | 微基准 (HotspotBench) | 真实算法 (RealWorkload) | 说明 |
|------|----------------------|----------------------|------|
| 数值向量 | 1.1-5.6x | — | 微基准数据布局完美适配 SIMD |
| 行消去/矩阵 | 1.1x (64×64) | **2.0x** (512×512 LU) | 大矩阵 JNI 边界成本占比低，SIMD 收益显著 |
| 距离计算 | — | **1.3x** (KMeans) | n×k×dim 三层循环，中间层向量化 |
| 稀疏计算 | — | **1.3x** (Sparse) | gather-scatter 限制，但仍有收益 |
| 模板迭代 | — | **1.2x** (SOR) | 红黑模式不规则访问，SIMD 无效 |
| FFT 蝶形 | — | **1.0x** (FFT) | 数据依赖强，twiddle 串行旋转 |

### 关键发现

1. **真实算法加速比低于微基准**：微基准的数据布局是手工优化的完美场景，真实算法有更多不规则访问和数据依赖
2. **LU 分解加速最大**（2.0x）：行消去是连续内存操作，SIMD 效率最高
3. **FFT 收益最小**（1.0x）：蝶形运算中 twiddle 因子递推是串行的，SIMD 无法有效向量化
4. **SOR 仍有收益**（1.2x）：虽然红黑跳步模式不适合 SIMD，但 C++ 编译器优化（循环展开、指令调度）仍带来收益
5. **数据规模决定 JNI 边界成本占比**：LU 512×512 比 HotspotBench 64×64 的加速比高得多，因为计算量增大后 JNI 开销占比降低

### 推荐加速策略（更新）

| 场景 | 推荐策略 | 预期加速 |
|------|---------|---------|
| 数值向量 map/reduce/filter | C++ SIMD + CriticalArray | 1.5-5.6x |
| 大矩阵行消去 (≥256×256) | C++ SIMD 行操作 | 2.0x |
| 数学函数（sigmoid/relu） | C++ libm + SIMD | 1.2-1.4x |
| KMeans 距离计算 | C++ SIMD 距离 | 1.3x |
| 稀疏矩阵向量乘 | C++ SIMD 逐行点积 | 1.3x |
| 大矩阵乘法 (≥256×256) | C++ cache-blocked GEMM | 2-4x |
| 模板迭代（SOR） | C++ 编译优化（非 SIMD） | 1.2x |
| FFT 蝶形运算 | 不推荐 C++ 化 | ~1.0x |
| 内存带宽密集型 | 不推荐 C++ 化 | ~1.0x |
| 字符串处理 | 不推荐 JNI C++ 化 | <1.0x |

---

## 下一步

- 支持 ARM SVE（可伸缩向量扩展）而非仅 NEON
- 批量字符串处理接口：减少 JNI 调用次数（一次传入 String[] 而非逐条）
- 自动回退机制：C++ 失败时自动切回 Java 实现
- 运行时监控：动态决策是否启用加速（基于输入规模阈值）
- 鲲鹏 930 实机验证真实 workload 加速比
- FFT 优化：尝试 4-step / split-radix 算法提高 SIMD 利用率

---

## 相关资源

- Velox 向量引擎：https://github.com/facebookincubator/velox
- Intel UMF：https://github.com/intel/umf
- NEON 编程指南：https://developer.arm.com/architectures/instruction-sets/simd-isas/neon
- JNI 规范：https://docs.oracle.com/javase/8/docs/technotes/guides/jni/
