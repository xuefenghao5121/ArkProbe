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

## 性能验证

```bash
# 启用详细 benchmark（1000 次迭代，100 次预热）
arkprobe hotspot --jvm-pid 12345 --output ./out \
  --duration 60 --benchmark-iterations 1000 --warmup 100

# 查看加速比报告
python3 -c "
import json
with open('./out/acceleration_result.json') as f:
    r = json.load(f)
for m in r['recommended_methods']:
    print(f\"{m['method']}: {m['speedup']:.2f}x\")
"
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
4. **Benchmark**：当前通过 jcmd JFR 触发测量，精度有限；生产环境建议使用 JMH

---

## 变更记录

### 2026-04-23 Bug 修复

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

## 下一步

- 实现真实的 Java 方法调用 harness（JMH benchmark）
- 支持 ARM SVE（可伸缩向量扩展）而非仅 NEON
- 自动回退机制：C++ 失败时自动切回 Java 实现
- 运行时监控：动态决策是否启用加速（基于输入规模）

---

## 相关资源

- Velox 向量引擎：https://github.com/facebookincubator/velox
- Intel UMF：https://github.com/intel/umf
- NEON 编程指南：https://developer.arm.com/architectures/instruction-sets/simd-isas/neon
- JNI 规范：https://docs.oracle.com/javase/8/docs/technotes/guides/jni/
