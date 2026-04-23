"""
Pattern matcher for classifying hotspot methods and evaluating SIMD potential.

Analyzes method signatures and bytecode patterns to identify:
- Vector expression patterns (map/filter/reduce operations)
- String processing patterns (split, regex, parsing)
- Math-intensive patterns (sin/cos/exp/log, matrix operations)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from arkprobe.hotspot.models import HotspotMethod

log = logging.getLogger(__name__)

# Pattern definitions for matching hotspot types
PATTERN_RULES = {
    "vector_expr": {
        "class_prefixes": [
            r".*Stream$",
            r".*Collector",
            r".*Vector$",
            r".*Array",
            r".*ArrayList",
            r".*IntStream",
            r".*DoubleStream",
        ],
        "method_names": [
            "forEach", "map", "filter", "reduce", "collect", "sum", "average",
            "min", "max", "sorted", "parallelStream", "stream",
        ],
        "signature_indicators": [
            r"\(Ljava/util/function/",
            r"\(\[D\)",
            r"\(\[I\)",
            r"\(\[J\)",
        ],
        "min_cpu_percent": 5.0,
    },
    "string": {
        "class_prefixes": [
            r".*String",
            r".*Regex",
            r".*Pattern",
            r".*Parser",
            r".*Tokenizer",
        ],
        "method_names": [
            "split", "replaceAll", "replaceFirst", "matches", "find",
            "parse", "toString", "format", "substring", "indexOf",
        ],
        "signature_indicators": [
            r"\(Ljava/lang/String;",
            r"\(Ljava/util/regex/",
        ],
        "min_cpu_percent": 3.0,
    },
    "math": {
        "class_prefixes": [
            r".*Math",
            r".*FastMath",
            r".*NumericUtils",
            r".*Matrix",
            r".*Vector",
            r".*BLAS",
            r".*DenseMatrix",
            r".*SparseMatrix",
        ],
        "method_names": [
            "sin", "cos", "tan", "exp", "log", "sqrt", "pow", "abs",
            "min", "max", "multiply", "dot", "gemm", "transpose", "inverse",
            "sigmoid", "relu", "softmax", "tanh",
        ],
        "signature_indicators": [
            r"\(D\)D",  # double -> double
            r"\(I\)I",  # int -> int
            r"\(\[D\[D\[D\)",  # matrix multiply signature
        ],
        "min_cpu_percent": 2.0,
    },
    "memory_bandwidth": {
        "class_prefixes": [
            r".*ByteBuffer",
            r".*DirectBuffer",
            r".*FloatBuffer",
            r".*IntBuffer",
            r".*LongBuffer",
            r".*MappedByteBuffer",
            r".*DataOutput",
            r".*DataInput",
            r".*ArrayUtil",
            r".*Arrays",
        ],
        "method_names": [
            "arraycopy", "copyOf", "fill", "set",
            "put", "get", "read", "write",
            "bulkLoad", "bulkStore", "stream",
            "scale", "add", "multiply",
        ],
        "signature_indicators": [
            r"(\[D\[D)V",   # void double[] op double[]
            r"(\[F\[F)V",   # void float[] op float[]
            r"(\[I\[I)V",   # void int[] op int[]
            r"\[B\[B",       # byte[] arrays
            r"Ljava/nio/Buffer",
            r"Ljava/nio/ByteBuffer",
        ],
        "min_cpu_percent": 3.0,
    },
}


@dataclass
class PatternClassification:
    """Result of pattern matching."""
    method: HotspotMethod
    pattern_type: str
    confidence: float  # 0.0 to 1.0
    matched_rules: list[str]
    pattern_subtype: str = "generic"  # Sub-pattern for detailed template selection

    @property
    def is_vector_expr(self) -> bool:
        return self.pattern_type == "vector_expr"

    @property
    def is_string(self) -> bool:
        return self.pattern_type == "string"

    @property
    def is_math(self) -> bool:
        return self.pattern_type == "math"

    @property
    def is_memory_bandwidth(self) -> bool:
        return self.pattern_type == "memory_bandwidth"


class PatternMatcher:
    """Classify hotspot methods using signature and bytecode pattern matching."""

    def __init__(self):
        self.rules = PATTERN_RULES

    def classify(self, method: HotspotMethod) -> PatternClassification:
        """Classify a hotspot method into a pattern type.

        Args:
            method: HotspotMethod to classify

        Returns:
            PatternClassification with type and confidence score
        """
        scored = self._score_all_rules(method)

        if not scored:
            return PatternClassification(
                method=method,
                pattern_type="unknown",
                confidence=0.0,
                matched_rules=[],
            )

        # Pick highest scoring pattern
        best_pattern, best_score, matched = max(scored, key=lambda x: x[1])

        log.info(
            "Classified %s as '%s' (confidence=%.2f)",
            method.name,
            best_pattern,
            best_score,
        )

        return PatternClassification(
            method=method,
            pattern_type=best_pattern,
            confidence=best_score,
            matched_rules=matched,
            pattern_subtype=self._infer_subtype(method, best_pattern),
        )

    def _infer_subtype(self, method: HotspotMethod, pattern_type: str) -> str:
        """Infer the specific subtype for template selection."""
        method_name = method.name.rsplit(".", 1)[-1]
        sig = method.signature

        if pattern_type == "memory_bandwidth":
            # Detect array operation subtype
            if "arraycopy" in method_name or "copyOf" in method_name:
                return "array_copy"
            elif sig.count("[") >= 2:  # Multiple arrays = binary op
                if "scale" in method_name.lower() or "mul" in method_name.lower():
                    return "array_scale"
                elif "add" in method_name.lower() or "sum" in method_name.lower():
                    return "array_add"
            elif "mul" in method_name.lower() or "gemm" in method_name.lower():
                return "matrix_mul"
            elif "prefetch" in method_name.lower():
                return "prefetch"
            return "generic"

        elif pattern_type == "vector_expr":
            if "map" in method_name:
                return "vector_map"
            elif "reduce" in method_name or "sum" in method_name:
                return "vector_reduce"
            elif "filter" in method_name:
                return "vector_filter"
            return "generic"

        elif pattern_type == "string":
            if "split" in method_name:
                return "string_split"
            elif "replace" in method_name:
                return "string_replace"
            elif "parse" in method_name:
                return "string_parse"
            return "generic"

        elif pattern_type == "math":
            if "sigmoid" in method_name.lower():
                return "math_sigmoid"
            elif "relu" in method_name.lower():
                return "math_relu"
            elif "gemm" in method_name.lower() or "matmul" in method_name.lower():
                return "math_gemm"
            return "generic"

        return "generic"

    def _score_all_rules(self, method: HotspotMethod) -> list[tuple[str, float, list[str]]]:
        """Score all pattern rules for a method."""
        results: list[tuple[str, float, list[str]]] = []

        for pattern_name, rule in self.rules.items():
            score, matched = self._score_rule(method, rule)
            if score > 0.3:  # Minimum threshold
                results.append((pattern_name, score, matched))

        return results

    def _score_rule(self, method: HotspotMethod, rule: dict) -> tuple[float, list[str]]:
        """Score a single rule against a method."""
        matched: list[str] = []
        score = 0.0

        class_name = method.name.rsplit(".", 1)[0] if "." in method.name else ""
        method_name = method.name.rsplit(".", 1)[-1] if "." in method.name else method.name

        # Check class prefix patterns
        for pattern in rule.get("class_prefixes", []):
            if re.match(pattern, class_name):
                matched.append(f"class_prefix:{pattern}")
                score += 0.4

        # Check method name patterns
        for mname in rule.get("method_names", []):
            if method_name == mname:
                matched.append(f"method_name:{mname}")
                score += 0.5

        # Check signature patterns
        for sig_pattern in rule.get("signature_indicators", []):
            if re.search(sig_pattern, method.signature):
                matched.append(f"signature:{sig_pattern}")
                score += 0.3

        # Check bytecode patterns (if available)
        if method.bytecode_hex:
            # Simple bytecode opcode heuristics
            if self._has_simd_opcodes(method.bytecode_hex):
                score += 0.2
                matched.append("simd_opcodes")

        # Check CPU percentage - higher percentages get slightly higher score
        if method.cpu_time_percent >= rule.get("min_cpu_percent", 0.0):
            score += 0.1

        return min(score, 1.0), matched

    def _has_simd_opcodes(self, bytecode_hex: str) -> bool:
        """Check if bytecode contains loop/array patterns suitable for SIMD.

        Looks for JVM opcodes that indicate array access and arithmetic loops,
        which are good candidates for SIMD vectorization in C++.
        """
        if not bytecode_hex:
            return False
        # JVM opcodes indicating array operations and loops
        # 0x2a=aload_0, 0x32=aaload, 0x2b=aload_1, 0x1a=iload_0
        # 0xbc=newarray, 0xbd=anewarray, 0xc5=multianewarray
        ARRAY_OPCODES = {"2a", "32", "2b", "1a", "bc", "bd", "c5"}
        # JVM opcodes for arithmetic on int/float/double
        # 0x60=iadd, 0x68=imul, 0x70=dmul, 0x6d=fadd, 0x6f=fmul
        ARITH_OPCODES = {"60", "68", "70", "6d", "6f", "84"}
        hex_clean = bytecode_hex.replace("0x", "").replace(" ", "")
        if len(hex_clean) < 4:
            return False
        # Check byte pairs for array + arithmetic co-occurrence
        has_array = any(op in hex_clean for op in ARRAY_OPCODES)
        has_arith = any(op in hex_clean for op in ARITH_OPCODES)
        return has_array and has_arith

    def estimate_simd_potential(self, method: HotspotMethod) -> float:
        """Estimate potential speedup from SIMD vectorization (0.0-1.0)."""
        base = 0.5

        # Loop indicators from bytecode: goto (0xa7) and if* opcodes
        if method.bytecode_hex:
            hex_clean = method.bytecode_hex.replace("0x", "").replace(" ", "")
            # goto=0xa7, ifeq=0x99, ifne=0x9a, if_icmpge=0xa2, if_icmplt=0xa1
            loop_ops = {"a7", "99", "9a", "a2", "a1"}
            loop_count = sum(1 for op in loop_ops if op in hex_clean)
            if loop_count > 2:
                base += 0.2

        # Array operations get extra boost
        if "java/util/ArrayList" in method.name or "[]" in method.signature:
            base += 0.2

        return min(base, 1.0)

    def estimate_deopt_risk(self, method: HotspotMethod) -> float:
        """Estimate deoptimization risk for this method (0.0-1.0)."""
        risk = 0.0

        # High compilation count suggests unstable code (frequent recompiles)
        if method.compilation_count > 5:
            risk += 0.4

        # Inline count: very high inlining suggests aggressive JIT optimizations
        if method.inline_count > 100:
            risk += 0.3

        # Bytecode size: very large methods (>200 bytes) are harder to optimize
        if method.bytecode_size > 200:
            risk += 0.3

        return min(risk, 1.0)


# Convenience function
def classify_hotspot_method(method: HotspotMethod) -> PatternClassification:
    """One-shot classification function."""
    return PatternMatcher().classify(method)
