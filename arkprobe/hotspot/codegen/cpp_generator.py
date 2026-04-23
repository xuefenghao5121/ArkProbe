"""
C++ code generator for hotspot methods using Jinja2 templates.

Generates SIMD-optimized C++ code from HotspotMethod objects.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jinja2

from arkprobe.hotspot.models import HotspotMethod
from arkprobe.hotspot.analyzer.pattern_matcher import PatternClassification

log = logging.getLogger(__name__)


@contextmanager
def _get_template_dir():
    """Get template directory, using importlib.resources when available."""
    try:
        from importlib.resources import as_file, files
        templates = files("arkprobe.hotspot.codegen").joinpath("templates")
        with as_file(templates) as path:
            yield path
    except ImportError:
        # Python < 3.9 fallback
        yield Path(__file__).parent / "templates"


@dataclass
class GenerationConfig:
    """Configuration for C++ code generation."""
    output_dir: Path
    library_name: str = "arkprobe_hotspot"
    optimization_level: str = "O3"
    target_arch: str = "armv8-a+simd"
    use_neon: bool = True


class CppGenerator:
    """Generate C++ JNI wrapper code for hotspot methods."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.config.output_dir = Path(config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Jinja2 environment
        with _get_template_dir() as template_dir:
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )

        # Register custom filters
        self.env.filters["mangle"] = self._mangle_class_name

    def _get_template_for_classification(self, classification: PatternClassification) -> str:
        """Map pattern classification to template file."""
        pattern_type = classification.pattern_type
        pattern_subtype = classification.pattern_subtype

        # Map to template files
        if pattern_type == "memory_bandwidth":
            return "umf_template.cpp.j2"
        elif pattern_type == "vector_expr":
            return "vector_expr.cpp.j2"
        elif pattern_type == "math":
            return "math.cpp.j2"
        elif pattern_type == "string":
            return "string.cpp.j2"
        else:
            return "generic.cpp.j2"

    def generate_for_method(
        self,
        method: HotspotMethod,
        classification: PatternClassification,
        original_bytecode: Optional[str] = None,
    ) -> Path:
        """Generate C++ source file for a single hotspot method.

        Args:
            method: HotspotMethod to generate code for
            classification: Pattern classification result
            original_bytecode: Optional bytecode hex for reference

        Returns:
            Path to generated .cpp file
        """
        class_name = method.name.rsplit(".", 1)[0]
        method_name = method.name.rsplit(".", 1)[-1]

        template_name = self._get_template_for_classification(classification)

        # Check template exists in the package template directory (not output_dir)
        template_dir = Path(__file__).parent / "templates"
        template_path = template_dir / template_name
        if not template_path.exists():
            log.warning("Template not found: %s, using generic", template_name)
            template_name = "generic.cpp.j2"

        template = self.env.get_template(template_name)

        class_name_simple = class_name.split(".")[-1]
        mangled_name = self._mangle_class_name(class_name)

        context = {
            "method": method,
            "class_name": class_name,
            "class_name_simple": class_name_simple,
            "mangled_class_name": mangled_name,
            "method_name": method_name,
            "pattern_type": classification.pattern_type,
            "pattern_subtype": classification.pattern_subtype,
            "original_bytecode": original_bytecode or "(bytecode not available)",
            "params": self._infer_params(method.signature),
            "param_count": 0,  # calculated
            "return_type": self._infer_return_type(method.signature),
            "library_name": self.config.library_name,
            "target_arch": self.config.target_arch,
        }

        # Calculate param count from params
        context["param_count"] = len(context["params"])

        output_content = template.render(**context)

        # Write to file
        output_file = self.config.output_dir / f"{mangled_name}_{method_name}.cpp"
        output_file.write_text(output_content, encoding="utf-8")

        log.info("Generated C++ code: %s", output_file)
        return output_file

    def generate_jni_bridge(self, methods: list[tuple[HotspotMethod, PatternClassification]]) -> Path:
        """Generate JNI bridge for multiple methods.

        Args:
            methods: List of (HotspotMethod, PatternClassification) tuples

        Returns:
            Path to generated jni_bridge.cpp
        """
        class_name = methods[0][0].name.rsplit(".", 1)[0] if methods else "com.example.Generated"
        class_name_simple = class_name.split(".")[-1]
        mangled_name = self._mangle_class_name(class_name)

        template = self.env.get_template("jni_bridge.cpp.j2")

        methods_data = []
        for method, classification in methods:
            method_name = method.name.rsplit(".", 1)[-1]
            params = self._infer_params(method.signature)
            return_type = self._infer_return_type(method.signature)
            mangled_method = f"{mangled_name}_{method_name}"

            methods_data.append({
                "method": method,
                "method_name": method_name,
                "mangled_name": mangled_method,
                "signature": method.signature,
                "params": params,
                "return_type": return_type,
                "pattern_type": classification.pattern_type,
            })

        context = {
            "class_name": class_name,
            "library_name": self.config.library_name,
            "methods": methods_data,
        }

        output_content = template.render(**context)
        output_file = self.config.output_dir / "jni_bridge.cpp"
        output_file.write_text(output_content, encoding="utf-8")

        log.info("Generated JNI bridge: %s", output_file)
        return output_file

    def generate_cmake(self, sources: list[str] | list[Path], output_lib: Optional[str] = None) -> Path:
        """Generate CMakeLists.txt for building the shared library.

        Args:
            sources: List of .cpp source files (str or Path)
            output_lib: Output library name (without extension)

        Returns:
            Path to generated CMakeLists.txt
        """
        if output_lib is None:
            output_lib = self.config.library_name

        template = self.env.get_template("CMakeLists.txt.j2")

        # Convert string paths to Path objects for relative_to, or keep as strings
        source_paths = []
        for s in sources:
            if isinstance(s, str):
                source_paths.append(s)
            else:
                source_paths.append(str(s.relative_to(self.config.output_dir)))

        context = {
            "library_name": output_lib,
            "sources": source_paths,
            "optimization_level": self.config.optimization_level,
            "target_arch": self.config.target_arch,
        }

        output_content = template.render(**context)
        output_file = self.config.output_dir / "CMakeLists.txt"
        output_file.write_text(output_content, encoding="utf-8")

        log.info("Generated CMakeLists.txt")
        return output_file

    def _mangle_class_name(self, class_name: str) -> str:
        """Convert Java class name to JNI mangled name per JNI spec.

        JNI mangling rules:
        - '.' → '_' (package separator)
        - '_' → '_1' (underscore escape)
        - ';' → '_2' (semicolon escape)
        - '[' → '_3' (array indicator)
        - '$' → '_00024' (Unicode escape for U+0024)
        - Any char > 0x7F → '_0xxxx' (4-digit hex Unicode escape)
        """
        result = []
        for ch in class_name:
            if ch == ".":
                result.append("_")
            elif ch == "_":
                result.append("_1")
            elif ch == ";":
                result.append("_2")
            elif ch == "[":
                result.append("_3")
            elif ch == "$":
                result.append("_00024")
            elif ch == "/":
                result.append("_")
            elif ord(ch) > 0x7F:
                result.append(f"_0{ord(ch):04x}")
            else:
                result.append(ch)
        return "".join(result)

    def _infer_params(self, signature: str) -> list[dict]:
        """Infer JNI parameter types from JVM method signature.

        Args:
            signature: JVM method descriptor, e.g., "(ID)Ljava/lang/String;"

        Returns:
            List of parameter descriptors with JNI types
        """
        param_types = self._parse_descriptor_params(signature)

        jni_type_map = {
            "Z": {"jni": "jboolean", "c": "bool"},
            "B": {"jni": "jbyte", "c": "int8_t"},
            "C": {"jni": "jchar", "c": "char"},
            "S": {"jni": "jshort", "c": "int16_t"},
            "I": {"jni": "jint", "c": "int32_t"},
            "J": {"jni": "jlong", "c": "int64_t"},
            "F": {"jni": "jfloat", "c": "float"},
            "D": {"jni": "jdouble", "c": "double"},
        }

        params = []
        for i, ptype in enumerate(param_types):
            info = jni_type_map.get(ptype, {"jni": "jobject", "c": "void*"})
            params.append({
                "index": i,
                "jni_type": info["jni"],
                "c_type": info["c"],
                "name": f"arg{i}",
            })

        return params

    def _infer_return_type(self, signature: str) -> str:
        """Infer JNI return type from JVM method descriptor."""
        return_type = self._parse_descriptor_return(signature)

        jni_type_map = {
            "V": "void",
            "Z": "jboolean",
            "B": "jbyte",
            "C": "jchar",
            "S": "jshort",
            "I": "jint",
            "J": "jlong",
            "F": "jfloat",
            "D": "jdouble",
            "Ljava/lang/String;": "jstring",
            "Ljava/util/List;": "jobject",
        }

        return jni_type_map.get(return_type, "jobject")

    def _parse_descriptor_params(self, signature: str) -> list[str]:
        """Parse parameter types from JVM descriptor."""
        # Validate signature format
        if not signature.startswith("("):
            raise ValueError(f"Invalid JVM signature: expected '(' at start, got {signature!r}")

        # Strip parentheses
        inner = signature[1:signature.find(")")] if ")" in signature else signature[1:]
        params = []
        i = 0

        while i < len(inner):
            c = inner[i]
            if c in "BCSIJDZ":  # Primitive types
                params.append(c)
                i += 1
            elif c == "[":  # Array
                depth = 1
                while i + depth < len(inner) and inner[i + depth] == "[":
                    depth += 1
                if i + depth < len(inner):
                    elem = inner[i + depth]
                    params.append(f"[{depth}]" + elem)
                    i += depth + 1
                else:
                    params.append("[]")
                    break
            elif c == "L":  # Object
                end = inner.find(";", i)
                if end != -1:
                    params.append(inner[i:end + 1])
                    i = end + 1
                else:
                    # Malformed object signature, skip
                    log.warning("Malformed object signature in %s", signature)
                    i += 1
            else:
                # Unknown descriptor character, skip
                log.debug("Unknown descriptor character '%s' in signature %s", c, signature)
                i += 1

        return params

    def _parse_descriptor_return(self, signature: str) -> str:
        """Parse return type from JVM descriptor."""
        # Find return type after params
        end_params = signature.find(")")
        if end_params == -1:
            return "V"

        ret = signature[end_params + 1:]
        if ret.startswith("L"):
            end = ret.find(";")
            return ret[:end + 1] if end != -1 else ret
        return ret or "V"


def generate_cpp_code(
    method: HotspotMethod,
    classification: PatternClassification,
    output_dir: Path,
    original_bytecode: Optional[str] = None,
) -> Path:
    """Convenience function to generate C++ for a single method."""
    config = GenerationConfig(output_dir=output_dir)
    generator = CppGenerator(config)
    return generator.generate_for_method(method, classification, original_bytecode)
