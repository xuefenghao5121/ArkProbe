"""
C++ compiler toolchain for building JNI shared libraries.

Uses GCC/Clang via CMake to compile generated C++ code into .so files.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from arkprobe.hotspot.codegen.cpp_generator import CppGenerator

log = logging.getLogger(__name__)


# Supported architectures and their compiler flags
ARCH_FLAGS = {
    # ARM
    "armv8-a": ["-march=armv8-a"],
    "armv8-a+simd": ["-march=armv8-a+simd"],
    "armv8.2-a": ["-march=armv8.2-a"],
    "armv8.2-a+simd": ["-march=armv8.2-a+simd"],
    # x86
    "x86-64": ["-march=x86-64"],
    "x86-64-v2": ["-march=x86-64-v2"],
    "x86-64-v3": ["-march=x86-64-v3"],
    "x86-64-v4": ["-march=x86-64-v4"],
    "haswell": ["-march=haswell"],
    "skylake": ["-march=skylake-avx512"],
    "native": [],  # Let compiler decide based on host
    "generic": [],  # No arch-specific flags
}


def detect_host_arch() -> str:
    """Detect the host CPU architecture."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "armv8-a+simd"
    elif machine in ("x86_64", "amd64"):
        return "native"  # Let GCC/Clang optimize for host
    else:
        log.warning("Unknown architecture %s, defaulting to generic", machine)
        return "generic"


@dataclass
class CompileResult:
    """Result of a compilation attempt."""
    success: bool
    output_file: Optional[Path] = None
    stderr: str = ""
    stdout: str = ""
    error: Optional[str] = None


class Compiler:
    """Compile C++ hotspot code into JNI shared libraries."""

    def __init__(
        self,
        build_dir: Optional[Path] = None,
        target_arch: Optional[str] = None,
    ):
        self.build_dir = Path(build_dir) if build_dir else Path("./build_hotspot")

        # Auto-detect host architecture if not specified
        if target_arch is None:
            target_arch = detect_host_arch()
        self.target_arch = target_arch

        # Check prerequisites
        self._validate_prerequisites()

    def _validate_prerequisites(self) -> None:
        """Validate that required build tools are available."""
        errors = []

        # CMake check
        cmake_path = shutil.which("cmake")
        if cmake_path:
            self.cmake_path = cmake_path
        else:
            errors.append(
                "CMake not found. Install cmake: apt install cmake (Debian/Ubuntu) or yum install cmake (RHEL/CentOS)"
            )

        # Compiler check
        for compiler_name in ("g++", "clang++"):
            compiler = shutil.which(compiler_name)
            if compiler:
                self.gpp_path = compiler
                break
        else:
            errors.append(
                "No C++ compiler found. Install g++ or clang++: apt install g++"
            )

        if errors:
            raise RuntimeError(
                "Compilation prerequisites not met:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def compile(
        self,
        generator: CppGenerator,
        output_lib: Optional[str] = None,
    ) -> CompileResult:
        """Compile generated C++ code into a shared library.

        Args:
            generator: CppGenerator with generated source files
            output_lib: Output library name (without lib prefix)

        Returns:
            CompileResult with success status and output path
        """
        if output_lib is None:
            output_lib = generator.config.library_name

        source_dir = generator.config.output_dir
        build_dir = self.build_dir
        build_dir.mkdir(parents=True, exist_ok=True)

        # Generate CMakeLists.txt if not present
        cmake_file = source_dir / "CMakeLists.txt"
        if not cmake_file.exists():
            generator.generate_cmake([], output_lib)

        # Run CMake configuration
        log.info("Running CMake configuration...")
        cmake_cmd = [
            self.cmake_path,
            "-S", str(source_dir),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        result = self._run_command(cmake_cmd, build_dir)
        if not result.success:
            return result

        # Run build
        log.info("Building hotspot library...")
        build_cmd = [
            self.cmake_path,
            "--build", str(build_dir),
            "--config", "Release",
            "-j", str(os.cpu_count() or 4),
        ]

        result = self._run_command(build_cmd, build_dir)
        if not result.success:
            return result

        # Find the built .so file
        so_file = build_dir / f"lib{output_lib}.so"
        if not so_file.exists():
            # Try alternate locations
            for candidate in build_dir.rglob(f"lib{output_lib}*.so"):
                so_file = candidate
                break

        if not so_file.exists():
            return CompileResult(
                success=False,
                error=f"Compiled library not found: {so_file}",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        log.info("Compilation successful: %s", so_file)
        return CompileResult(
            success=True,
            output_file=so_file,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def compile_single(
        self,
        cpp_file: Path,
        output_so: Optional[Path] = None,
        extra_cflags: Optional[list[str]] = None,
    ) -> CompileResult:
        """Compile a single C++ file directly without CMake.

        Args:
            cpp_file: Path to .cpp source file
            output_so: Output .so path
            extra_cflags: Additional compiler flags

        Returns:
            CompileResult with output path
        """
        if output_so is None:
            output_so = cpp_file.with_suffix(".so")

        compiler = self.gpp_path or "g++"
        cmd = [
            compiler,
            "-shared",
            "-fPIC",
            "-O3",
        ]

        # Add architecture-specific flags
        arch_flags = ARCH_FLAGS.get(self.target_arch, [])
        cmd.extend(arch_flags)

        if extra_cflags:
            cmd.extend(extra_cflags)

        # JNI includes
        java_home = os.environ.get("JAVA_HOME")
        if not java_home:
            # Try to find JAVA_HOME
            java_exe = shutil.which("java")
            if java_exe:
                java_home = str(Path(java_exe).parent.parent)

        if java_home and Path(java_home).exists():
            cmd.extend([
                f"-I{java_home}/include",
            ])
            # Platform-specific JNI include subdirectory
            jni_subdir = self._jni_include_subdir()
            if (Path(java_home) / "include" / jni_subdir).exists():
                cmd.append(f"-I{java_home}/include/{jni_subdir}")
        else:
            # Try common locations
            machine = platform.machine().lower()
            arch_suffix = "amd64" if machine in ("x86_64", "amd64") else "arm64"
            for common_java_home in [
                "/usr/lib/jvm/default-java",
                f"/usr/lib/jvm/java-11-openjdk-{arch_suffix}",
                f"/usr/lib/jvm/java-17-openjdk-{arch_suffix}",
                f"/usr/lib/jvm/java-21-openjdk-{arch_suffix}",
            ]:
                if Path(common_java_home).exists():
                    cmd.extend([
                        f"-I{common_java_home}/include",
                        f"-I{common_java_home}/include/{self._jni_include_subdir()}",
                    ])
                    break

        cmd.extend(["-o", str(output_so), str(cpp_file)])

        result = self._run_command(cmd, cpp_file.parent)
        if result.success:
            result.output_file = output_so

        return result

    def _jni_include_subdir(self) -> str:
        """Return platform-specific JNI include subdirectory."""
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        elif system == "darwin":
            return "darwin"
        elif system == "windows":
            return "win32"
        return "linux"  # default

    def _run_command(self, cmd: list[str], cwd: Path) -> CompileResult:
        """Run a shell command and capture output."""
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return CompileResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                error="Compilation timed out after 5 minutes",
            )
        except Exception as e:
            return CompileResult(
                success=False,
                error=str(e),
            )


def compile_hotspot(
    generator: CppGenerator,
    build_dir: Optional[Path] = None,
    target_arch: Optional[str] = None,
) -> CompileResult:
    """Convenience function to compile hotspot code."""
    compiler = Compiler(build_dir, target_arch)
    return compiler.compile(generator)
