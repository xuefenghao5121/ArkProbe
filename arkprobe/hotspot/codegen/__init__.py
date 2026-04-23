"""Code generation module for hotspot C++ acceleration."""

from .cpp_generator import CppGenerator, GenerationConfig, generate_cpp_code

__all__ = ["CppGenerator", "GenerationConfig", "generate_cpp_code"]
