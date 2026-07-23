from __future__ import annotations

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


def _scip_include_dir() -> str:
    candidates = []
    if prefix := os.environ.get("SCIPOPTDIR"):
        candidates.append(Path(prefix) / "include")
    candidates.extend((Path("/opt/homebrew/include"), Path("/usr/local/include")))
    for candidate in candidates:
        if (candidate / "scip" / "scip.h").is_file():
            return str(candidate)
    raise RuntimeError("SCIP headers not found; set SCIPOPTDIR to the SCIP prefix")


setup(
    ext_modules=cythonize(
        [
            Extension(
                "scip_cut_trace_v2._scip_pointer",
                ["src/scip_cut_trace_v2/_scip_pointer.pyx"],
                include_dirs=[_scip_include_dir()],
            )
        ],
        compiler_directives={"language_level": 3},
    )
)
