"""JIT build for the CUDA kernels.

Windows specifics handled here (see NOTES.md):
  - cl.exe is not on PATH, so the MSVC environment is imported from vcvarsall.bat.
  - nvcc exits 2 printing NOTHING when %TMP% contains a space, which it does here
    (.../Park Jiho/AppData/Local/Temp). TMP/TEMP are remapped to their 8.3 short form.
  - the extension dir has the same space problem with ninja, so it is remapped too.
"""

import functools
import os
import pathlib
import shutil
import subprocess

CSRC = pathlib.Path(__file__).resolve().parent / "csrc"
CUDA_ARCH = "12.0"  # RTX 5060 Ti (sm_120)

# no --use_fast_math: it would break bit-exactness with the PyTorch reference.
# --use-local-env: torch 2.11 does not pass it on Windows, so nvcc re-runs vcvars64.bat
# itself and dies with "Could not set up the environment"; this makes it reuse ours.
CUDA_FLAGS = ["-O3", "--expt-relaxed-constexpr", "-lineinfo"]
if os.name == "nt":
    CUDA_FLAGS.append("--use-local-env")
CXX_FLAGS = ["/O2"] if os.name == "nt" else ["-O3"]

_MSVC_ENV_KEYS = (
    "PATH", "INCLUDE", "LIB", "LIBPATH",
    "VCINSTALLDIR", "VCToolsInstallDir", "VSINSTALLDIR",
    "WindowsSdkDir", "WindowsSdkVerBinPath", "WindowsSDKVersion", "UniversalCRTSdkDir",
)


def _vcvarsall() -> pathlib.Path:
    vswhere = pathlib.Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vswhere = vswhere / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        raise RuntimeError(f"vswhere.exe not found at {vswhere}; install VS Build Tools")
    install = subprocess.run(
        [str(vswhere), "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not install:
        raise RuntimeError("vswhere found no VS install with the C++ x64 toolset")
    path = pathlib.Path(install) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not path.exists():
        raise RuntimeError(f"vcvarsall.bat not found at {path}")
    return path


def ensure_msvc_env(arch: str = "x64") -> None:
    """Put cl.exe and the MSVC/SDK include+lib dirs into this process's environment."""
    if os.name != "nt" or shutil.which("cl"):
        return
    # Must be one string, not a list: list2cmdline would escape the inner quotes as \" and
    # cmd would fail to find the batch file. The outer pair is what /s strips.
    command = f'cmd /s /c ""{_vcvarsall()}" {arch} >nul && set"'
    dumped = subprocess.run(
        command, capture_output=True, text=True, check=True, errors="replace",
    ).stdout
    env = {}
    for line in dumped.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key.upper()] = value
    for key in _MSVC_ENV_KEYS:
        if key.upper() in env:
            os.environ[key] = env[key.upper()]
    # vcvarsall shells out to a bare "vswhere.exe", which is not on PATH by default
    installer = str(pathlib.Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
                    / "Microsoft Visual Studio" / "Installer")
    if os.path.isdir(installer) and installer not in os.environ["PATH"]:
        os.environ["PATH"] = installer + ";" + os.environ["PATH"]
    if not shutil.which("cl"):
        raise RuntimeError("vcvarsall ran but cl.exe is still not on PATH")


def _short_path(path: pathlib.Path) -> pathlib.Path:
    """8.3 form, so ninja never sees a space. Returns path unchanged if unavailable."""
    if os.name != "nt":
        return path
    path.mkdir(parents=True, exist_ok=True)
    # one string, not a list, for the same quoting reason as ensure_msvc_env
    out = subprocess.run(
        f'cmd /c for %I in ("{path}") do @echo %~sI',
        capture_output=True, text=True, errors="replace",
    ).stdout.strip()
    return pathlib.Path(out) if out and " " not in out else path


def ensure_space_free_tmp() -> None:
    """nvcc dies with exit code 2 and no diagnostic at all when %TMP% has a space in it."""
    if os.name != "nt":
        return
    for key in ("TMP", "TEMP"):
        value = os.environ.get(key)
        if value and " " in value:
            short = str(_short_path(pathlib.Path(value)))
            if " " not in short:
                os.environ[key] = short


def ensure_extensions_dir() -> str:
    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        default = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home())) / "torch_extensions"
        os.environ["TORCH_EXTENSIONS_DIR"] = str(_short_path(default))
    return os.environ["TORCH_EXTENSIONS_DIR"]


@functools.lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    """Compile (once per process) and return the llmquant_kernels extension module."""
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the kernels need a GPU to build and run")
    ensure_msvc_env()
    ensure_space_free_tmp()
    ensure_extensions_dir()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", CUDA_ARCH)
    return load(
        name="llmquant_kernels",
        sources=[
            str(CSRC / "fake_quant.cpp"),
            str(CSRC / "fake_quant.cu"),
            str(CSRC / "wq_gemv.cu"),
            str(CSRC / "wq_gemv_batched.cu"),
        ],
        extra_cflags=CXX_FLAGS,
        extra_cuda_cflags=CUDA_FLAGS,
        verbose=verbose,
    )
