"""Optional compiled kernels (loaded through ctypes).

``load()`` returns the shared library with the (T) partition kernel or
None.  The library is built by ``pip install`` (setup.py, optional
extension); if it is missing and a C compiler is available it is compiled
on first use into this directory.
"""

import ctypes
import glob
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None
_tried = False


def _candidates():
    return sorted(glob.glob(os.path.join(_HERE, '_ccsd_t_eda*.so'))
                  + glob.glob(os.path.join(_HERE, '_ccsd_t_eda*.dylib'))
                  + glob.glob(os.path.join(_HERE, '_ccsd_t_eda*.pyd')))


def compile_kernel(verbose=False):
    """Compile ccsd_t_eda.c with the system C compiler (OpenMP if possible)."""
    src = os.path.join(_HERE, 'ccsd_t_eda.c')
    out = os.path.join(_HERE, '_ccsd_t_eda.so')
    cc = os.environ.get('CC', 'cc')
    for flags in (['-fopenmp'], []):
        cmd = [cc, '-O3', '-fPIC', '-shared'] + flags + ['-o', out, src]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except OSError:
            return None
        if r.returncode == 0:
            if verbose:
                sys.stderr.write('compiled %s (%s)\n' % (out, ' '.join(flags) or 'no OpenMP'))
            return out
        if verbose:
            sys.stderr.write(r.stderr)
    return None


def load(allow_compile=True):
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    paths = _candidates()
    if not paths and allow_compile and os.access(_HERE, os.W_OK):
        p = compile_kernel()
        paths = [p] if p else []
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
            lib.ccsd_t_eda_partition.restype = ctypes.c_int
            _lib = lib
            return _lib
        except OSError:
            continue
    return None


def dgemm_pointer():
    """C function pointer of BLAS dgemm (Fortran interface) from SciPy, or None."""
    try:
        from scipy.linalg import cython_blas
    except ImportError:
        return None
    cap = cython_blas.__pyx_capi__.get('dgemm')
    if cap is None:
        return None
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    name = ctypes.pythonapi.PyCapsule_GetName(cap)
    if b'int *' not in name:          # unexpected integer width: fall back to the C loops
        return None
    return ctypes.pythonapi.PyCapsule_GetPointer(cap, name)


def num_threads():
    lib = load()
    if lib is None:
        return 1
    return int(lib.ccsd_t_eda_num_threads())
