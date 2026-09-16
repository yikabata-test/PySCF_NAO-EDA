"""Build configuration: the metadata lives in pyproject.toml; this file only
adds the optional compiled kernel for the CCSD(T)-EDA triples partition.
If no C compiler (or OpenMP) is available the extension is skipped and the
package falls back to its numpy implementation."""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Try OpenMP first, then plain C; never fail the installation."""

    def build_extension(self, ext):
        base_cflags = list(ext.extra_compile_args)
        base_ldflags = list(ext.extra_link_args)
        for cflags, ldflags in ((base_cflags + ['-fopenmp'], base_ldflags + ['-fopenmp']),
                                (base_cflags, base_ldflags)):
            ext.extra_compile_args = cflags
            ext.extra_link_args = ldflags
            try:
                build_ext.build_extension(self, ext)
                return
            except Exception as exc:      # noqa: BLE001
                sys.stderr.write('pyscf-eda: building %s failed (%s)\n' % (ext.name, exc))
        sys.stderr.write('pyscf-eda: the compiled (T) partition kernel is not available; '
                         'the numpy implementation will be used\n')


ext = Extension(
    'pyscf_eda.lib._ccsd_t_eda',
    sources=[os.path.join('pyscf_eda', 'lib', 'ccsd_t_eda.c')],
    extra_compile_args=['-O3', '-fPIC'],
    optional=True,
)

setup(ext_modules=[ext], cmdclass={'build_ext': OptionalBuildExt})
