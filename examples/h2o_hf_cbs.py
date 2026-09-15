#!/usr/bin/env python
"""CBS estimates of the atomic Hartree-Fock energies of H2O.

Runs RHF/cc-pVDZ, TZ, QZ with the NAO-EDA and applies the two-point linear
extrapolations (Halkier, Karton-Martin) atom by atom.  All schemes are
linear, so the atomic sums equal the molecular extrapolations; the table
lists the atomic values of every scheme and these sum checks.
"""

from pyscf import gto
from pyscf_eda import cbs as eda_cbs

mol = gto.M(
    atom='''
    O  0.000000  0.000000  0.000000
    H  0.000000  0.757193  0.586080
    H  0.000000 -0.757193  0.586080
    ''',
    basis='cc-pvdz', verbose=0)

for orbital_basis in ('nao', 'ao'):
    res = eda_cbs.HFCBS(mol, orbital_basis=orbital_basis).kernel()
    print(res.summary())
    print()
