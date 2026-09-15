#!/usr/bin/env python
"""CBS estimates of the atomic Hartree-Fock energies of H2O.

Runs RHF/cc-pVDZ, TZ, QZ with the NAO-EDA and applies the two-point linear
(Halkier, Karton-Martin) and the three-point nonlinear (Feller)
extrapolations atom by atom.  The table shows the atomic-sum errors and the
fitted exponents alpha_A, which reveal whether an atomic HF energy converges
monotonically with the cardinal number.
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
