#!/usr/bin/env python
"""EDA of H2O at the RHF/cc-pVDZ level.

Same geometry as Table 1 of H. Nakai, Chem. Phys. Lett. 363, 73 (2002)
(R(O-H) = 0.9575 A, angle = 104.51 deg).  The paper reports B3LYP values;
here the analysis is done for closed-shell Hartree-Fock.
"""

from pyscf import gto, scf
from pyscf_eda import rhf as eda_rhf

mol = gto.M(
    atom='''
    O  0.000000  0.000000  0.000000
    H  0.000000  0.757193  0.586080
    H  0.000000 -0.757193  0.586080
    ''',
    basis='cc-pvdz',
    verbose=0,
)
mf = scf.RHF(mol).run()

# Default: E_Ne is partitioned half by basis functions, half by nuclei.
res = eda_rhf.EDA(mf).kernel()
print(res.summary())

# Atomic total energies as a numpy array
print('\nAtomic energies (hartree):', res.e_tot)

# Original 2002 scheme (E_Ne partitioned purely by basis functions)
res_mul = eda_rhf.EDA(mf, ne_partition='mulliken').kernel()
print('\nE_Ne with pure Mulliken-type partitioning:', res_mul.e_ne)
print('E_Ne with half/half partitioning          :', res.e_ne)
