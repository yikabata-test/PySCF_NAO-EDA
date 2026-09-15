#!/usr/bin/env python
"""EDA of H2O at the RHF/cc-pVDZ level.

Same geometry as Table 1 of H. Nakai, Chem. Phys. Lett. 363, 73 (2002)
(R(O-H) = 0.9575 A, angle = 104.51 deg).  The paper reports B3LYP values;
here the analysis is done for closed-shell Hartree-Fock.

The differences from the isolated atoms (parentheses in Table 1) are
obtained with ordinary UHF calculations of the free atoms
(H: doublet, O: triplet); no EDA is needed for a single atom because its
total energy is trivially its own atomic energy.
"""

from pyscf import gto, scf
from pyscf_eda import rhf as eda_rhf
import numpy
numpy.set_printoptions(precision=10, floatmode='fixed', suppress=True, linewidth=200)

BASIS = 'cc-pvdz'

mol = gto.M(
    atom='''
    O  0.000000  0.000000  0.000000
    H  0.000000  0.757193  0.586080
    H  0.000000 -0.757193  0.586080
    ''',
    basis=BASIS,
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

# --- Differences from the isolated atoms (UHF reference) ------------------
e_atom = {}
for symb, spin in (('O', 2), ('H', 1)):
    atom = gto.M(atom=f'{symb} 0 0 0', basis=BASIS, spin=spin, verbose=0)
    e_atom[symb] = float(scf.UHF(atom).run().e_tot)

print('\nIsolated-atom UHF energies (hartree):', e_atom)
print(f"{'Atom':<6}{'E_TOT^A':>20}{'E(atom, UHF)':>20}{'Difference':>18}")
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    d = res.e_tot[ia] - e_atom[symb]
    print(f"{symb + str(ia):<6}{res.e_tot[ia]:>20.10f}{e_atom[symb]:>20.10f}{d:>18.10f}")
e_atoms_sum = sum(e_atom[mol.atom_symbol(ia)] for ia in range(mol.natm))
print(f"{'Sum':<6}{res.e_tot.sum():>20.10f}{e_atoms_sum:>20.10f}"
      f"{mf.e_tot - e_atoms_sum:>18.10f}   (= -atomization energy)")
