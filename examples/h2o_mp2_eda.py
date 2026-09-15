#!/usr/bin/env python
"""MP2-EDA of H2O (MP2/cc-pVDZ, frozen core).

The correlation energy is partitioned following Kobayashi, Imamura, Nakai,
J. Chem. Phys. 127, 074103 (2007), Eq. (18), with the occupied-orbital
weight w_occ = 1 adopted in that paper.  Isolated-atom reference energies
are obtained from ordinary UMP2 calculations.
"""

from pyscf import gto, scf, mp
from pyscf_eda import mp2 as eda_mp2
import numpy
numpy.set_printoptions(precision=10, floatmode='fixed', suppress=True, linewidth=200)

BASIS = 'cc-pvdz'
mol = gto.M(
    atom='''
    O  0.000000  0.000000  0.000000
    H  0.000000  0.757193  0.586080
    H  0.000000 -0.757193  0.586080
    ''',
    basis=BASIS, verbose=0)
mf = scf.RHF(mol).run()
pt = mp.MP2(mf, frozen=1).run()

res = eda_mp2.EDA(pt).kernel()                        # conventional (AO) partition
print(res.summary())

print('\nOccupied vs. virtual partition of E_corr (w_occ = 1 / 0 / 0.5):')
print('  occ  :', res.e_corr_occ.round(10))
print('  vir  :', res.e_corr_vir.round(10))
print('  half :', (0.5 * (res.e_corr_occ + res.e_corr_vir)).round(10))

res_nao = eda_mp2.EDA(pt, orbital_basis='nao').kernel()   # NAO-based partition
print('\nNAO-EDA: E_corr^A =', res_nao.e_corr.round(10), ' E_MP2^A =', res_nao.e_tot.round(10))

# --- differences from the isolated atoms (UMP2 reference) -----------------
e_atom = {}
for symb, spin, ncore in (('O', 2, 1), ('H', 1, 0)):
    atom = gto.M(atom=f'{symb} 0 0 0', basis=BASIS, spin=spin, verbose=0)
    umf = scf.UHF(atom).run()
    e_atom[symb] = float(mp.UMP2(umf, frozen=ncore).run().e_tot)

print('\nIsolated-atom UMP2 energies (hartree):', e_atom)
print(f"{'Atom':<6}{'E_MP2^A':>20}{'E(atom, UMP2)':>20}{'Difference':>18}")
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    print(f"{symb + str(ia):<6}{res.e_tot[ia]:>20.10f}{e_atom[symb]:>20.10f}"
          f"{res.e_tot[ia] - e_atom[symb]:>18.10f}")
e_atoms_sum = sum(e_atom[mol.atom_symbol(ia)] for ia in range(mol.natm))
print(f"{'Sum':<6}{res.e_tot.sum():>20.10f}{e_atoms_sum:>20.10f}"
      f"{pt.e_tot - e_atoms_sum:>18.10f}   (= -atomization energy)")
