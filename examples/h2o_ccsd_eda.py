#!/usr/bin/env python
"""CCSD-EDA of H2O (CCSD/cc-pVDZ, frozen core), compared with MP2-EDA.

The correlation energy is partitioned following Kobayashi and Nakai,
J. Chem. Phys. 129, 044103 (2008), Eq. (10), with the occupied-orbital
weight w_occ = 1 (Eq. 14).  Isolated-atom reference energies are obtained
from ordinary UCCSD calculations.
"""

from pyscf import gto, scf, mp, cc
from pyscf_eda import mp2 as eda_mp2
from pyscf_eda import ccsd as eda_ccsd
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
mycc = cc.CCSD(mf, frozen=1).run()
pt = mp.MP2(mf, frozen=1).run()

res = eda_ccsd.EDA(mycc).kernel()                     # conventional (AO) partition
print(res.summary())

res_mp2 = eda_mp2.EDA(pt).kernel()
print('\nAtomic correlation energies (w_occ = 1):')
print('  MP2  :', res_mp2.e_corr.round(10))
print('  CCSD :', res.e_corr.round(10))
print('CCSD occupied vs. virtual partition:')
print('  occ  :', res.e_corr_occ.round(10))
print('  vir  :', res.e_corr_vir.round(10))

res_nao = eda_ccsd.EDA(mycc, orbital_basis='nao').kernel()
print('\nNAO-EDA: E_corr^A =', res_nao.e_corr.round(10), ' E_CCSD^A =', res_nao.e_tot.round(10))

# --- differences from the isolated atoms (UCCSD reference) ----------------
e_atom = {}
for symb, spin, ncore in (('O', 2, 1), ('H', 1, 0)):
    atom = gto.M(atom=f'{symb} 0 0 0', basis=BASIS, spin=spin, verbose=0)
    umf = scf.UHF(atom).run()
    e_atom[symb] = float(cc.UCCSD(umf, frozen=ncore).run().e_tot)

print('\nIsolated-atom UCCSD energies (hartree):', e_atom)
print(f"{'Atom':<6}{'E_CCSD^A':>20}{'E(atom, UCCSD)':>20}{'Difference':>18}")
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    print(f"{symb + str(ia):<6}{res.e_tot[ia]:>20.10f}{e_atom[symb]:>20.10f}"
          f"{res.e_tot[ia] - e_atom[symb]:>18.10f}")
e_atoms_sum = sum(e_atom[mol.atom_symbol(ia)] for ia in range(mol.natm))
print(f"{'Sum':<6}{res.e_tot.sum():>20.10f}{e_atoms_sum:>20.10f}"
      f"{mycc.e_tot - e_atoms_sum:>18.10f}   (= -atomization energy)")
