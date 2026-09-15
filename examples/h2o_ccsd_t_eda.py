#!/usr/bin/env python
"""CCSD(T)-EDA of H2O (CCSD(T)/cc-pVDZ, frozen core).

The (T) correction is partitioned following Kobayashi and Nakai,
J. Chem. Phys. 131, 114108 (2009), with the U^{0,0} partition (occupied
orbital k, w_occ = 1) adopted in that paper.  Isolated-atom reference
energies are obtained from ordinary UCCSD(T) calculations.
"""

from pyscf import gto, scf, cc
from pyscf_eda import ccsd_t as eda_ccsd_t
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

res = eda_ccsd_t.EDA(mycc).kernel()                   # conventional (AO) partition
print(res.summary())
print('\nE_(T)^A  (U^{0,0}) :', res.e_t_occ.round(10))
print('E_(T)^A  (U^{2,2}) :', res.e_t_vir.round(10))
print('E_T[4]^A           :', res.e_t4.round(10))
print('E_ST[5]^A          :', res.e_st5.round(10))
print('E_(T) total        : %.10f' % res.e_t.sum())

res_nao = eda_ccsd_t.EDA(mycc, orbital_basis='nao').kernel()
print('\nNAO-EDA: E_(T)^A =', res_nao.e_t.round(10), ' E_CCSD(T)^A =', res_nao.e_tot.round(10))

# --- differences from the isolated atoms (UCCSD(T) reference) -------------
e_atom = {}
for symb, spin, ncore in (('O', 2, 1), ('H', 1, 0)):
    atom = gto.M(atom=f'{symb} 0 0 0', basis=BASIS, spin=spin, verbose=0)
    umf = scf.UHF(atom).run()
    ucc = cc.UCCSD(umf, frozen=ncore).run()
    e_atom[symb] = float(ucc.e_tot + ucc.ccsd_t())

print('\nIsolated-atom UCCSD(T) energies (hartree):', e_atom)
print(f"{'Atom':<6}{'E_CCSD(T)^A':>20}{'E(atom)':>20}{'Difference':>18}")
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    print(f"{symb + str(ia):<6}{res.e_tot[ia]:>20.10f}{e_atom[symb]:>20.10f}"
          f"{res.e_tot[ia] - e_atom[symb]:>18.10f}")
e_atoms_sum = sum(e_atom[mol.atom_symbol(ia)] for ia in range(mol.natm))
e_tot = mycc.e_tot + res.e_t.sum()
print(f"{'Sum':<6}{res.e_tot.sum():>20.10f}{e_atoms_sum:>20.10f}"
      f"{e_tot - e_atoms_sum:>18.10f}   (= -atomization energy)")
