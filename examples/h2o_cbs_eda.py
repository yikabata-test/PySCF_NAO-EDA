#!/usr/bin/env python
"""Atomic energies of H2O at the CBS limit with the QDD and QTD models
(Seino & Nakai, J. Comput. Chem. 37, 2304 (2016), 3SLF, cc-pVXZ).

One input drives all required calculations:
  QDD : CCSD(T)/cc-pVDZ chain, MP2/cc-pVTZ, MP2/cc-pVQZ
  QTD : CCSD(T)/cc-pVDZ chain, CCSD/cc-pVTZ chain, MP2/cc-pVQZ
The lower-level energies of each basis set (MP2, CCSD) come from the same
chain of calculations.  Frozen-core correlation energies are used as in the
paper.
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

res_qtd = eda_cbs.QTD(mol).kernel()
print(res_qtd.summary())

res_qdd = eda_cbs.QDD(mol).kernel()
print('\nQDD atomic CBS correlation energies :', res_qdd.e_corr.round(6))
print('QTD atomic CBS correlation energies :', res_qtd.e_corr.round(6))
print('QDD / QTD molecular E_corr(CBS)      : %.6f / %.6f' % (res_qdd.e_corr_mol, res_qtd.e_corr_mol))
print('CCSD(T)/cc-pVDZ E_corr for reference : %.6f' % res_qtd.corr_mol[('CCSD(T)', 'D')])

# NAO-based partition with the HF part extrapolated (Karton-Martin, T/Q)
res_nao = eda_cbs.QTD(mol, orbital_basis='nao', hf_cbs='karton-martin').kernel()
print('\nQTD (NAO, HF extrapolated) atomic energies:', res_nao.e_tot.round(6))
