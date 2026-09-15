#!/usr/bin/env python
"""CCSD(T)/CBS atomic energies (QDD and QTD) and the analytic nuclear
gradient of the CBS energy of H2O from one input.

The gradients of every level (RHF, MP2, CCSD, CCSD(T); frozen core) are
evaluated from the same calculations that provide the energies, so no
energy calculation is repeated, and are combined with the fitting
coefficients of the model.  The CCSD(T) gradient uses the CCSD(T) lambda
equations of PySCF (pyscf_eda.grad.ccsd_t_gradient).
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

for scheme in ('QDD', 'QTD'):
    res = eda_cbs.CompositeEDA(mol, scheme=scheme, with_grad=True).kernel()
    print(res.summary())
    print()
    print(f'{scheme}: atomic CCSD(T)/CBS energies (hartree):', res.e_tot.round(6))
    print(f'{scheme}: CBS gradient (hartree/bohr):')
    print(res.grad.round(6))
    print('per-level gradients available in res.grads, e.g. CCSD(T)/DZ:')
    print(res.grads[('CCSD(T)', 'D')].round(6))
    print()
