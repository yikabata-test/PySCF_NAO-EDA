#!/usr/bin/env python
"""Basis-set dependence of conventional, LSO- and NAO-EDA for CO2 (RHF).

Corresponds to Tables 1 and 2 of T. Baba, M. Takeuchi, H. Nakai,
Chem. Phys. Lett. 424, 193 (2006) (there at the B3LYP level).
R(C-O) = 1.16 A.  Printed are the population of the C atom
(Mulliken / Loewdin / natural) and the atomic energy of C as a
percentage of the total energy.
"""

import numpy
from pyscf import gto, scf
from pyscf_eda import rhf as eda_rhf

BASES = ['sto-3g', '6-31g', '6-31g(d)', '6-31+g(d)', '6-311+g(2d)',
         'cc-pvdz', 'cc-pvtz', 'aug-cc-pvdz', 'aug-cc-pvtz']
SCHEMES = [('ao', 'EDA'), ('lso', 'LSO-EDA'), ('nao', 'NAO-EDA')]

print(f"{'basis':<14}" + ''.join(f"{'q_C ' + n:>14}" for _, n in SCHEMES)
      + ''.join(f"{'E_C% ' + n:>14}" for _, n in SCHEMES) + f"{'E_total':>14}")
fractions = {key: [] for key, _ in SCHEMES}
for basis in BASES:
    mol = gto.M(atom='C 0 0 0; O 0 0 1.16; O 0 0 -1.16', basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    pops, fracs = [], []
    for key, _ in SCHEMES:
        res = eda_rhf.kernel(mf, orbital_basis=key)
        pops.append(res.pop[0])
        fracs.append(100 * res.e_tot[0] / mf.e_tot)
        fractions[key].append(fracs[-1])
    print(f"{basis:<14}" + ''.join(f"{p:>14.3f}" for p in pops)
          + ''.join(f"{f:>14.2f}" for f in fracs) + f"{mf.e_tot:>14.5f}")

print()
for key, name in SCHEMES:
    f = numpy.array(fractions[key])
    print(f"{name:<8} E_C/E_total: min {f.min():6.2f}  max {f.max():6.2f}  "
          f"mean {f.mean():6.2f}  std {f.std():5.2f} (%)")
