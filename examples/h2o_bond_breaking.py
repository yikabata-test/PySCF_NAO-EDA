#!/usr/bin/env python
"""Homogeneous / heterogeneous O-H bond stretching of H2O (cf. Table 2 of
H. Nakai, Chem. Phys. Lett. 363, 73 (2002)) at the RHF/cc-pVDZ level.

Prints the atomic total energies and their changes relative to the
equilibrium structure.
"""

import numpy
from pyscf import gto, scf
from pyscf_eda import rhf as eda_rhf
import numpy
numpy.set_printoptions(precision=10, floatmode='fixed', suppress=True, linewidth=200)

RE = 0.9575          # angstrom
ANGLE = 104.51       # degree


def h2o(ra, rb):
    th = numpy.radians(ANGLE / 2)
    return gto.M(
        atom=[['O', (0, 0, 0)],
              ['H', (0, ra * numpy.sin(th), ra * numpy.cos(th))],
              ['H', (0, -rb * numpy.sin(th), rb * numpy.cos(th))]],
        basis='cc-pvdz', verbose=0)


def atomic_energies(ra, rb):
    mf = scf.RHF(h2o(ra, rb)).run()
    return eda_rhf.atom_energies(mf), mf.e_tot


e_ref, etot_ref = atomic_energies(RE, RE)
print(f"{'R(O-Ha)':>8}{'R(O-Hb)':>8}{'E(O)':>18}{'E(Ha)':>18}{'E(Hb)':>18}"
      f"{'dE(O)':>14}{'dE(Ha)':>14}{'dE(Hb)':>14}{'dE(tot)':>14}")
for fa, fb in [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (1.0, 1.5), (1.0, 2.0)]:
    e, etot = atomic_energies(fa * RE, fb * RE)
    d = e - e_ref
    print(f"{fa:>6.1f}Re{fb:>6.1f}Re{e[0]:>18.10f}{e[1]:>18.10f}{e[2]:>18.10f}"
          f"{d[0]:>14.10f}{d[1]:>14.10f}{d[2]:>14.10f}{etot - etot_ref:>14.10f}")
