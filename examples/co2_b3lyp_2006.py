#!/usr/bin/env python
"""Compare with Tables 1-2 of T. Baba, M. Takeuchi, H. Nakai,
Chem. Phys. Lett. 424, 193 (2006): populations (MPA / LPA / NPA) and
conventional / LSO- / NAO-EDA energies of the C atom in CO2 at B3LYP with
a series of basis sets.

Conditions matching the GAMESS calculations of the paper: Cartesian
functions, B3LYP with VWN5 ('B3LYP5' in PySCF), Becke partition with
Bragg radii and the four smoothing iterations of GAMESS (code='gamess'),
half/half nucleus-electron partition.
"""

from pyscf import gto, dft
from pyscf_eda import rhf as eda_rhf

# paper values: (MPA, LPA, NPA, E_C^EDA, E_C^LSO, E_C^NAO, E_total)
PAPER = {
    'sto-3g':      (5.665, 5.829, 5.633, -37.55102, -38.43323, -38.47270, -185.88273),
    '6-31g':       (5.443, 5.727, 5.051, -38.01449, -37.96637, -39.31266, -188.41470),
    '6-31g(d)':    (5.284, 5.776, 4.984, -38.02342, -38.32583, -39.41701, -188.49796),
    '6-31g(2d)':   (5.365, 6.113, 4.997, -38.01635, -37.57143, -39.50650, -188.50961),
    '6-31g(3df)':  (3.897, 6.475, 4.996, -37.63640, -38.16258, -39.26726, -188.52753),
    '6-311g':      (5.495, 5.738, 5.081, -37.99481, -38.18361, -39.18979, -188.47227),
    '6-311g(d)':   (5.489, 6.037, 5.037, -38.00342, -38.16769, -39.38255, -188.56045),
    '6-311g(2d)':  (5.617, 6.299, 5.011, -38.08182, -37.82744, -39.36032, -188.56470),
    '6-311g(3df)': (5.841, 6.529, 5.020, -38.06297, -38.22434, -39.41735, -188.57821),
    '6-31+g':      (5.326, 5.585, 5.054, -37.97814, -37.99135, -39.25424, -188.42386),
    '6-31+g(d)':   (5.252, 5.646, 4.984, -38.00467, -38.16049, -39.33633, -188.50747),
    '6-31+g(2d)':  (5.562, 5.972, 4.985, -37.98960, -37.51728, -39.44493, -188.52156),
    '6-31+g(3df)': (3.482, 6.266, 5.000, -37.90856, -38.02774, -39.30324, -188.53428),
    '6-311+g':     (5.414, 5.598, 5.085, -37.95659, -38.13234, -39.16070, -188.47768),
    '6-311+g(d)':  (5.499, 5.896, 5.034, -37.99613, -38.08650, -39.33182, -188.56633),
    '6-311+g(2d)': (5.647, 6.145, 5.006, -38.10145, -37.78692, -39.32100, -188.57006),
    '6-311+g(3df)':(5.269, 6.324, 5.019, -38.07671, -38.12326, -39.36560, -188.58206),
    'cc-pvdz':     (5.670, 6.102, 4.981, -37.98529, -38.53061, -39.44289, -188.51825),
    'cc-pvtz':     (5.768, 6.422, 5.035, -38.00323, -37.94612, -39.36470, -188.58015),
    'aug-cc-pvdz': (5.831, 6.201, 4.964, -37.90462, -36.81228, -39.46692, -188.53549),
    'aug-cc-pvtz': (4.419, 6.281, 5.029, -37.18110, -37.70634, -39.30588, -188.58439),
}

print(f"{'basis':<13}{'E_total':>17}{'dE':>15} | {'MPA':>6}{'LPA':>6}{'NPA':>6} | "
      f"{'EDA':>16}{'dEDA':>15}{'LSO':>16}{'dLSO':>15}{'NAO':>16}{'dNAO':>15}   (d = this - paper)")
for basis, ref in PAPER.items():
    try:
        mol = gto.M(atom='C 0 0 0; O 0 0 1.16; O 0 0 -1.16', basis=basis, cart=True, verbose=0)
    except Exception as exc:  # basis not available in PySCF
        print(f"{basis:<13} skipped ({exc})")
        continue
    mf = dft.RKS(mol, xc='B3LYP5')
    mf.grids = eda_rhf.becke_grids(mf, level=5, code='gamess')
    mf.run(conv_tol=1e-10)
    out = {}
    for ob in ('ao', 'lso', 'nao'):
        r = eda_rhf.kernel(mf, orbital_basis=ob, ne_partition='half')
        out[ob] = (r.pop[0], r.e_tot[0])
    print(f"{basis:<13}{mf.e_tot:>17.10f}{mf.e_tot-ref[6]:>15.10f} | "
          f"{out['ao'][0]:>6.3f}{out['lso'][0]:>6.3f}{out['nao'][0]:>6.3f} | "
          f"{out['ao'][1]:>16.10f}{out['ao'][1]-ref[3]:>15.10f}"
          f"{out['lso'][1]:>16.10f}{out['lso'][1]-ref[4]:>15.10f}"
          f"{out['nao'][1]:>16.10f}{out['nao'][1]-ref[5]:>15.10f}")
