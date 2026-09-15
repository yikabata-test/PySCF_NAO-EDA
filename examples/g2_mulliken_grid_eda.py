#!/usr/bin/env python
"""Mulliken-EDA, Grid-EDA and conventional EDA for closed-shell members of
Tables II and III of Y. Kikuchi, Y. Imamura, H. Nakai, Int. J. Quantum
Chem. 109, 2464 (2009): C2H2, C2H4, C2H6, SiH2 (1A1), SiH4 and HF.

Conditions of the paper: B3LYP with VWN5 ('B3LYP5'), 6-31G(d,p) with
Cartesian d functions (GAMESS default), Becke partition with Bragg radii
and the four smoothing iterations of GAMESS (code='gamess').
Geometries are the MP2(full)/6-31G(d) structures of the G2-1 set (values
below are taken from the literature to ~1e-3 A; the paper does not list
them, so differences of ~1 mhartree in the atomic energies are expected).
Isolated-atom energies (for Delta E, kcal/mol) come from unrestricted
B3LYP5 calculations of the ground-state atoms.
"""

import numpy
from pyscf import gto, dft
from pyscf_eda import rhf as eda_rhf

HARTREE2KCAL = 627.5095
BASIS = '6-31g(d,p)'

MOLECULES = {
    'C2H2': ('C 0 0 0.6082; C 0 0 -0.6082; H 0 0 1.6745; H 0 0 -1.6745', {'C': 0, 'H': 2}),
    'C2H4': ('''C 0 0 0.6676; C 0 0 -0.6676;
               H 0 0.9234 1.2373; H 0 -0.9234 1.2373; H 0 0.9234 -1.2373; H 0 -0.9234 -1.2373''',
             {'C': 0, 'H': 2}),
    'C2H6': ('''C 0 0 0.7631; C 0 0 -0.7631;
               H 1.0198 0 1.1596; H -0.5099 0.8832 1.1596; H -0.5099 -0.8832 1.1596;
               H -1.0198 0 -1.1596; H 0.5099 0.8832 -1.1596; H 0.5099 -0.8832 -1.1596''',
             {'C': 0, 'H': 2}),
    'SiH2(1A1)': ('Si 0 0 0; H 0 1.0923 1.0616; H 0 -1.0923 1.0616', {'Si': 0, 'H': 1}),
    'SiH4': ('''Si 0 0 0; H 0.8560 0.8560 0.8560; H -0.8560 -0.8560 0.8560;
               H -0.8560 0.8560 -0.8560; H 0.8560 -0.8560 -0.8560''', {'Si': 0, 'H': 1}),
    'HF': ('F 0 0 0; H 0 0 0.9337', {'F': 0, 'H': 1}),
}
# paper values: (q_Mull, q_Grid, E_Mull, E_Grid, E_conv) and Delta E in kcal/mol
PAPER = {
    ('C2H2', 'C'): (6.15, 6.13, -38.081, -38.067, -38.085, (161.0, 151.8, 163.4)),
    ('C2H2', 'H'): (0.85, 0.87, -0.558, -0.572, -0.554, (38.2, 47.3, 35.7)),
    ('C2H4', 'C'): (6.20, 6.13, -38.140, -38.108, -38.138, (197.9, 177.8, 196.5)),
    ('C2H4', 'H'): (0.90, 0.93, -0.564, -0.580, -0.565, (42.0, 52.0, 42.7)),
    ('C2H6', 'C'): (6.31, 6.14, -38.203, -38.141, -38.180, (237.0, 198.3, 222.8)),
    ('C2H6', 'H'): (0.90, 0.95, -0.561, -0.582, -0.569, (40.5, 53.4, 45.2)),
    ('SiH2(1A1)', 'Si'): (13.83, 13.86, -289.420, -289.433, -289.420, (63.0, 71.2, 62.9)),
    ('SiH2(1A1)', 'H'): (1.08, 1.07, -0.567, -0.561, -0.568, (44.3, 40.3, 44.4)),
    ('SiH4', 'Si'): (13.78, 13.85, -289.526, -289.574, -289.524, (129.6, 159.6, 128.5)),
    ('SiH4', 'H'): (1.05, 1.04, -0.573, -0.561, -0.574, (48.0, 40.5, 48.3)),
    ('HF', 'F'): (9.36, 9.07, -99.898, -99.764, -99.827, (136.0, 51.8, 91.3)),
    ('HF', 'H'): (0.64, 0.93, -0.491, -0.625, -0.563, (-3.5, 80.1, 41.2)),
}
ATOM_SPIN = {'H': 1, 'C': 2, 'Si': 2, 'F': 1}


def atom_energy(symb):
    mol = gto.M(atom=f'{symb} 0 0 0', basis=BASIS, cart=True, spin=ATOM_SPIN[symb], verbose=0)
    mf = dft.UKS(mol, xc='B3LYP5')
    mf.grids = eda_rhf.becke_grids(mf, level=5, code='gamess')
    return mf.run(conv_tol=1e-10).e_tot


e_atom = {symb: atom_energy(symb) for symb in ATOM_SPIN}
print('isolated-atom energies (UKS B3LYP5):', {k: round(v, 5) for k, v in e_atom.items()})
print(f"\n{'':11}{'':3}{'q(Mull)':>15}{'q(Grid)':>15}{'E(Mull)':>21}{'E(Grid)':>21}{'E(conv)':>21}")
print(f"{'':14}" + ''.join(f"{'this':>8}{'paper':>7}" for _ in range(2))
      + ''.join(f"{'this':>11}{'paper':>10}" for _ in range(3)))
for name, (atom, sites) in MOLECULES.items():
    mol = gto.M(atom=atom, basis=BASIS, cart=True, verbose=0)
    mf = dft.RKS(mol, xc='B3LYP5')
    mf.grids = eda_rhf.becke_grids(mf, level=5, code='gamess')
    mf.run(conv_tol=1e-10)
    r_mull = eda_rhf.mulliken_eda(mf, orbital_basis='ao')
    r_grid = eda_rhf.grid_eda(mf)
    r_conv = eda_rhf.kernel(mf, orbital_basis='ao')
    print(f"{name:<11} E_total = {mf.e_tot:.5f}")
    for symb, ia in sites.items():
        p = PAPER[(name, symb)]
        vals = (r_mull.pop[ia], p[0], r_grid.pop[ia], p[1],
                r_mull.e_tot[ia], p[2], r_grid.e_tot[ia], p[3], r_conv.e_tot[ia], p[4])
        line = f"{'':11}{symb:<3}" + ''.join(f"{vals[2*i]:>8.2f}{vals[2*i+1]:>7.2f}" for i in range(2))
        line += ''.join(f"{vals[4+2*i]:>11.3f}{vals[5+2*i]:>10.3f}" for i in range(3))
        print(line)
        d = [-(r.e_tot[ia] - e_atom[symb]) * HARTREE2KCAL for r in (r_mull, r_grid, r_conv)]
        print(f"{'':14}Delta E (kcal/mol)  Mull {d[0]:7.1f} ({p[5][0]:6.1f})   "
              f"Grid {d[1]:7.1f} ({p[5][1]:6.1f})   conv {d[2]:7.1f} ({p[5][2]:6.1f})")
