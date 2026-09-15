#!/usr/bin/env python
"""Reproduce Table 1 of H. Nakai, Chem. Phys. Lett. 363, 73 (2002):
EDA of H2O at B3LYP/cc-pVDZ.

Conditions matching the HONDO99 calculation of the paper:
  * Cartesian d functions (6d), cc-pVDZ
  * B3LYP with the VWN-RPA correlation (PySCF's 'B3LYP')
  * exchange-correlation partition: Becke's original scheme (three
    smoothing iterations, as in HONDO99) with the Bragg-radius size
    adjustment (pyscf_eda.rhf.becke_grids(code='hondo'))
  * nucleus-electron attraction: half by basis functions, half by nuclei
"""

from pyscf import gto, dft
from pyscf_eda import rhf as eda_rhf

mol = gto.M(
    atom='''
    O  0.000000  0.000000  0.000000
    H  0.000000  0.757193  0.586080
    H  0.000000 -0.757193  0.586080
    ''',
    basis='cc-pvdz', cart=True, verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.grids = eda_rhf.becke_grids(mf, level=5, code='hondo')
mf.run(conv_tol=1e-10)
dm = mf.make_rdm1()

res = eda_rhf.EDA(mf, orbital_basis='ao', ne_partition='half').kernel()
comp = {xc: eda_rhf.xc_energy_by_atom(mf, xc, dm)
        for xc in ('LDA_X', 'GGA_X_B88', 'LDA_C_VWN_RPA', 'GGA_C_LYP')}
e_x_hf = res.e_x / 0.2
slater, b88 = comp['LDA_X'], comp['GGA_X_B88'] - comp['LDA_X']
vwn, lyp = comp['LDA_C_VWN_RPA'], comp['GGA_C_LYP']
e_x = 0.2 * e_x_hf + 0.8 * slater + 0.72 * b88
e_c = 0.19 * vwn + 0.81 * lyp

paper = {   # Table 1: (H, O, H2O)
    'E_NN': (2.38539, 4.42132, 9.19211), 'T_S': (0.71150, 74.50994, 75.93293),
    'E_Ne': (-5.80840, -187.52466, -199.14147), 'E_1EL': (-5.09691, -113.01472, -123.20853),
    'E_CLB': (2.55793, 41.84508, 46.96094), 'E_X^EXC': (-0.36919, -8.22906, -8.96744),
    'E_X^SLT': (-0.30032, -7.53400, -8.13464), 'E_X^B88': (-0.04239, -0.78332, -0.86810),
    'E_X': (-0.34462, -8.23700, -8.92624), 'E_C^VWN': (-0.06338, -0.73469, -0.86145),
    'E_C^LYP': (-0.01952, -0.30188, -0.34093), 'E_C': (-0.02786, -0.38412, -0.43983),
    'E_ELC': (-2.91145, -79.79075, -85.61366), 'E_TOT': (-0.52606, -75.36943, -76.42155)}
mine = {'E_NN': res.e_nn, 'T_S': res.e_kin, 'E_Ne': res.e_ne, 'E_1EL': res.e_1el, 'E_CLB': res.e_coul,
        'E_X^EXC': e_x_hf, 'E_X^SLT': slater, 'E_X^B88': b88, 'E_X': e_x, 'E_C^VWN': vwn,
        'E_C^LYP': lyp, 'E_C': e_c, 'E_ELC': res.e_elec, 'E_TOT': res.e_tot}

print(f"{'Component':<9}{'H (this)':>13}{'H (paper)':>12}{'O (this)':>14}{'O (paper)':>12}{'H2O (this)':>14}{'H2O (paper)':>12}")
for key, (ph, po, pt) in paper.items():
    v = mine[key]
    print(f"{key:<9}{v[1]:>13.5f}{ph:>12.5f}{v[0]:>14.5f}{po:>12.5f}{v.sum():>14.5f}{pt:>12.5f}")
print(f"\nmax |this - paper| over all entries: "
      f"{max(max(abs(mine[k][1]-ph), abs(mine[k][0]-po), abs(mine[k].sum()-pt)) for k,(ph,po,pt) in paper.items()):.2e} hartree")
