import numpy
import pytest
from pyscf import gto, scf, dft, mp, cc

from pyscf_eda import rhf as eda_rhf
from pyscf_eda import mp2 as eda_mp2
from pyscf_eda import ccsd as eda_ccsd

H2O_ATOM = '''
O  0.000000  0.000000  0.000000
H  0.000000  0.757193  0.586080
H  0.000000 -0.757193  0.586080
'''


def _rks(xc, basis='cc-pvdz', cart=False, becke=False, level=None, **kw):
    mol = gto.M(atom=H2O_ATOM, basis=basis, cart=cart, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    if becke:
        mf.grids = eda_rhf.becke_grids(mf, level=level)
    elif level is not None:
        mf.grids.level = level
    return mf.run(conv_tol=1e-10, **kw)


@pytest.fixture(scope='module')
def h2o_b3lyp():
    return _rks('B3LYP')


@pytest.mark.parametrize('xc', ('B3LYP', 'PBE', 'wB97X', 'TPSS'))
@pytest.mark.parametrize('orbital_basis', ('ao', 'nao'))
def test_sum_rules(xc, orbital_basis):
    mf = _rks(xc)
    mol = mf.mol
    dm = mf.make_rdm1()
    res = eda_rhf.kernel(mf, orbital_basis=orbital_basis)
    assert res.xc == xc
    assert res.e_xc is not None
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-8
    assert abs(res.e_xc.sum() - mf.scf_summary['exc']) < 1e-8 or \
        abs(res.e_xc.sum() + res.e_x.sum() - mf.scf_summary['exc']) < 1e-8
    assert abs(res.e_kin.sum() - numpy.einsum('ij,ji', dm, mol.intor_symmetric('int1e_kin'))) < 1e-9
    assert abs(res.e_tot[1] - res.e_tot[2]) < 1e-8
    assert abs(res.pop.sum() - mol.nelectron) < 1e-9
    if xc in ('PBE', 'TPSS'):      # no exact exchange
        assert numpy.all(res.e_x == 0)
    else:
        assert res.e_x.sum() < 0
    assert 'E_XC' in res.summary() and f'RKS {xc}' in res.summary()


def test_hf_functional_equals_rhf():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf_hf = scf.RHF(mol).run(conv_tol=1e-11)
    mf_ks = dft.RKS(mol, xc='HF').run(conv_tol=1e-11)
    r_hf = eda_rhf.kernel(mf_hf, orbital_basis='ao')
    r_ks = eda_rhf.kernel(mf_ks, orbital_basis='ao')
    assert numpy.allclose(r_hf.e_tot, r_ks.e_tot, atol=1e-7)
    assert numpy.allclose(r_hf.e_x, r_ks.e_x, atol=1e-7)
    assert numpy.all(r_ks.e_xc == 0)


def test_xc_components_add_up(h2o_b3lyp):
    mf = h2o_b3lyp
    dm = mf.make_rdm1()
    parts = {xc: eda_rhf.xc_energy_by_atom(mf, xc, dm)
             for xc in ('LDA_X', 'GGA_X_B88', 'LDA_C_VWN_RPA', 'GGA_C_LYP')}
    # B3LYP (VWN-RPA) = 0.08 Slater + 0.72 B88 + 0.19 VWN_RPA + 0.81 LYP  (+ 0.2 HF exchange)
    b3lyp = (0.08 * parts['LDA_X'] + 0.72 * parts['GGA_X_B88']
             + 0.19 * parts['LDA_C_VWN_RPA'] + 0.81 * parts['GGA_C_LYP'])
    res = eda_rhf.kernel(mf, orbital_basis='ao')
    assert numpy.allclose(b3lyp, res.e_xc, atol=1e-8)
    assert abs(res.e_xc.sum() - (mf.scf_summary['exc'] - res.e_x.sum())) < 1e-8


def test_grid_partition_independent_of_sorting(h2o_b3lyp):
    mf = h2o_b3lyp
    dm = mf.make_rdm1()
    ref = eda_rhf.xc_energy_by_atom(mf, 'B3LYP', dm)
    grids = dft.Grids(mf.mol)
    grids.build(sort_grids=False, with_non0tab=True)
    unsorted = eda_rhf.xc_energy_by_atom(mf, 'B3LYP', dm, grids=grids)
    assert numpy.allclose(ref, unsorted, atol=1e-8)


def test_reproduces_nakai_2002_table1():
    # H2O, B3LYP (VWN-RPA) / Cartesian cc-pVDZ, Becke partition with Bragg
    # radii, half/half nucleus-electron partition.  Table 1 of
    # H. Nakai, Chem. Phys. Lett. 363, 73 (2002).
    mf = _rks('B3LYP', cart=True, becke=True, level=5)
    dm = mf.make_rdm1()
    res = eda_rhf.kernel(mf, orbital_basis='ao', ne_partition='half')
    comp = {xc: eda_rhf.xc_energy_by_atom(mf, xc, dm)
            for xc in ('LDA_X', 'GGA_X_B88', 'LDA_C_VWN_RPA', 'GGA_C_LYP')}
    paper = {   # (O, H)
        'E_NN': (4.42132, 2.38539), 'T_S': (74.50994, 0.71150), 'E_Ne': (-187.52466, -5.80840),
        'E_CLB': (41.84508, 2.55793), 'E_X^HF': (-8.22906, -0.36919), 'SLT': (-7.53400, -0.30032),
        'B88': (-0.78332, -0.04239), 'VWN': (-0.73469, -0.06338), 'LYP': (-0.30188, -0.01952),
        'E_TOT': (-75.36943, -0.52606)}
    mine = {'E_NN': res.e_nn, 'T_S': res.e_kin, 'E_Ne': res.e_ne, 'E_CLB': res.e_coul,
            'E_X^HF': res.e_x / 0.2, 'SLT': comp['LDA_X'], 'B88': comp['GGA_X_B88'] - comp['LDA_X'],
            'VWN': comp['LDA_C_VWN_RPA'], 'LYP': comp['GGA_C_LYP'], 'E_TOT': res.e_tot}
    for key, (o, h) in paper.items():
        tol = 5e-4 if key == 'E_NN' else 3e-4     # E_NN: older Bohr-radius constant in HONDO
        assert abs(mine[key][0] - o) < tol, (key, mine[key][0], o)
        assert abs(mine[key][1] - h) < tol, (key, mine[key][1], h)
    assert abs(mf.e_tot - (-76.42155)) < 2e-5


def test_reproduces_baba_2006_co2_conventional():
    # CO2, B3LYP (VWN5) / Cartesian cc-pVDZ: total energy, Mulliken
    # population and conventional EDA of the C atom, Tables 1-2 of
    # Baba, Takeuchi, Nakai, Chem. Phys. Lett. 424, 193 (2006).
    mol = gto.M(atom='C 0 0 0; O 0 0 1.16; O 0 0 -1.16', basis='cc-pvdz', cart=True, verbose=0)
    mf = dft.RKS(mol, xc='B3LYP5')
    mf.grids = eda_rhf.becke_grids(mf, level=5)
    mf.run(conv_tol=1e-10)
    assert abs(mf.e_tot - (-188.51825)) < 1e-4
    res = eda_rhf.kernel(mf, orbital_basis='ao', ne_partition='half')
    assert abs(res.pop[0] - 5.670) < 2e-3
    # the conventional EDA is shifted by -0.05 hartree for every basis set
    # (E_XC grid partition of GAMESS vs PySCF); the NAO-EDA agrees to 3 mhartree
    assert abs(res.e_tot[0] - (-37.98529)) < 0.06
    assert abs(100 * res.e_tot[0] / mf.e_tot - 20.15) < 0.05
    lso = eda_rhf.kernel(mf, orbital_basis='lso', ne_partition='half')
    assert abs(lso.pop[0] - 6.102) < 0.02
    assert abs(100 * lso.e_tot[0] / mf.e_tot - 20.44) < 0.1
    nao = eda_rhf.kernel(mf, orbital_basis='nao', ne_partition='half')
    assert abs(nao.pop[0] - 4.981) < 0.03
    assert abs(nao.e_tot[0] - (-39.44289)) < 0.02
    assert abs(100 * nao.e_tot[0] / mf.e_tot - 20.92) < 0.02


def test_correlated_methods_reject_dft_reference(h2o_b3lyp):
    with pytest.raises(TypeError):
        eda_mp2.EDA(mp.MP2(h2o_b3lyp))
    with pytest.raises((TypeError, RuntimeError)):   # PySCF itself refuses a DFT reference
        eda_ccsd.EDA(cc.CCSD(h2o_b3lyp))


def test_rejects_uks():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    with pytest.raises((TypeError, NotImplementedError)):
        eda_rhf.EDA(dft.UKS(mol))
