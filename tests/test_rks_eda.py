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


# ---------------------------------------------------------------------------
# Mulliken-EDA and Grid-EDA (Kikuchi, Imamura, Nakai, IJQC 109, 2464 (2009))
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def h2o_b3lyp5_cart():
    mol = gto.M(atom=H2O_ATOM, basis='6-31g(d,p)', cart=True, verbose=0)
    mf = dft.RKS(mol, xc='B3LYP5')
    mf.grids = eda_rhf.becke_grids(mf, level=5)
    return mf.run(conv_tol=1e-10)


@pytest.mark.parametrize('orbital_basis', ('ao', 'nao', 'lso'))
def test_mulliken_eda_sum_rules(h2o_b3lyp5_cart, orbital_basis):
    mf = h2o_b3lyp5_cart
    res = eda_rhf.kernel(mf, orbital_basis=orbital_basis, xc_partition='mulliken')
    conv = eda_rhf.kernel(mf, orbital_basis=orbital_basis, xc_partition='grid')
    assert res.xc_partition == 'mulliken' and res.scheme == 'mulliken'
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-8
    assert abs(res.e_xc.sum() - conv.e_xc.sum()) < 1e-8      # same total, different partition
    assert not numpy.allclose(res.e_xc, conv.e_xc, atol=1e-4)
    # all non-XC terms are identical to the conventional EDA
    for key in ('e_kin', 'e_ne', 'e_coul', 'e_x', 'pop'):
        assert numpy.allclose(getattr(res, key), getattr(conv, key), atol=1e-12)
    assert 'Mulliken-EDA' in res.summary()


def test_mulliken_xc_partition_matches_direct_evaluation(h2o_b3lyp5_cart):
    # E_XC^A = sum_g w_g eps(g) rho_A(g),  rho_A = sum_{mu in A} chi_mu (chi P)_mu
    from pyscf.dft import numint
    mf = h2o_b3lyp5_cart
    mol = mf.mol
    dm = mf.make_rdm1()
    ni = mf._numint
    grids = mf.grids
    ao = numint.eval_ao(mol, grids.coords, deriv=1)
    rho = ni.eval_rho(mol, ao, dm, xctype='GGA')
    exc = ni.eval_xc_eff('B3LYP5', rho, deriv=0, xctype='GGA', spin=0)[0]
    d = ao[0] @ dm
    e_mu = numpy.einsum('g,gm,gm->m', grids.weights * exc, ao[0], d)
    ref = [e_mu[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()]
    res = eda_rhf.xc_energy_by_atom(mf, 'B3LYP5', dm, partition='mulliken')
    assert numpy.allclose(res, ref, atol=1e-8)
    # the exact-exchange-free part sums to the functional energy
    assert abs(res.sum() - numpy.dot(grids.weights * exc, rho[0])) < 1e-8


def test_grid_eda_sum_rules(h2o_b3lyp5_cart):
    mf = h2o_b3lyp5_cart
    mol = mf.mol
    dm = mf.make_rdm1()
    res = eda_rhf.grid_eda(mf)
    assert res.scheme == 'grid'
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-5            # quadrature accuracy
    assert abs(res.pop.sum() - mol.nelectron) < 1e-6
    t = numpy.einsum('ij,ji', dm, mol.intor_symmetric('int1e_kin'))
    v = numpy.einsum('ij,ji', dm, mol.intor_symmetric('int1e_nuc'))
    ej = 0.5 * numpy.einsum('ij,ji', dm, mf.get_j(mol, dm))
    ek = -0.25 * 0.2 * numpy.einsum('ij,ji', dm, mf.get_k(mol, dm))
    assert abs(res.e_kin.sum() - t) < 1e-5
    assert abs(res.e_ne.sum() - v) < 1e-5
    assert abs(res.e_coul.sum() - ej) < 1e-6
    assert abs(res.e_x.sum() - ek) < 1e-6
    # Becke populations differ from Mulliken ones, symmetric hydrogens equal
    assert abs(res.pop[1] - res.pop[2]) < 1e-8
    mull = eda_rhf.kernel(mf, orbital_basis='ao')
    assert abs(res.pop[0] - mull.pop[0]) > 0.05
    assert 'Grid-EDA' in res.summary()


def test_grid_eda_hartree_fock():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    mf.grids = eda_rhf.becke_grids(mol, level=5)
    mf.grids.build(with_non0tab=True)
    res = eda_rhf.grid_eda(mf)
    assert res.e_xc is None
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-5


def test_grid_partition_ne_halves(h2o_b3lyp5_cart):
    # the two halves of Eq. (24) sum to the same total V_ne
    mf = h2o_b3lyp5_cart
    dm = mf.make_rdm1()
    g = eda_rhf.grid_partition(mf, dm)
    v = numpy.einsum('ij,ji', dm, mf.mol.intor_symmetric('int1e_nuc'))
    assert abs(g['ne'].sum() - v) < 1e-5
    assert abs(g['exch'].sum() + 0.25 * numpy.einsum('ij,ji', dm, mf.get_k(mf.mol, dm))) < 1e-6


def test_hf_molecule_against_table3():
    # HF molecule, Table III of Kikuchi et al. (2009): populations and atomic energies
    mol = gto.M(atom='F 0 0 0; H 0 0 0.9337', basis='6-31g(d,p)', cart=True, verbose=0)
    mf = dft.RKS(mol, xc='B3LYP5')
    mf.grids = eda_rhf.becke_grids(mf, level=5)
    mf.run(conv_tol=1e-10)
    r_mull = eda_rhf.mulliken_eda(mf, orbital_basis='ao')
    r_grid = eda_rhf.grid_eda(mf)
    r_conv = eda_rhf.kernel(mf, orbital_basis='ao')
    # Mulliken-EDA reproduces the paper to the printed digits; the Becke
    # partition of GAMESS differs slightly from PySCF's (Bragg radii), which
    # shows up in the grid population and in the E_XC partition of the
    # conventional EDA (~0.02 hartree for F).
    assert abs(r_mull.pop[0] - 9.36) < 0.01 and abs(r_grid.pop[0] - 9.07) < 0.06
    assert abs(r_mull.e_tot[0] - (-99.898)) < 2e-3 and abs(r_mull.e_tot[1] - (-0.491)) < 2e-3
    assert abs(r_grid.e_tot[0] - (-99.764)) < 5e-3 and abs(r_grid.e_tot[1] - (-0.625)) < 5e-3
    assert abs(r_conv.e_tot[0] - (-99.827)) < 3e-2 and abs(r_conv.e_tot[1] - (-0.563)) < 3e-2
