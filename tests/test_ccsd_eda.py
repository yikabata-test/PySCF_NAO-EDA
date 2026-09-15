import numpy
import pytest
from pyscf import gto, scf, cc, ao2mo

from pyscf_eda import rhf as eda_rhf
from pyscf_eda import ccsd as eda_ccsd
from pyscf_eda import mp2 as eda_mp2

H2O_ATOM = '''
O  0.000000  0.000000  0.000000
H  0.000000  0.757193  0.586080
H  0.000000 -0.757193  0.586080
'''


@pytest.fixture(scope='module')
def h2o_ccsd():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    return mycc.run()


@pytest.fixture(scope='module')
def h2o_ccsd_frozen():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf, frozen=1)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    return mycc.run()


def _reference_partition(mycc, tau, x=None):
    """Eq. (10) evaluated independently through MO-basis atomic projectors."""
    mol = mycc.mol
    mf = mycc._scf
    c_all = mf.mo_coeff
    c_act = c_all[:, mycc.get_frozen_mask()]
    nocc = mycc.nocc
    c_occ, c_vir = c_act[:, :nocc], c_act[:, nocc:]
    nvir = c_vir.shape[1]
    s = mol.intor_symmetric('int1e_ovlp')
    if x is None:
        x = numpy.eye(mol.nao)
    xinv = numpy.linalg.inv(x)
    left = x.T @ s @ c_all
    tbar = 2 * tau - tau.transpose(0, 1, 3, 2)
    nmo = c_all.shape[1]
    g_pajb = ao2mo.general(mf._eri, (c_all, c_vir, c_occ, c_vir), compact=False)
    g_pajb = g_pajb.reshape(nmo, nvir, nocc, nvir)
    g_ipjb = ao2mo.general(mf._eri, (c_occ, c_all, c_occ, c_vir), compact=False)
    g_ipjb = g_ipjb.reshape(nocc, nmo, nocc, nvir)
    e_occ = numpy.zeros(mol.natm)
    e_vir = numpy.zeros(mol.natm)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        q = (xinv @ c_occ)[p0:p1].T @ left[p0:p1]
        e_occ[ia] = numpy.einsum('ip,pajb,ijab->', q, g_pajb, tbar)
        q = (xinv @ c_vir)[p0:p1].T @ left[p0:p1]
        e_vir[ia] = numpy.einsum('ap,ipjb,ijab->', q, g_ipjb, tbar)
    return e_occ, e_vir


def test_effective_amplitudes_reproduce_ccsd_energy(h2o_ccsd):
    # sanity check of tau = t2 + t1 t1 against the closed-shell energy formula
    mycc = h2o_ccsd
    mf = mycc._scf
    nocc = mycc.nocc
    c = mf.mo_coeff
    ovov = ao2mo.general(mf._eri, (c[:, :nocc], c[:, nocc:], c[:, :nocc], c[:, nocc:]),
                         compact=False)
    nvir = c.shape[1] - nocc
    ovov = ovov.reshape(nocc, nvir, nocc, nvir)
    tau = eda_ccsd.effective_amplitudes(mycc.t1, mycc.t2)
    e = 2 * numpy.einsum('iajb,ijab', ovov, tau) - numpy.einsum('iajb,ijba', ovov, tau)
    assert abs(e - mycc.e_corr) < 1e-9
    # the antisymmetrized form of Eq. (8) does NOT reproduce the closed-shell energy
    tau_anti = tau - numpy.einsum('ib,ja->ijab', mycc.t1, mycc.t1)
    e_anti = 2 * numpy.einsum('iajb,ijab', ovov, tau_anti) - numpy.einsum('iajb,ijba', ovov, tau_anti)
    assert abs(e_anti - mycc.e_corr) > 1e-6


@pytest.mark.parametrize('orbital_basis', ('ao', 'nao', 'lso'))
@pytest.mark.parametrize('w_occ', (1.0, 0.0, 0.5))
def test_sum_rules(h2o_ccsd, orbital_basis, w_occ):
    mycc = h2o_ccsd
    res = eda_ccsd.kernel(mycc, orbital_basis=orbital_basis, w_occ=w_occ)
    assert res.method == 'CCSD'
    assert abs(res.e_corr_occ.sum() - mycc.e_corr) < 1e-9
    assert abs(res.e_corr_vir.sum() - mycc.e_corr) < 1e-9
    assert abs(res.e_corr.sum() - mycc.e_corr) < 1e-9
    assert abs(res.e_hf.sum() - mycc._scf.e_tot) < 1e-9
    assert abs(res.e_tot.sum() - mycc.e_tot) < 1e-9
    assert abs(res.e_corr[1] - res.e_corr[2]) < 1e-9
    hf = eda_rhf.kernel(mycc._scf, orbital_basis=orbital_basis)
    assert numpy.allclose(res.e_hf, hf.e_tot, atol=1e-10)


def test_against_mo_projector_reference(h2o_ccsd):
    mycc = h2o_ccsd
    tau = eda_ccsd.effective_amplitudes(mycc.t1, mycc.t2)
    for x in (None, eda_ccsd.nao_eda(mycc).orth_coeff):
        e_occ, e_vir = eda_ccsd.corr_energy_by_atom(mycc, x=x)
        ref_occ, ref_vir = _reference_partition(mycc, tau, x=x)
        assert numpy.allclose(e_occ, ref_occ, atol=1e-8)
        assert numpy.allclose(e_vir, ref_vir, atol=1e-8)
        # the singles term sums to zero for canonical HF orbitals
        s_occ, s_vir = eda_ccsd.corr_energy_by_atom(mycc, x=x, with_singles=True)
        assert abs(s_occ.sum() - e_occ.sum()) < 1e-8
        assert abs(s_vir.sum() - e_vir.sum()) < 1e-8


def test_frozen_core(h2o_ccsd_frozen):
    mycc = h2o_ccsd_frozen
    res = eda_ccsd.kernel(mycc)
    assert abs(res.e_corr.sum() - mycc.e_corr) < 1e-9
    assert abs(res.e_tot.sum() - mycc.e_tot) < 1e-9
    tau = eda_ccsd.effective_amplitudes(mycc.t1, mycc.t2)
    ref_occ, ref_vir = _reference_partition(mycc, tau)
    assert numpy.allclose(res.e_corr_occ, ref_occ, atol=1e-8)
    assert numpy.allclose(res.e_corr_vir, ref_vir, atol=1e-8)


def test_singles_partition_sum_rule(h2o_ccsd):
    # with a perturbed (non-canonical) t1 the singles term is non-zero and
    # both partitions must still sum to 2 sum f_ia t_ia
    from pyscf_eda import corr
    mycc = h2o_ccsd
    mol = mycc.mol
    c_occ, c_vir = corr.active_orbitals(mycc)
    fock_ao = mycc._scf.get_fock()
    rng = numpy.random.default_rng(0)
    t1 = rng.standard_normal(mycc.t1.shape)
    fock_ao = fock_ao + 0.05 * rng.standard_normal(fock_ao.shape)
    fock_ao = 0.5 * (fock_ao + fock_ao.T)
    f_ia = c_occ.T @ fock_ao @ c_vir
    ref = 2 * numpy.einsum('ia,ia', f_ia, t1)
    for x in (None, eda_ccsd.nao_eda(mycc).orth_coeff):
        s_occ, s_vir = corr.partition_singles(mol, fock_ao, c_occ, c_vir, t1, x=x)
        assert abs(s_occ.sum() - ref) < 1e-10
        assert abs(s_vir.sum() - ref) < 1e-10


def test_mp2_vs_ccsd_partition_consistency(h2o_ccsd):
    # using MP2 amplitudes as t2 with t1 = 0 in the CCSD machinery must give
    # the MP2-EDA partition
    from pyscf import mp
    mycc = h2o_ccsd
    pt = mp.MP2(mycc._scf).run()
    res_mp2 = eda_mp2.kernel(pt)
    t1 = numpy.zeros_like(mycc.t1)
    e_occ, e_vir = eda_ccsd.corr_energy_by_atom(mycc, t1=t1, t2=pt.t2)
    assert numpy.allclose(e_occ, res_mp2.e_corr_occ, atol=1e-9)
    assert numpy.allclose(e_vir, res_mp2.e_corr_vir, atol=1e-9)


def test_correlation_energy_mostly_on_oxygen(h2o_ccsd):
    res = eda_ccsd.kernel(h2o_ccsd)
    assert numpy.all(res.e_corr < 0)
    assert abs(res.e_corr[0]) > 0.5 * abs(res.e_corr.sum())


def test_isolated_fragments_are_additive():
    h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    cc_h2 = cc.CCSD(scf.RHF(h2).run(conv_tol=1e-12)).run(conv_tol=1e-10, conv_tol_normt=1e-8)
    res_h2 = eda_ccsd.kernel(cc_h2)
    dimer = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 200; H 0 0 200.74',
                  basis='cc-pvdz', verbose=0)
    cc_dimer = cc.CCSD(scf.RHF(dimer).run(conv_tol=1e-12)).run(conv_tol=1e-10, conv_tol_normt=1e-8)
    res_dimer = eda_ccsd.kernel(cc_dimer)
    assert numpy.allclose(res_dimer.e_corr[:2], res_h2.e_corr, atol=1e-6)
    assert numpy.allclose(res_dimer.e_corr[2:], res_h2.e_corr, atol=1e-6)


def test_density_fitted_ccsd():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-11)
    mycc = cc.CCSD(mf).run(conv_tol=1e-9, conv_tol_normt=1e-7)
    assert hasattr(mycc, 'with_df')
    res = eda_ccsd.kernel(mycc)
    assert abs(res.e_corr.sum() - mycc.e_corr) < 1e-7
    assert abs(res.e_tot.sum() - mycc.e_tot) < 1e-7


def test_runs_ccsd_if_needed():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf)
    res = eda_ccsd.kernel(mycc)
    assert mycc.e_corr is not None
    assert abs(res.e_tot.sum() - mycc.e_tot) < 1e-8


def test_rejects_uccsd_and_mp2():
    from pyscf import mp
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.UHF(mol).run()
    with pytest.raises(TypeError):
        eda_ccsd.EDA(cc.UCCSD(mf))
    mfr = scf.RHF(mol).run()
    with pytest.raises(TypeError):
        eda_ccsd.EDA(mp.MP2(mfr))


def test_summary(h2o_ccsd):
    text = eda_ccsd.kernel(h2o_ccsd).summary()
    assert 'E_CCSD' in text and 'CCSD total energy' in text and 'CCSD, w_occ=1' in text
