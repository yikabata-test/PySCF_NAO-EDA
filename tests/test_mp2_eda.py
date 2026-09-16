import numpy
import pytest
from pyscf import gto, scf, mp, ao2mo

from pyscf_eda import rhf as eda_rhf
from pyscf_eda import mp2 as eda_mp2

H2O_ATOM = '''
O  0.000000  0.000000  0.000000
H  0.000000  0.757193  0.586080
H  0.000000 -0.757193  0.586080
'''


@pytest.fixture(scope='module')
def h2o_mp2():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return mp.MP2(mf).run()


@pytest.fixture(scope='module')
def h2o_mp2_frozen():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return mp.MP2(mf, frozen=1).run()


@pytest.mark.parametrize('orbital_basis', ('ao', 'nao', 'lso'))
@pytest.mark.parametrize('w_occ', (1.0, 0.0, 0.5))
def test_sum_rules(h2o_mp2, orbital_basis, w_occ):
    pt = h2o_mp2
    res = eda_mp2.kernel(pt, orbital_basis=orbital_basis, w_occ=w_occ)
    assert abs(res.e_corr_occ.sum() - pt.e_corr) < 1e-9
    assert abs(res.e_corr_vir.sum() - pt.e_corr) < 1e-9
    assert abs(res.e_corr.sum() - pt.e_corr) < 1e-9
    assert abs(res.e_hf.sum() - pt._scf.e_tot) < 1e-9
    assert abs(res.e_tot.sum() - pt.e_tot) < 1e-9
    # symmetry-equivalent hydrogens
    assert abs(res.e_corr[1] - res.e_corr[2]) < 1e-9
    assert abs(res.e_tot[1] - res.e_tot[2]) < 1e-9
    # HF part identical to the standalone RHF-EDA
    hf = eda_rhf.kernel(pt._scf, orbital_basis=orbital_basis)
    assert numpy.allclose(res.e_hf, hf.e_tot, atol=1e-10)
    assert numpy.allclose(res.pop, hf.pop, atol=1e-10)


def test_w_occ_interpolation(h2o_mp2):
    r1 = eda_mp2.kernel(h2o_mp2, w_occ=1.0, orbital_basis='ao')
    r0 = eda_mp2.kernel(h2o_mp2, w_occ=0.0, orbital_basis='ao')
    rh = eda_mp2.kernel(h2o_mp2, w_occ=0.5, orbital_basis='ao')
    assert numpy.allclose(r1.e_corr, r1.e_corr_occ, atol=1e-12)
    assert numpy.allclose(r0.e_corr, r0.e_corr_vir, atol=1e-12)
    assert numpy.allclose(rh.e_corr, 0.5 * (r1.e_corr_occ + r0.e_corr_vir), atol=1e-12)
    with pytest.raises(ValueError):
        eda_mp2.kernel(h2o_mp2, w_occ=1.5)


def _reference_partition(pt, x=None):
    """Independent evaluation of Eq. (18) through MO-basis atomic projectors.

    (mu a|jb) = sum_p (S C)_{mu p} (pa|jb) when the MOs span the AO space, so
    sum_{l in A} C'_{li} (l a|jb) = sum_p Q^A_{ip} (pa|jb) with
    Q^A_{ip} = sum_{l in A} (X^-1 C)_{li} (X^T S C)_{lp}.
    """
    mol = pt.mol
    mf = pt._scf
    mask = pt.get_frozen_mask()
    c_all = mf.mo_coeff                       # all MOs span the AO space
    c_act = c_all[:, mask]
    nocc = pt.nocc
    c_occ, c_vir = c_act[:, :nocc], c_act[:, nocc:]
    nvir = c_vir.shape[1]
    s = mol.intor_symmetric('int1e_ovlp')
    if x is None:
        x = numpy.eye(mol.nao)
    xinv = numpy.linalg.inv(x)
    left = x.T @ s @ c_all                    # (l, p)
    t2 = pt.t2
    tau = 2 * t2 - t2.transpose(0, 1, 3, 2)
    # (pa|jb) and (ip|jb) with p over all MOs
    nmo = c_all.shape[1]
    g_pajb = ao2mo.general(mf._eri, (c_all, c_vir, c_occ, c_vir), compact=False)
    g_pajb = g_pajb.reshape(nmo, nvir, nocc, nvir)
    g_ipjb = ao2mo.general(mf._eri, (c_occ, c_all, c_occ, c_vir), compact=False)
    g_ipjb = g_ipjb.reshape(nocc, nmo, nocc, nvir)
    e_occ = numpy.zeros(mol.natm)
    e_vir = numpy.zeros(mol.natm)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        q = (xinv @ c_occ)[p0:p1].T @ left[p0:p1]        # (i, p)
        e_occ[ia] = numpy.einsum('ip,pajb,ijab->', q, g_pajb, tau)
        q = (xinv @ c_vir)[p0:p1].T @ left[p0:p1]        # (a, p)
        e_vir[ia] = numpy.einsum('ap,ipjb,ijab->', q, g_ipjb, tau)
    return e_occ, e_vir


def test_against_mo_projector_reference(h2o_mp2):
    e_occ, e_vir = eda_mp2.corr_energy_by_atom(h2o_mp2)
    ref_occ, ref_vir = _reference_partition(h2o_mp2)
    assert numpy.allclose(e_occ, ref_occ, atol=1e-9)
    assert numpy.allclose(e_vir, ref_vir, atol=1e-9)


def test_against_mo_projector_reference_nao(h2o_mp2):
    res = eda_mp2.nao_eda(h2o_mp2)
    ref_occ, ref_vir = _reference_partition(h2o_mp2, x=res.orth_coeff)
    assert numpy.allclose(res.e_corr_occ, ref_occ, atol=1e-9)
    assert numpy.allclose(res.e_corr_vir, ref_vir, atol=1e-9)


def test_frozen_core(h2o_mp2_frozen):
    pt = h2o_mp2_frozen
    res = eda_mp2.kernel(pt, orbital_basis='ao')
    assert abs(res.e_corr.sum() - pt.e_corr) < 1e-9
    assert abs(res.e_tot.sum() - pt.e_tot) < 1e-9
    e_occ, e_vir = _reference_partition(pt)
    assert numpy.allclose(res.e_corr_occ, e_occ, atol=1e-9)
    assert numpy.allclose(res.e_corr_vir, e_vir, atol=1e-9)


def test_correlation_energy_is_negative_and_mostly_on_oxygen(h2o_mp2):
    res = eda_mp2.kernel(h2o_mp2)
    assert numpy.all(res.e_corr < 0)
    assert res.e_corr[0] < res.e_corr[1]
    assert abs(res.e_corr[0]) > 0.5 * abs(res.e_corr.sum())


def test_isolated_fragments_are_additive():
    h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    pt_h2 = mp.MP2(scf.RHF(h2).run(conv_tol=1e-12)).run()
    res_h2 = eda_mp2.kernel(pt_h2)
    dimer = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 200; H 0 0 200.74',
                  basis='cc-pvdz', verbose=0)
    pt_dimer = mp.MP2(scf.RHF(dimer).run(conv_tol=1e-12)).run()
    res_dimer = eda_mp2.kernel(pt_dimer)
    assert numpy.allclose(res_dimer.e_corr[:2], res_h2.e_corr, atol=1e-6)
    assert numpy.allclose(res_dimer.e_corr[2:], res_h2.e_corr, atol=1e-6)
    assert numpy.allclose(res_dimer.e_tot[:2], res_h2.e_tot, atol=1e-6)


def test_density_fitted_mp2():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-11)
    pt = mp.MP2(mf).run()
    res = eda_mp2.kernel(pt)
    assert abs(res.e_corr.sum() - pt.e_corr) < 1e-8
    assert abs(res.e_tot.sum() - pt.e_tot) < 1e-8


def test_outcore_integrals():
    # no cached AO integrals -> transformation from the Mole object
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    pt = mp.MP2(mf).run()
    ref = eda_mp2.kernel(pt, orbital_basis='ao').e_corr
    mf._eri = None
    res = eda_mp2.kernel(pt, orbital_basis='ao')
    assert numpy.allclose(res.e_corr, ref, atol=1e-9)


def test_runs_mp2_if_needed():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    pt = mp.MP2(mf)
    res = eda_mp2.kernel(pt)
    assert pt.e_corr is not None
    assert abs(res.e_tot.sum() - pt.e_tot) < 1e-9


def test_rejects_ump2():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.UHF(mol).run()
    with pytest.raises(TypeError):
        eda_mp2.EDA(mp.UMP2(mf))


def test_summary(h2o_mp2):
    text = eda_mp2.kernel(h2o_mp2).summary()
    assert 'E_HF' in text and 'E_corr' in text and 'E_MP2' in text and 'MP2, w_occ=1' in text


def test_partition_doubles_matches_per_atom_reference():
    """Blocked single transformation == per-atom transformation of the 2008 paper's formula."""
    from pyscf import lib
    from pyscf_eda import corr, orth
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    pt = mp.MP2(mf, frozen=1).run()
    c_occ, c_vir = corr.active_orbitals(pt)
    transform = corr.eri_transformer(pt)
    tbar = 2.0 * pt.t2 - pt.t2.transpose(0, 1, 3, 2)
    nocc, nvir = c_occ.shape[1], c_vir.shape[1]
    for ob in ('ao', 'nao'):
        x = numpy.eye(mol.nao) if ob == 'ao' else orth.orth_coeff(mol, ob, mf.make_rdm1(), mf.get_ovlp())
        xinv = numpy.linalg.inv(x)
        cp_occ, cp_vir = xinv @ c_occ, xinv @ c_vir
        ref_occ, ref_vir = numpy.zeros(mol.natm), numpy.zeros(mol.natm)
        for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
            x_a = x[:, p0:p1]
            eri = transform((x_a, c_vir, c_occ, c_vir)).reshape(p1 - p0, nvir, nocc, nvir)
            ref_occ[ia] = numpy.einsum('li,lajb,ijab->', cp_occ[p0:p1], eri, tbar)
            eri = transform((c_occ, x_a, c_occ, c_vir)).reshape(nocc, p1 - p0, nocc, nvir)
            ref_vir[ia] = numpy.einsum('la,iljb,ijab->', cp_vir[p0:p1], eri, tbar)
        for max_memory in (None, lib.current_memory()[0] + 1):   # one block / one orbital per block
            e_occ, e_vir = corr.partition_doubles(mol, transform, c_occ, c_vir, pt.t2, x,
                                                  max_memory=max_memory)
            assert numpy.allclose(e_occ, ref_occ, atol=1e-12)
            assert numpy.allclose(e_vir, ref_vir, atol=1e-12)
            assert abs(e_occ.sum() - pt.e_corr) < 1e-10 and abs(e_vir.sum() - pt.e_corr) < 1e-10


def test_outcore_transformation_matches_incore():
    """Out-of-core path (single HDF5 transformation, blocks read back) == in-core path."""
    from pyscf import lib
    from pyscf_eda import corr, ccsd_t as eda_ccsd_t
    from pyscf import cc
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    pt = mp.MP2(mf, frozen=1).run()
    ref = eda_mp2.EDA(pt).kernel()
    mycc = cc.CCSD(mf, frozen=1); mycc.conv_tol = 1e-10; mycc.conv_tol_normt = 1e-8; mycc.run()
    ref_t = eda_ccsd_t.triples_by_atom(mycc)
    mf._eri = None                     # force the out-of-core transformation
    tr = corr.eri_transformer(pt)
    assert getattr(tr, 'outcore', False)
    for max_memory in (None, lib.current_memory()[0] + 1):
        c_occ, c_vir = corr.active_orbitals(pt)
        e_occ, e_vir = corr.partition_doubles(mol, tr, c_occ, c_vir, pt.t2, ref.orth_coeff,
                                              max_memory=max_memory)
        assert numpy.allclose(e_occ, ref.e_corr_occ, atol=1e-12)
        assert numpy.allclose(e_vir, ref.e_corr_vir, atol=1e-12)
    res_t = eda_ccsd_t.triples_by_atom(mycc, max_memory=lib.current_memory()[0] + 0.05)
    for name in ('occ', 'vir'):
        assert numpy.allclose(res_t[name][0], ref_t[name][0], atol=1e-12)
