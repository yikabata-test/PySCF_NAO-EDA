import numpy
import pytest
from pyscf import gto, scf, cc, ao2mo, lib

from pyscf_eda import ccsd as eda_ccsd
from pyscf_eda import ccsd_t as eda_ccsd_t

H2O_ATOM = '''
O  0.000000  0.000000  0.000000
H  0.000000  0.757193  0.586080
H  0.000000 -0.757193  0.586080
'''


def _run_ccsd(mol, frozen=None):
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    return mycc.run()


@pytest.fixture(scope='module')
def h2o_ccsd():
    return _run_ccsd(gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0), frozen=1)


@pytest.fixture(scope='module')
def h2o_ccsd_all():
    return _run_ccsd(gto.M(atom=H2O_ATOM, basis='6-31g', verbose=0))


def _full_tensors(mycc, x=None):
    """Full W, Z, D and per-atom raw R^A tensors (small molecules only)."""
    mol = mycc.mol
    mf = mycc._scf
    mask = mycc.get_frozen_mask()
    c = mf.mo_coeff[:, mask]
    e = mf.mo_energy[mask]
    nocc = mycc.nocc
    co, cv = c[:, :nocc], c[:, nocc:]
    nvir = cv.shape[1]
    t1, t2 = mycc.t1, mycc.t2
    eri = mf._eri
    if x is None:
        x = numpy.eye(mol.nao)
    xinv = numpy.linalg.inv(x)
    cpo, cpv = xinv @ co, xinv @ cv
    ovov = ao2mo.general(eri, (co, cv, co, cv), compact=False).reshape(nocc, nvir, nocc, nvir)
    g_full = ao2mo.general(eri, (cv, cv, cv, x), compact=False).reshape(nvir, nvir, nvir, mol.nao)
    h_full = ao2mo.general(eri, (co, co, cv, x), compact=False).reshape(nocc, nocc, nvir, mol.nao)
    g = numpy.einsum('becl,lk->beck', g_full, cpo)
    h = numpy.einsum('mjcl,lk->mjck', h_full, cpo)
    r = numpy.einsum('ijae,beck->ijkabc', t2, g) - numpy.einsum('imab,mjck->ijkabc', t2, h)
    perms = [(0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4), (1, 0, 2, 4, 3, 5),
             (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4), (2, 1, 0, 5, 4, 3)]
    w = sum(r.transpose(p) for p in perms)
    z = (numpy.einsum('ia,jbkc->ijkabc', t1, ovov) + numpy.einsum('jb,iakc->ijkabc', t1, ovov)
         + numpy.einsum('kc,iajb->ijkabc', t1, ovov))
    d = lib.direct_sum('i+j+k-a-b-c->ijkabc', e[:nocc], e[:nocc], e[:nocc],
                       e[nocc:], e[nocc:], e[nocc:])
    r_atoms = []
    for _, _, p0, p1 in mol.aoslice_by_atom():
        ga = numpy.einsum('becl,lk->beck', g_full[:, :, :, p0:p1], cpo[p0:p1])
        ha = numpy.einsum('mjcl,lk->mjck', h_full[:, :, :, p0:p1], cpo[p0:p1])
        r_atoms.append(numpy.einsum('ijae,beck->ijkabc', t2, ga)
                       - numpy.einsum('imab,mjck->ijkabc', t2, ha))
    return w, z, d, r_atoms


def _r3(w):
    return (4 * w + w.transpose(1, 2, 0, 3, 4, 5) + w.transpose(2, 0, 1, 3, 4, 5)
            - 2 * w.transpose(2, 1, 0, 3, 4, 5) - 2 * w.transpose(0, 2, 1, 3, 4, 5)
            - 2 * w.transpose(1, 0, 2, 3, 4, 5))


def test_paper_energy_expressions_match_pyscf(h2o_ccsd):
    mycc = h2o_ccsd
    et_ref = mycc.ccsd_t()
    w, z, d, _ = _full_tensors(mycc)
    t = w + z
    # Eqs. (9) + (10): sum [4/3 T_abc - 2 T_acb + 2/3 T_bca] W / D
    y = 4 / 3 * t - 2 * t.transpose(0, 1, 2, 3, 5, 4) + 2 / 3 * t.transpose(0, 1, 2, 4, 5, 3)
    e_paper = numpy.einsum('ijkabc,ijkabc', y / d, w)
    assert abs(e_paper - et_ref) < 1e-9
    # symmetrized form used in the implementation
    e_impl = numpy.einsum('ijkabc,ijkabc', _r3(t) / d, w) / 3
    assert abs(e_impl - et_ref) < 1e-9


def test_w_block_matches_full_tensor(h2o_ccsd):
    from pyscf_eda import corr
    mycc = h2o_ccsd
    w_full, _, _, _ = _full_tensors(mycc)
    c_occ, c_vir = corr.active_orbitals(mycc)
    x = numpy.eye(mycc.mol.nao)
    xinv = x
    g_atoms, h_atoms = eda_ccsd_t._partitioned_integrals(
        mycc.mol, corr.eri_transformer(mycc), c_occ, c_vir, x, xinv @ c_occ, xinv @ c_vir, 'occ')
    g = sum(g_atoms)
    h = sum(h_atoms)
    for a in (0, 3, c_vir.shape[1] - 1):
        w_blk = eda_ccsd_t._w_block(mycc.t2, g, h, a)
        assert numpy.allclose(w_blk, w_full[:, :, :, a], atol=1e-12)


def test_triples_partition_against_full_tensor(h2o_ccsd):
    mycc = h2o_ccsd
    et_ref = mycc.ccsd_t()
    for x in (None, eda_ccsd.nao_eda(mycc).orth_coeff):
        w, z, d, r_atoms = _full_tensors(mycc, x=x)
        y4 = 2 * _r3(w) / d
        y5 = 2 * _r3(z) / d
        ref_t = numpy.array([numpy.einsum('ijkabc,ijkabc', y4 + y5, r) for r in r_atoms])
        ref_t4 = numpy.array([numpy.einsum('ijkabc,ijkabc', y4, r) for r in r_atoms])
        parts = eda_ccsd_t.triples_by_atom(mycc, x=x)
        e_t, e_t4 = parts['occ']
        assert numpy.allclose(e_t, ref_t, atol=1e-10)
        assert numpy.allclose(e_t4, ref_t4, atol=1e-10)
        assert abs(e_t.sum() - et_ref) < 1e-9
        assert abs(parts['vir'][0].sum() - et_ref) < 1e-9


@pytest.mark.parametrize('orbital_basis', ('ao', 'nao', 'lso'))
@pytest.mark.parametrize('w_occ', (1.0, 0.0, 0.5))
def test_sum_rules(h2o_ccsd, orbital_basis, w_occ):
    mycc = h2o_ccsd
    et_ref = mycc.ccsd_t()
    res = eda_ccsd_t.kernel(mycc, orbital_basis=orbital_basis, w_occ=w_occ)
    assert res.method == 'CCSD(T)'
    assert abs(res.e_ccsd.sum() - mycc.e_corr) < 1e-9
    assert abs(res.e_t.sum() - et_ref) < 1e-9
    assert abs(res.e_t4.sum() + res.e_st5.sum() - et_ref) < 1e-9
    assert abs(res.e_corr.sum() - mycc.e_corr - et_ref) < 1e-9
    assert abs(res.e_tot.sum() - mycc.e_tot - et_ref) < 1e-9
    assert abs(res.e_t[1] - res.e_t[2]) < 1e-9
    # CCSD part identical to the standalone CCSD-EDA
    res_ccsd = eda_ccsd.kernel(mycc, orbital_basis=orbital_basis, w_occ=w_occ)
    assert numpy.allclose(res.e_ccsd, res_ccsd.e_corr, atol=1e-10)
    assert numpy.allclose(res.e_hf, res_ccsd.e_hf, atol=1e-10)


def test_all_electron_and_signs(h2o_ccsd_all):
    mycc = h2o_ccsd_all
    et_ref = mycc.ccsd_t()
    res = eda_ccsd_t.kernel(mycc)
    assert abs(res.e_t.sum() - et_ref) < 1e-9
    # (T) is negative overall; E_T[4] < 0 and E_ST[5] > 0 in total
    assert res.e_t.sum() < 0
    assert res.e_t4.sum() < 0 < res.e_st5.sum()
    assert abs(res.e_t[0]) > abs(res.e_t[1])


def test_isolated_fragments_are_additive():
    h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    res_h2 = eda_ccsd_t.kernel(_run_ccsd(h2))
    dimer = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 200; H 0 0 200.74',
                  basis='cc-pvdz', verbose=0)
    res_dimer = eda_ccsd_t.kernel(_run_ccsd(dimer))
    assert numpy.allclose(res_dimer.e_t[:2], res_h2.e_t, atol=1e-7)
    assert numpy.allclose(res_dimer.e_t[2:], res_h2.e_t, atol=1e-7)
    assert numpy.allclose(res_dimer.e_tot[:2], res_h2.e_tot, atol=1e-6)


def test_density_fitted_ccsd_t():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-11)
    mycc = cc.CCSD(mf, frozen=1).run(conv_tol=1e-10, conv_tol_normt=1e-8)
    et_ref = mycc.ccsd_t()
    res = eda_ccsd_t.kernel(mycc)
    assert abs(res.e_t.sum() - et_ref) < 1e-7
    assert abs(res.e_tot.sum() - mycc.e_tot - et_ref) < 1e-7


def test_summary(h2o_ccsd):
    text = eda_ccsd_t.kernel(h2o_ccsd).summary()
    assert 'E_(T)' in text and 'E_corr(CCSD)' in text and 'E_CCSD(T)' in text
    assert 'CCSD(T) total energy' in text
