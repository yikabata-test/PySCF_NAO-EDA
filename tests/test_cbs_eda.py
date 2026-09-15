import numpy
import pytest
from pyscf import gto, scf, mp, cc

from pyscf_eda import cbs as eda_cbs
from pyscf_eda import ccsd as eda_ccsd
from pyscf_eda import ccsd_t as eda_ccsd_t
from pyscf_eda import mp2 as eda_mp2

HF_ATOM = 'F 0 0 0; H 0 0 0.917'


@pytest.fixture(scope='module')
def hf_mol():
    return gto.M(atom=HF_ATOM, basis='sto-3g', verbose=0)


def test_scheme_tables_are_consistent():
    for scheme, families in eda_cbs.SCHEMES.items():
        for family, coeff in families.items():
            keys = set(coeff)
            assert {('MP2', 'D'), ('MP2', 'T'), ('MP2', 'Q')} <= keys
            total = sum(coeff.values())
            assert 0.75 < total < 1.25, (scheme, family, total)
    assert set(eda_cbs.SCHEMES['QDD']['cc-pv']) == {
        ('MP2', 'D'), ('MP2', 'T'), ('MP2', 'Q'), ('CCSD', 'D'), ('CCSD(T)', 'D')}
    assert set(eda_cbs.SCHEMES['QTD']['cc-pv']) == {
        ('MP2', 'D'), ('MP2', 'T'), ('MP2', 'Q'), ('CCSD', 'D'), ('CCSD', 'T'), ('CCSD(T)', 'D')}


def test_required_levels(hf_mol):
    assert eda_cbs.QDD(hf_mol).required_levels() == {'D': 'CCSD(T)', 'T': 'MP2', 'Q': 'MP2'}
    assert eda_cbs.QTD(hf_mol).required_levels() == {'D': 'CCSD(T)', 'T': 'CCSD', 'Q': 'MP2'}
    assert eda_cbs.QTT(hf_mol).required_levels() == {'D': 'CCSD(T)', 'T': 'CCSD(T)', 'Q': 'MP2'}
    assert eda_cbs.QTN(hf_mol).required_levels() == {'D': 'CCSD', 'T': 'CCSD', 'Q': 'MP2'}


def test_hf_cbs_coefficients():
    c = eda_cbs.hf_cbs_coefficients('largest', ['D', 'T', 'Q'])
    assert c == {'Q': 1.0}
    for method in ('karton-martin', 'halkier'):
        c = eda_cbs.hf_cbs_coefficients(method, ['D', 'T', 'Q'])
        assert abs(sum(c.values()) - 1.0) < 1e-12
        assert c['Q'] > 1.0 > 0.0 > c['T']
    with pytest.raises(ValueError):
        eda_cbs.hf_cbs_coefficients('no-such-scheme', ['T', 'Q'])


@pytest.fixture(scope='module')
def qtd_result(hf_mol):
    return eda_cbs.QTD(hf_mol, verbose=0).kernel()


def test_qtd_runs_needed_levels_only(qtd_result):
    res = qtd_result
    assert set(res.corr) == set(res.coeff)
    assert set(res.hf) == {'D', 'T', 'Q'}
    assert res.frozen == 'auto'
    # frozen core: F 1s
    assert res.eda_results[('MP2', 'Q')].hf_result is not None


def test_qtd_linear_combination_matches_independent_calculations(qtd_result, hf_mol):
    res = qtd_result
    coeff = eda_cbs.SCHEMES['QTD']['cc-pv']
    e_mol = {}
    for x, basis in eda_cbs.BASIS_FAMILIES['cc-pv'].items():
        mol = gto.M(atom=HF_ATOM, basis=basis, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        e_mol[('MP2', x)] = mp.MP2(mf, frozen=1).run().e_corr
        if x in ('D', 'T'):
            mycc = cc.CCSD(mf, frozen=1).run(conv_tol=1e-9, conv_tol_normt=1e-7)
            e_mol[('CCSD', x)] = mycc.e_corr
            if x == 'D':
                e_mol[('CCSD(T)', x)] = mycc.e_corr + mycc.ccsd_t()
        assert abs(res.hf_mol[x] - mf.e_tot) < 1e-8
    e_ref = sum(c * e_mol[k] for k, c in coeff.items())
    assert abs(res.e_corr_mol - e_ref) < 1e-7
    for key in coeff:
        assert abs(res.corr_mol[key] - e_mol[key]) < 1e-7
        assert abs(res.corr[key].sum() - e_mol[key]) < 1e-7
    # atomic sums reproduce the molecular linear combinations
    assert abs(res.e_corr.sum() - res.e_corr_mol) < 1e-8
    assert abs(res.e_tot.sum() - res.e_tot_mol) < 1e-8
    assert abs(res.e_hf.sum() - res.e_hf_mol) < 1e-8
    assert res.hf_cbs == 'halkier'
    # atomic combination equals coefficient-weighted atomic energies
    manual = sum(c * res.corr[k] for k, c in coeff.items())
    assert numpy.allclose(res.e_corr, manual, atol=1e-12)


def test_qtd_uses_underlying_eda_results(qtd_result):
    res = qtd_result
    r_t = res.eda_results[('CCSD(T)', 'D')]
    assert isinstance(r_t, eda_ccsd_t.CCSDTEDAResult)
    assert numpy.allclose(res.corr[('CCSD(T)', 'D')], r_t.e_corr, atol=1e-12)
    assert numpy.allclose(res.corr[('CCSD', 'D')], r_t.e_ccsd, atol=1e-12)
    assert isinstance(res.eda_results[('CCSD', 'T')], eda_ccsd.CCSDEDAResult)
    assert isinstance(res.eda_results[('MP2', 'Q')], eda_mp2.MP2EDAResult)
    # CBS correlation energy should be more negative than any single MP2/CCSD value
    assert res.e_corr_mol < res.corr_mol[('CCSD(T)', 'D')]
    assert res.e_corr_mol < res.corr_mol[('MP2', 'Q')]


def test_qdd_and_karton_martin(hf_mol):
    res = eda_cbs.QDD(hf_mol, hf_cbs='karton-martin', verbose=0).kernel()
    assert set(res.corr) == {('MP2', 'D'), ('MP2', 'T'), ('MP2', 'Q'), ('CCSD', 'D'), ('CCSD(T)', 'D')}
    assert ('CCSD', 'T') not in res.eda_results
    assert abs(res.e_tot.sum() - res.e_tot_mol) < 1e-8
    c = res.hf_coeff
    e_hf_ref = c['Q'] * res.hf_mol['Q'] + c['T'] * res.hf_mol['T']
    assert abs(res.e_hf_mol - e_hf_ref) < 1e-10
    assert abs(res.e_hf.sum() - e_hf_ref) < 1e-8
    # extrapolated HF lies below HF/QZ
    assert res.e_hf_mol < res.hf_mol['Q']


def test_nao_basis_and_explicit_frozen(hf_mol):
    res = eda_cbs.CompositeEDA(hf_mol, scheme='QTN', orbital_basis='nao', frozen=1,
                               verbose=0).kernel()
    assert abs(res.e_tot.sum() - res.e_tot_mol) < 1e-8
    assert res.eda_results[('MP2', 'Q')].orbital_basis == 'nao'
    assert abs(res.eda_results[('MP2', 'Q')].pop.sum() - 10) < 1e-8
    res_ao = eda_cbs.CompositeEDA(hf_mol, scheme='QTN', orbital_basis='ao', frozen=1,
                                  verbose=0).kernel()
    assert res_ao.eda_results[('MP2', 'Q')].orbital_basis == 'ao'
    assert abs(res_ao.e_tot_mol - res.e_tot_mol) < 1e-8      # molecular values agree
    assert not numpy.allclose(res_ao.e_tot, res.e_tot)        # atomic partitions differ


def test_bad_inputs(hf_mol):
    with pytest.raises(ValueError):
        eda_cbs.CompositeEDA(hf_mol, scheme='XYZ')
    with pytest.raises(ValueError):
        eda_cbs.CompositeEDA(hf_mol, basis_family='def2')
    with pytest.raises(ValueError):
        eda_cbs.CompositeEDA(hf_mol, basis_family={'D': 'cc-pvdz', 'T': 'cc-pvtz', 'Q': 'cc-pvqz'})


def test_summary(qtd_result):
    text = qtd_result.summary()
    assert 'QTD' in text and 'E_corr (CBS)' in text and 'CCSD(T)[DZ]' in text


# ---------------------------------------------------------------------------
# HF energies at the CBS limit
# ---------------------------------------------------------------------------

def test_hf_cbs_estimate_linear_schemes_are_consistent():
    rng = numpy.random.default_rng(1)
    atoms = {x: rng.standard_normal(3) for x in ('D', 'T', 'Q')}
    mol = {x: float(v.sum()) for x, v in atoms.items()}
    for method in ('largest', 'karton-martin', 'halkier'):
        est = eda_cbs.hf_cbs_estimate(method, atoms)
        assert abs(est.sum() - eda_cbs.hf_cbs_estimate(method, mol)) < 1e-12
        c = eda_cbs.hf_cbs_coefficients(method, ['D', 'T', 'Q'])
        assert abs(sum(c.values()) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        eda_cbs.hf_cbs_coefficients('feller', ['D', 'T', 'Q'])


def test_feller_formula_and_alpha():
    # exact exponential series E(X) = E_CBS + A exp(-alpha X) is recovered
    e_cbs, a, alpha = -100.0, 0.5, 1.3
    e = {x: e_cbs + a * numpy.exp(-alpha * eda_cbs.CARDINAL[x]) for x in ('D', 'T', 'Q')}
    assert abs(eda_cbs.hf_cbs_estimate('feller', e) - e_cbs) < 1e-10
    assert abs(eda_cbs.feller_alpha(e) - alpha) < 1e-10
    # halkier with the exact alpha is exact as well
    assert abs(eda_cbs.hf_cbs_estimate('halkier', e, alpha=alpha) - e_cbs) < 1e-10
    # non-monotonic sequence: alpha undefined (nan)
    e_bad = {'D': -1.0, 'T': -1.2, 'Q': -1.1}
    assert numpy.isnan(eda_cbs.feller_alpha(e_bad))
    # element-wise on arrays
    arr = {x: numpy.array([e[x], e_bad[x]]) for x in e}
    alpha_arr = eda_cbs.feller_alpha(arr)
    assert abs(alpha_arr[0] - alpha) < 1e-10 and numpy.isnan(alpha_arr[1])


@pytest.fixture(scope='module')
def hf_cbs_result(hf_mol):
    return eda_cbs.HFCBS(hf_mol, verbose=0).kernel()


def test_hf_cbs_driver(hf_cbs_result, hf_mol):
    res = hf_cbs_result
    assert set(res.hf) == {'D', 'T', 'Q'}
    for x, basis in eda_cbs.BASIS_FAMILIES['cc-pv'].items():
        mf = scf.RHF(gto.M(atom=HF_ATOM, basis=basis, verbose=0)).run(conv_tol=1e-10)
        assert abs(res.hf_mol[x] - mf.e_tot) < 1e-8
        assert abs(res.hf[x].sum() - mf.e_tot) < 1e-8
    est = res.estimates
    assert set(est) >= {'largest', 'karton-martin', 'halkier', 'feller', 'feller_alpha'}
    for method in ('largest', 'karton-martin', 'halkier'):
        assert abs(est[method]['sum_error']) < 1e-8
    # linear extrapolations lie below HF/QZ (HF converges from above)
    assert est['halkier']['mol'] < res.hf_mol['Q']
    assert est['karton-martin']['mol'] < res.hf_mol['Q']
    # feller: molecular value sensible, atomic sum generally not equal to it
    assert est['feller']['mol'] < res.hf_mol['Q']
    assert numpy.isfinite(est['feller']['mol'])
    text = res.summary()
    assert 'HF/CBS feller' in text and 'feller alpha_A' in text
    assert res.eda_results['Q'].orbital_basis == 'nao'


def test_composite_hf_cbs_schemes(hf_mol):
    for method in ('largest', 'karton-martin', 'halkier', 'feller'):
        res = eda_cbs.QTN(hf_mol, hf_cbs=method, verbose=0).kernel()
        est = res.hf_estimates[method]
        assert numpy.allclose(res.e_hf, est['atoms'], equal_nan=True)
        assert abs(res.e_hf_mol - est['mol']) < 1e-12
        if method != 'feller':
            assert abs(res.e_tot.sum() - res.e_tot_mol) < 1e-8
            assert res.hf_coeff is not None
        else:
            assert res.hf_coeff is None
            assert abs(res.e_tot.sum() - res.e_tot_mol - res.hf_sum_error) < 1e-8
        assert 'feller_alpha' in res.hf_estimates
    with pytest.raises(ValueError):
        eda_cbs.QTN(hf_mol, hf_cbs='no-such-scheme')


# ---------------------------------------------------------------------------
# analytic gradients of the CBS energy
# ---------------------------------------------------------------------------

def test_ccsd_t_gradient_uses_t_lambda():
    from pyscf_eda import grad as eda_grad
    from pyscf.grad import ccsd_t as ccsd_t_grad
    B = 0.52917721092
    def run(z):
        m = gto.M(atom=f'F 0 0 0; H 0 0 {z}', basis='cc-pvdz', verbose=0)
        f = scf.RHF(m).run(conv_tol=1e-12)
        c = cc.CCSD(f, frozen=1)
        c.conv_tol, c.conv_tol_normt = 1e-11, 1e-9
        return c.run()
    c0 = run(0.917)
    g = eda_grad.ccsd_t_gradient(c0)
    h = 1e-3
    cp, cm = run(0.917 + h), run(0.917 - h)
    fd = ((cp.e_tot + cp.ccsd_t()) - (cm.e_tot + cm.ccsd_t())) / (2 * h) * B
    assert abs(g[1, 2] - fd) < 1e-5
    # the plain PySCF call without lambda amplitudes is *not* the CCSD(T) gradient
    g_wrong = ccsd_t_grad.Gradients(c0).kernel()
    assert abs(g_wrong[1, 2] - fd) > 1e-4
    assert abs(g.sum(axis=0)).max() < 1e-6   # translational invariance


def test_hf_cbs_gradient_feller_chain_rule():
    from pyscf_eda import grad as eda_grad
    rng = numpy.random.default_rng(3)
    e = {'D': -1.0, 'T': -1.2, 'Q': -1.25}
    g = {x: rng.standard_normal((2, 3)) for x in e}
    # finite difference along a random direction of the energies
    d = {x: rng.standard_normal() for x in e}
    h = 1e-6
    ep = {x: e[x] + h * d[x] for x in e}
    em = {x: e[x] - h * d[x] for x in e}
    fd = (eda_cbs.hf_cbs_estimate('feller', ep) - eda_cbs.hf_cbs_estimate('feller', em)) / (2 * h)
    gr = eda_grad.hf_cbs_gradient('feller', e, g)
    # directional derivative: sum_x d[x] * dE/dE_x ; compare via linearity in g
    gr_dir = eda_grad.hf_cbs_gradient('feller', e, {x: numpy.full((2, 3), d[x]) for x in e})
    assert abs(gr_dir[0, 0] - fd) < 1e-6
    for method in ('largest', 'halkier', 'karton-martin'):
        gl = eda_grad.hf_cbs_gradient(method, e, g)
        c = eda_cbs.hf_cbs_coefficients(method, list(e))
        assert numpy.allclose(gl, sum(c[x] * g[x] for x in c))


@pytest.fixture(scope='module')
def h2_qtd_grad():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    return eda_cbs.QTD(mol, with_grad=True, verbose=0).kernel()


def test_cbs_gradient_against_finite_difference(h2_qtd_grad):
    res = h2_qtd_grad
    assert res.grad.shape == (2, 3)
    assert abs(res.grad.sum(axis=0)).max() < 1e-6
    assert set(res.grads) == {('MP2', 'D'), ('MP2', 'T'), ('MP2', 'Q'), ('CCSD', 'D'),
                              ('CCSD', 'T'), ('CCSD(T)', 'D')}
    B = 0.52917721092
    h = 2e-3
    e = {}
    for dz in (h, -h):
        mol = gto.M(atom=f'H 0 0 0; H 0 0 {0.74 + dz}', basis='sto-3g', verbose=0)
        e[dz] = eda_cbs.QTD(mol, verbose=0).kernel().e_tot_mol
    fd = (e[h] - e[-h]) / (2 * h) * B
    assert abs(res.grad[1, 2] - fd) < 2e-5
    # HF part with the Feller formula (nonlinear) also differentiates correctly
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    res_f = eda_cbs.QTD(mol, with_grad=True, hf_cbs='feller', verbose=0).kernel()
    ef = {}
    for dz in (h, -h):
        m = gto.M(atom=f'H 0 0 0; H 0 0 {0.74 + dz}', basis='sto-3g', verbose=0)
        ef[dz] = eda_cbs.QTD(m, hf_cbs='feller', verbose=0).kernel().e_tot_mol
    fd_f = (ef[h] - ef[-h]) / (2 * h) * B
    assert abs(res_f.grad[1, 2] - fd_f) < 2e-5
    assert 'Gradient of E_QTD(CBS)' in res.summary()


def test_run_scf_newton_reaches_tight_thresholds():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586', basis='cc-pvdz', verbose=0)
    ref = scf.RHF(mol)
    ref.conv_tol, ref.conv_tol_grad = 1e-14, 1e-9
    ref.run()
    g_ref = ref.nuc_grad_method().kernel()
    for newton in (True, False):
        mf = eda_cbs.run_scf(mol, conv_tol=1e-10, conv_tol_grad=1e-7, newton=newton)
        assert type(mf) is scf.hf.RHF          # SOSCF wrapper removed
        assert mf.converged
        assert mf.conv_tol_grad == 1e-7
        assert numpy.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) < 1e-7
        assert abs(mf.e_tot - ref.e_tot) < 1e-11
        g = mf.nuc_grad_method().kernel()
        assert abs(g - g_ref).max() < 1e-8


def test_composite_newton_option(hf_mol):
    kw = dict(scheme='QDD', with_grad=True, verbose=0)
    drv = eda_cbs.CompositeEDA(hf_mol, newton=True, **kw)
    res_n = drv.kernel()
    res_d = eda_cbs.CompositeEDA(hf_mol, newton=False, **kw).kernel()
    assert all(mf.converged for mf in drv.mfs.values())
    assert all(numpy.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) < 1e-7
               for mf in drv.mfs.values())
    assert numpy.allclose(res_n.e_tot, res_d.e_tot, atol=1e-9)
    assert numpy.allclose(res_n.grad, res_d.grad, atol=1e-8)
    hf_n = eda_cbs.HFCBS(hf_mol, cardinals=('D', 'T'), newton=True, conv_tol_grad=1e-8,
                         verbose=0).kernel()
    hf_d = eda_cbs.HFCBS(hf_mol, cardinals=('D', 'T'), newton=False, verbose=0).kernel()
    assert numpy.allclose(hf_n.hf_mol['T'], hf_d.hf_mol['T'], atol=1e-9)


def test_total_energies_for_every_hf_scheme(qtd_result):
    res = qtd_result
    assert set(res.e_tot_by_hf) == {m for m in eda_cbs.HF_CBS_METHODS if m in res.hf_estimates}
    assert numpy.allclose(res.e_tot_by_hf[res.hf_cbs], res.e_tot)
    for m, e in res.e_tot_by_hf.items():
        assert numpy.allclose(e, res.hf_estimates[m]['atoms'] + res.e_corr)
        assert abs(res.e_tot_mol_by_hf[m] - (res.hf_estimates[m]['mol'] + res.e_corr_mol)) < 1e-12
        assert f'E_QTD {m}' in res.summary()


def test_gradients_for_every_hf_scheme(h2_qtd_grad):
    res = h2_qtd_grad
    assert set(res.grad_by_hf) == set(res.e_tot_by_hf)
    assert numpy.allclose(res.grad_by_hf[res.hf_cbs], res.grad)
    for m in ('largest', 'karton-martin'):
        expect = sum(c * res.hf_grads[x] for x, c in
                     eda_cbs.hf_cbs_coefficients(m, list(res.hf)).items()) + res.grad_corr
        assert numpy.allclose(res.grad_by_hf[m], expect)
