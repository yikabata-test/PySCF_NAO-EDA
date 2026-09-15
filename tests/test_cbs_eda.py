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
    c = eda_cbs.hf_cbs_coefficients('karton-martin', ['D', 'T', 'Q'])
    assert abs(sum(c.values()) - 1.0) < 1e-12
    assert c['Q'] > 1.0 > 0.0 > c['T']
    with pytest.raises(ValueError):
        eda_cbs.hf_cbs_coefficients('feller', ['T', 'Q'])


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
    assert abs(res.e_hf.sum() - res.hf_mol['Q']) < 1e-8
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
