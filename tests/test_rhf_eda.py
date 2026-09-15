import numpy
import pytest
from pyscf import gto, scf

from pyscf_eda import rhf as eda_rhf

# Experimental H2O geometry used in Nakai, CPL 363, 73 (2002):
# R(O-H) = 0.9575 A, angle(H-O-H) = 104.51 deg
H2O_ATOM = '''
O  0.000000  0.000000  0.000000
H  0.000000  0.757193  0.586080
H  0.000000 -0.757193  0.586080
'''


@pytest.fixture(scope='module')
def h2o_rhf():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return mf


@pytest.mark.parametrize('ne_partition', eda_rhf.NE_PARTITIONS)
def test_sum_rule_each_component(h2o_rhf, ne_partition):
    mf = h2o_rhf
    mol = mf.mol
    dm = mf.make_rdm1()
    res = eda_rhf.EDA(mf, ne_partition=ne_partition).kernel()

    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    vj, vk = mf.get_jk(mol, dm)
    assert abs(res.e_nn.sum() - mol.energy_nuc()) < 1e-10
    assert abs(res.e_kin.sum() - numpy.einsum('ij,ji', dm, t)) < 1e-10
    assert abs(res.e_ne.sum() - numpy.einsum('ij,ji', dm, v)) < 1e-10
    assert abs(res.e_coul.sum() - 0.5 * numpy.einsum('ij,ji', dm, vj)) < 1e-10
    assert abs(res.e_x.sum() + 0.25 * numpy.einsum('ij,ji', dm, vk)) < 1e-10
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-9
    assert abs(res.e_elec.sum() - (mf.e_tot - mol.energy_nuc())) < 1e-9


def test_symmetry_equivalent_hydrogens(h2o_rhf):
    res = eda_rhf.kernel(h2o_rhf)
    assert res.orbital_basis == 'nao'   # NAO-EDA is the default
    for key in res.components:
        vals = getattr(res, key)
        assert abs(vals[1] - vals[2]) < 1e-8, key


def test_signs_and_ordering(h2o_rhf):
    # Same qualitative behaviour as Table 1 of the paper:
    # E_NN, T_S, E_CLB > 0 ; E_Ne, E_X < 0 ; |E_1EL| > E_CLB > E_NN and E_CLB > |E_X|
    res = eda_rhf.kernel(h2o_rhf, orbital_basis='ao')
    for ia in range(res.mol.natm):
        assert res.e_nn[ia] > 0
        assert res.e_kin[ia] > 0
        assert res.e_coul[ia] > 0
        assert res.e_ne[ia] < 0
        assert res.e_x[ia] < 0
        assert abs(res.e_1el[ia]) > res.e_coul[ia] > res.e_nn[ia]
        assert res.e_coul[ia] > abs(res.e_x[ia])
    # oxygen carries most of the energy
    assert res.e_tot[0] < -70 and -0.7 < res.e_tot[1] < -0.3


def test_half_partition_is_mean_of_mulliken_and_nuclear(h2o_rhf):
    r_half = eda_rhf.kernel(h2o_rhf, ne_partition='half', orbital_basis='ao')
    r_mul = eda_rhf.kernel(h2o_rhf, ne_partition='mulliken', orbital_basis='ao')
    r_nuc = eda_rhf.kernel(h2o_rhf, ne_partition='nuclear', orbital_basis='ao')
    assert numpy.allclose(r_half.e_ne, 0.5 * (r_mul.e_ne + r_nuc.e_ne), atol=1e-10)
    # the other components are independent of the scheme
    for key in ('e_nn', 'e_kin', 'e_coul', 'e_x'):
        assert numpy.allclose(getattr(r_mul, key), getattr(r_nuc, key), atol=1e-12)


def test_nuc_attraction_split_matches_int1e_nuc(h2o_rhf):
    mol = h2o_rhf.mol
    v_atoms = eda_rhf.nuc_attraction_by_nucleus(mol)
    assert numpy.allclose(v_atoms.sum(axis=0), mol.intor_symmetric('int1e_nuc'), atol=1e-12)


def test_isolated_fragments_are_additive():
    # Two H2 molecules far apart: the atomic energies of each H2 unit must
    # equal those of an isolated H2 to high accuracy.
    h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    mf_h2 = scf.RHF(h2).run(conv_tol=1e-12)
    res_h2 = eda_rhf.kernel(mf_h2)

    dimer = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 200; H 0 0 200.74',
                  basis='cc-pvdz', verbose=0)
    mf_dimer = scf.RHF(dimer).run(conv_tol=1e-12)
    res_dimer = eda_rhf.kernel(mf_dimer)
    assert numpy.allclose(res_dimer.e_tot[:2], res_h2.e_tot, atol=1e-6)
    assert numpy.allclose(res_dimer.e_tot[2:], res_h2.e_tot, atol=1e-6)


def test_ecp_molecule_sum_rule():
    mol = gto.M(atom='I 0 0 0; H 0 0 1.61', basis='def2-svp', ecp='def2-svp', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    res = eda_rhf.kernel(mf)
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-8
    v_ecp = eda_rhf.ecp_by_atom(mol)
    assert v_ecp is not None
    assert numpy.allclose(v_ecp.sum(axis=0), mol.intor('ECPscalar'), atol=1e-12)


def test_density_fitting_sum_rule():
    mol = gto.M(atom=H2O_ATOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-11)
    res = eda_rhf.kernel(mf)
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-8


def test_ghost_atom():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74; ghost-H 0 0 3.0', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    res = eda_rhf.kernel(mf)
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-9
    assert res.e_nn[2] == 0.0


def test_rejects_open_shell():
    from pyscf import dft
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    with pytest.raises(TypeError):
        eda_rhf.EDA(scf.UHF(mol))
    with pytest.raises((TypeError, NotImplementedError)):
        eda_rhf.EDA(dft.UKS(mol))
    with pytest.raises(NotImplementedError):
        eda_rhf.EDA(scf.ROHF(mol))


def test_summary_output(h2o_rhf):
    res = eda_rhf.kernel(h2o_rhf)
    text = res.summary()
    assert 'E_TOT' in text and 'O0' in text and 'H1' in text
    assert 'Difference' in text


# ---------------------------------------------------------------------------
# NAO-EDA / LSO-EDA (Baba, Takeuchi, Nakai, CPL 424, 193 (2006))
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('orbital_basis', ('nao', 'nao_pyscf', 'lso', 'lowdin', 'meta_lowdin'))
def test_orthogonal_basis_sum_rules(h2o_rhf, orbital_basis):
    mf = h2o_rhf
    mol = mf.mol
    dm = mf.make_rdm1()
    res = eda_rhf.kernel(mf, orbital_basis=orbital_basis)
    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    vj, vk = mf.get_jk(mol, dm)
    assert abs(res.e_kin.sum() - numpy.einsum('ij,ji', dm, t)) < 1e-10
    assert abs(res.e_ne.sum() - numpy.einsum('ij,ji', dm, v)) < 1e-10
    assert abs(res.e_coul.sum() - 0.5 * numpy.einsum('ij,ji', dm, vj)) < 1e-10
    assert abs(res.e_x.sum() + 0.25 * numpy.einsum('ij,ji', dm, vk)) < 1e-10
    assert abs(res.e_tot.sum() - mf.e_tot) < 1e-9
    assert abs(res.pop.sum() - mol.nelectron) < 1e-10
    assert abs(res.e_tot[1] - res.e_tot[2]) < 1e-8
    # transformation matrix is S-orthonormal
    s = mol.intor_symmetric('int1e_ovlp')
    assert numpy.allclose(res.orth_coeff.T @ s @ res.orth_coeff, numpy.eye(mol.nao), atol=1e-8)


def test_ao_basis_is_conventional_eda(h2o_rhf):
    # orbital_basis='ao' (default) must be the plain Mulliken-type partition
    mf = h2o_rhf
    mol = mf.mol
    dm = mf.make_rdm1()
    res = eda_rhf.kernel(mf, orbital_basis='ao')
    assert res.orbital_basis == 'ao'
    t = mol.intor_symmetric('int1e_kin')
    diag = numpy.einsum('ij,ji->i', dm, t)
    ref = [diag[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()]
    assert numpy.allclose(res.e_kin, ref, atol=1e-12)
    # populations are Mulliken populations
    pop = scf.hf.mulliken_pop(mol, dm, verbose=0)[0]
    ref_pop = [pop[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()]
    assert numpy.allclose(res.pop, ref_pop, atol=1e-10)


def test_nao_pyscf_populations_match_pyscf_npa(h2o_rhf):
    from pyscf.lo import orth as pyscf_orth
    mf = h2o_rhf
    mol = mf.mol
    dm = mf.make_rdm1()
    s = mol.intor_symmetric('int1e_ovlp')
    c = pyscf_orth.orth_ao(mf, method='nao', s=s)
    p_nao = c.T @ s @ dm @ s @ c
    ref = [p_nao.diagonal()[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()]
    res = eda_rhf.kernel(mf, orbital_basis='nao_pyscf')
    assert numpy.allclose(res.pop, ref, atol=1e-8)
    assert numpy.allclose(abs(res.orth_coeff), abs(c), atol=1e-8)


def test_nao_reed_weinhold_properties(h2o_rhf):
    from pyscf_eda import orth
    from pyscf.lo import nao as pyscf_nao
    mf = h2o_rhf
    mol = mf.mol
    dm = mf.make_rdm1()
    s = mol.intor_symmetric('int1e_ovlp')
    x = orth.nao_coeff(mol, dm, s)
    assert numpy.allclose(x.T @ s @ x, numpy.eye(mol.nao), atol=1e-10)
    # the NMB (core+valence) block carries almost all electrons
    core, val, ryd = pyscf_nao._core_val_ryd_list(mol)
    p_nao = x.T @ s @ dm @ s @ x
    occ = numpy.einsum('ii->i', p_nao)
    assert occ[core + val].sum() > 9.9
    assert occ[ryd].sum() < 0.1
    res = eda_rhf.nao_eda(mf)
    assert numpy.allclose(res.pop, [occ[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()], atol=1e-8)


def test_lso_populations_are_lowdin_populations(h2o_rhf):
    from pyscf.lo import orth as pyscf_orth
    mf = h2o_rhf
    mol = mf.mol
    dm = mf.make_rdm1()
    s = mol.intor_symmetric('int1e_ovlp')
    s_half = numpy.linalg.inv(pyscf_orth.lowdin(s))     # S^{1/2}
    p_lowdin = s_half @ dm @ s_half
    ref = [p_lowdin.diagonal()[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()]
    res = eda_rhf.lso_eda(mf)
    assert numpy.allclose(res.pop, ref, atol=1e-8)


def test_nao_eda_formula_eq6_eq7(h2o_rhf):
    # E_KIN^{NAO,A} = sum_{l in A} {(X^-1 P X^-1^T)(X^T T X)}_ll   (Eqs. 6, 7)
    mf = h2o_rhf
    mol = mf.mol
    dm = mf.make_rdm1()
    res = eda_rhf.nao_eda(mf)
    x = res.orth_coeff
    xinv = numpy.linalg.inv(x)
    p_nao = xinv @ dm @ xinv.T
    t_nao = x.T @ mol.intor_symmetric('int1e_kin') @ x
    diag = numpy.einsum('ij,ji->i', p_nao, t_nao)
    ref = [diag[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()]
    assert numpy.allclose(res.e_kin, ref, atol=1e-10)


def test_custom_transformation_matrix(h2o_rhf):
    from pyscf.lo import orth as pyscf_orth
    mol = h2o_rhf.mol
    x = pyscf_orth.lowdin(mol.intor_symmetric('int1e_ovlp'))
    res_custom = eda_rhf.kernel(h2o_rhf, orbital_basis=x)
    res_lso = eda_rhf.lso_eda(h2o_rhf)
    assert res_custom.orbital_basis == 'custom'
    assert numpy.allclose(res_custom.e_tot, res_lso.e_tot, atol=1e-10)
    with pytest.raises(ValueError):
        eda_rhf.kernel(h2o_rhf, orbital_basis='no-such-basis')


def test_nao_eda_weak_basis_set_dependence():
    # C atom in CO2 (R = 1.16 A): the NAO-EDA energy fraction should be
    # nearly basis-set independent (cf. Table 2 of the 2006 paper).
    fractions = {'ao': [], 'lso': [], 'nao': []}
    for basis in ('6-31g', 'cc-pvdz', 'aug-cc-pvdz'):
        mol = gto.M(atom='C 0 0 0; O 0 0 1.16; O 0 0 -1.16', basis=basis, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        for key in fractions:
            res = eda_rhf.kernel(mf, orbital_basis=key)
            assert abs(res.e_tot.sum() - mf.e_tot) < 1e-8
            fractions[key].append(100 * res.e_tot[0] / mf.e_tot)
    nao = numpy.array(fractions['nao'])
    assert nao.max() - nao.min() < 0.15
    assert 20.6 < nao.mean() < 21.1
    # LSO-EDA shows the known outlier with diffuse functions
    lso = numpy.array(fractions['lso'])
    assert lso.max() - lso.min() > nao.max() - nao.min()


def test_single_atom_gets_total_energy():
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    for key in ('ao', 'lso', 'nao'):
        res = eda_rhf.kernel(mf, orbital_basis=key)
        assert abs(res.e_tot[0] - mf.e_tot) < 1e-9
        assert abs(res.pop[0] - 10) < 1e-10


def test_nao_rydberg_weight_variants(h2o_rhf):
    from pyscf_eda import orth
    mf = h2o_rhf
    res = {}
    for name in ('nao', 'nao:pre', 'nao:post', 'nao:lowdin'):
        r = eda_rhf.kernel(mf, orbital_basis=name)
        assert abs(r.e_tot.sum() - mf.e_tot) < 1e-9
        assert abs(r.pop.sum() - mf.mol.nelectron) < 1e-9
        res[name] = r
    assert numpy.allclose(res['nao'].e_tot, res['nao:pre'].e_tot, atol=1e-12)
    assert not numpy.allclose(res['nao:pre'].e_tot, res['nao:post'].e_tot, atol=1e-4)
    assert not numpy.allclose(res['nao:pre'].e_tot, res['nao:lowdin'].e_tot, atol=1e-4)
    with pytest.raises(ValueError):
        orth.nao_coeff(mf.mol, mf.make_rdm1(), nrb_weights='no-such')
