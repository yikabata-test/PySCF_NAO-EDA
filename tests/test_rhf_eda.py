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
    for key in res.components:
        vals = getattr(res, key)
        assert abs(vals[1] - vals[2]) < 1e-8, key


def test_signs_and_ordering(h2o_rhf):
    # Same qualitative behaviour as Table 1 of the paper:
    # E_NN, T_S, E_CLB > 0 ; E_Ne, E_X < 0 ; |E_1EL| > E_CLB > E_NN and E_CLB > |E_X|
    res = eda_rhf.kernel(h2o_rhf)
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
    r_half = eda_rhf.kernel(h2o_rhf, ne_partition='half')
    r_mul = eda_rhf.kernel(h2o_rhf, ne_partition='mulliken')
    r_nuc = eda_rhf.kernel(h2o_rhf, ne_partition='nuclear')
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


def test_rejects_uhf_and_dft():
    from pyscf import dft
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    with pytest.raises(TypeError):
        eda_rhf.EDA(scf.UHF(mol))
    with pytest.raises(NotImplementedError):
        eda_rhf.EDA(dft.RKS(mol))


def test_summary_output(h2o_rhf):
    res = eda_rhf.kernel(h2o_rhf)
    text = res.summary()
    assert 'E_TOT' in text and 'O0' in text and 'H1' in text
    assert 'Difference' in text
