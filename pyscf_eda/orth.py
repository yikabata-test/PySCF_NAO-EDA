"""Orthogonal atomic-orbital bases for EDA (NAO-EDA and LSO-EDA).

Reference
---------
T. Baba, M. Takeuchi, H. Nakai, "Natural atomic orbital based energy density
analysis: Implementation and applications",
Chem. Phys. Lett. 424, 193-198 (2006).

An orthonormal one-centre basis {phi_l} is written in terms of the AOs
{chi_m} with a transformation matrix X (Eq. 5 of the paper)

    phi_l = sum_m X_{m l} chi_m ,

so that X^dagger S X = 1.  Because every column of X is centred on a single
atom, the assignment of orbitals to atoms is the same as for the AOs
(``mol.aoslice_by_atom()``).

Available bases
---------------
'ao'          : the raw (non-orthogonal) AO basis, X = 1 -> conventional EDA
'nao'         : natural atomic orbitals following the procedure of Reed,
                Weinstock and Weinhold, J. Chem. Phys. 83, 735 (1985):
                pre-NAOs from the atomic blocks of the density (spherically
                averaged per angular momentum), occupancy-weighted symmetric
                orthogonalization (OWSO) of the natural minimal basis (NMB:
                core + valence) of all atoms together, Schmidt
                orthogonalization of the Rydberg set (NRB) to the NMB and
                OWSO of the NRB.  Reproduces the NAO-EDA numbers of Baba et
                al. (2006) for CO2 to a few mhartree.
'nao:post', 'nao:lowdin' : 'nao' with the other Rydberg-set weightings
                (see ``nao_coeff``); 'nao' is 'nao:pre'
'nao_pyscf'   : the NAO variant of ``pyscf.lo.nao`` (Loewdin for the core
                set, OWSO for the valence set with the core projected out,
                Loewdin for the Rydberg set).  Kept for reference; it differs
                from 'nao' by ~0.4 hartree in the C-atom energy of CO2.
'lso'/'lowdin': Loewdin symmetrically orthogonalized AOs, X = S^{-1/2}
'meta_lowdin' : Loewdin orthogonalization within the core, valence and
                Rydberg spaces separately (PySCF default for population
                analysis; not part of the 2006 paper)
"""

import numpy
from pyscf import lib
from pyscf.lo import orth as pyscf_orth
from pyscf.lo import nao as pyscf_nao

ORBITAL_BASES = ('ao', 'nao', 'nao_pyscf', 'lso', 'lowdin', 'meta_lowdin')
OWSO_MIN_WEIGHT = 1e-4


def orth_coeff(mol, method='nao', dm=None, s=None):
    """Transformation matrix X (AO -> orthogonal atomic basis).

    Parameters
    ----------
    mol : pyscf.gto.Mole
    method : str or ndarray
        One of ``ORBITAL_BASES`` or an explicit (nao, nao) matrix X.
    dm : ndarray
        Spin-summed AO density matrix; required for ``method='nao'``.
    s : ndarray, optional
        AO overlap matrix.

    Returns
    -------
    ndarray of shape (nao, nao)
    """
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    nao = s.shape[0]

    if isinstance(method, numpy.ndarray):
        x = numpy.asarray(method, dtype=float)
        if x.shape != (nao, nao):
            raise ValueError(f'orthogonalization matrix must have shape {(nao, nao)}')
        return x

    method = method.lower()
    if method == 'ao':
        return numpy.eye(nao)
    if method in ('lso', 'lowdin'):
        # plain S^{-1/2} (no projection onto a reference AO basis)
        return pyscf_orth.lowdin(s)
    if method in ('nao', 'nao_pyscf') or method.startswith('nao:'):
        if dm is None:
            raise ValueError(f"a density matrix is required for method='{method}'")
        if method == 'nao_pyscf':
            return nao_coeff_pyscf(mol, dm, s)
        nrb = method.split(':', 1)[1] if ':' in method else 'pre'
        return nao_coeff(mol, dm, s, nrb_weights=nrb)
    if method == 'meta_lowdin':
        return pyscf_orth.orth_ao(mol, method='meta_lowdin', s=s)
    raise ValueError(f"unknown orbital basis '{method}'; choose from {ORBITAL_BASES} "
                     'or pass a transformation matrix')


def _spin_summed(dm):
    dm = numpy.asarray(dm)
    if dm.ndim == 3:  # (alpha, beta)
        dm = dm[0] + dm[1]
    return dm


def _fix_phase(c):
    for i in range(c.shape[1]):
        if c[i, i] < 0:
            c[:, i] *= -1
    return c


NRB_WEIGHTS = ('pre', 'post', 'lowdin')


def nao_coeff(mol, dm, s=None, min_weight=OWSO_MIN_WEIGHT, nrb_weights='pre'):
    """NAO transformation matrix (Reed-Weinstock-Weinhold procedure).

    ``nrb_weights`` selects the OWSO weights of the Rydberg set (NRB) after
    its Schmidt orthogonalization to the NMB, a detail that the NAO-EDA
    papers do not specify and that changes the NAO-EDA atomic energies by
    up to ~0.2 hartree with polarized/diffuse basis sets:
    'pre' (default): pre-NAO occupancies; 'post': diagonal occupancies of
    the Schmidt-projected NRB functions; 'lowdin': equal weights.

    Steps
    -----
    1. pre-NAOs: diagonalize the spherically averaged atomic blocks of the
       density operator S P S for every atom and angular momentum
       (``pyscf.lo.nao._prenao_sub``); their eigenvalues are the pre-NAO
       occupancies.
    2. Partition into the natural minimal basis (NMB, core + valence shells
       of the ground-state configuration) and the Rydberg set (NRB).
    3. OWSO of all NMB pre-NAOs together, with the occupancies as weights
       (weights below ``min_weight`` are raised to it).
    4. Schmidt orthogonalization of the NRB pre-NAOs to the NMB set, then
       OWSO of the NRB set with their occupancies as weights.
    5. Restoration of the natural character (re-diagonalization of the
       atomic blocks; skipped for Cartesian functions as in PySCF).  This
       step is a rotation within each atom and does not affect the EDA.
    """
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    dm = _spin_summed(dm)
    p = lib.dot(lib.dot(s, dm), s)
    pre_occ, pre_nao = pyscf_nao._prenao_sub(mol, p, s)
    core_lst, val_lst, ryd_lst = pyscf_nao._core_val_ryd_list(mol)
    nmb = sorted(core_lst + val_lst)
    nao = s.shape[0]
    c = numpy.zeros((nao, nao))
    weights = numpy.maximum(pre_occ, min_weight)
    if nmb:
        cn = pre_nao[:, nmb]
        s1 = lib.dot(lib.dot(cn.T, s), cn)
        c[:, nmb] = lib.dot(cn, pyscf_orth.weight_orth(s1, weights[nmb]))
    if ryd_lst:
        cr = pre_nao[:, ryd_lst].copy()
        if nmb:
            cn = c[:, nmb]
            cr -= lib.dot(cn, lib.dot(lib.dot(cn.T, s), cr))      # Schmidt to NMB
        s1 = lib.dot(lib.dot(cr.T, s), cr)
        if nrb_weights == 'pre':
            w_ryd = weights[ryd_lst]
        elif nrb_weights == 'post':
            p1 = lib.dot(lib.dot(cr.T, p), cr)
            w_ryd = numpy.maximum(numpy.einsum('ii->i', p1) / numpy.einsum('ii->i', s1), min_weight)
        elif nrb_weights == 'lowdin':
            w_ryd = numpy.ones(len(ryd_lst))
        else:
            raise ValueError(f'nrb_weights must be one of {NRB_WEIGHTS}')
        c[:, ryd_lst] = lib.dot(cr, pyscf_orth.weight_orth(s1, w_ryd))
    # remove round-off from the weighted orthogonalizations
    c = lib.dot(c, pyscf_orth.lowdin(lib.dot(lib.dot(c.T, s), c)))
    if not mol.cart:
        p_nao = lib.dot(lib.dot(c.T, p), c)
        c = lib.dot(c, pyscf_nao._prenao_sub(mol, p_nao, numpy.eye(nao))[1])
    return _fix_phase(c)


def nao_coeff_pyscf(mol, dm, s=None):
    """NAO transformation matrix with the algorithm of ``pyscf.lo.nao.nao``.

    Core: Loewdin orthogonalization; valence: OWSO with the core projected
    out; Rydberg: Schmidt orthogonalization to core + valence followed by
    Loewdin orthogonalization; then restoration of the natural character.
    """
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    dm = _spin_summed(dm)
    p = lib.dot(lib.dot(s, dm), s)
    pre_occ, pre_nao = pyscf_nao._prenao_sub(mol, p, s)
    cnao = pyscf_nao._nao_sub(mol, pre_occ, pre_nao, s)
    if not mol.cart:
        p_nao = lib.dot(lib.dot(cnao.T, p), cnao)
        cnao = lib.dot(cnao, pyscf_nao._prenao_sub(mol, p_nao, numpy.eye(p_nao.shape[0]))[1])
    return _fix_phase(cnao)


def check_orthonormal(x, s, tol=1e-8):
    """Return max |X^T S X - 1|; raise ``ValueError`` if it exceeds ``tol``."""
    err = abs(x.T.dot(s).dot(x) - numpy.eye(x.shape[1])).max()
    if err > tol:
        raise ValueError(f'orthogonal basis is not orthonormal: max deviation {err:.3e}')
    return err


def transform_density(x, s, dm):
    """P' = X^{-1} P X^{-1 dagger} (Eq. 6).  For an S-orthonormal X, X^{-1} = X^T S."""
    xinv = x.T.dot(s)
    return xinv.dot(dm).dot(xinv.T)


def transform_operator(x, mat):
    """M' = X^dagger M X (Eq. 7)."""
    return x.T.dot(mat).dot(x)
