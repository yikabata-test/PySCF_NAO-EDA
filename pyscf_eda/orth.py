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
'nao'         : natural atomic orbitals (Reed, Weinstock, Weinhold 1985),
                obtained through the occupancy-weighted symmetric
                orthogonalization (OWSO) implemented in ``pyscf.lo.nao``
'lso'/'lowdin': Loewdin symmetrically orthogonalized AOs, X = S^{-1/2}
'meta_lowdin' : Loewdin orthogonalization within the core, valence and
                Rydberg spaces separately (PySCF default for population
                analysis; not part of the 2006 paper)
"""

import numpy
from pyscf import lib
from pyscf.lo import orth as pyscf_orth
from pyscf.lo import nao as pyscf_nao

ORBITAL_BASES = ('ao', 'nao', 'lso', 'lowdin', 'meta_lowdin')


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
    if method == 'nao':
        if dm is None:
            raise ValueError("a density matrix is required for method='nao'")
        return nao_coeff(mol, dm, s)
    if method == 'meta_lowdin':
        return pyscf_orth.orth_ao(mol, method='meta_lowdin', s=s)
    raise ValueError(f"unknown orbital basis '{method}'; choose from {ORBITAL_BASES} "
                     'or pass a transformation matrix')


def nao_coeff(mol, dm, s=None):
    """NAO transformation matrix built from the AO density matrix ``dm``.

    Same algorithm as ``pyscf.lo.nao.nao`` (pre-NAO diagonalization per atom
    and angular momentum, OWSO of the valence set, Loewdin orthogonalization
    of the core and Rydberg sets, then restoration of the natural character),
    but taking the density matrix explicitly instead of an SCF object.
    """
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    dm = numpy.asarray(dm)
    if dm.ndim == 3:  # (alpha, beta)
        dm = dm[0] + dm[1]
    p = lib.dot(lib.dot(s, dm), s)
    pre_occ, pre_nao = pyscf_nao._prenao_sub(mol, p, s)
    cnao = pyscf_nao._nao_sub(mol, pre_occ, pre_nao, s)
    if not mol.cart:
        # restore natural character (as in pyscf.lo.nao.nao with restore=True)
        p_nao = lib.dot(lib.dot(cnao.T, p), cnao)
        s_nao = numpy.eye(p_nao.shape[0])
        cnao = lib.dot(cnao, pyscf_nao._prenao_sub(mol, p_nao, s_nao)[1])
    # fix the phase so that the diagonal is positive
    for i in range(cnao.shape[1]):
        if cnao[i, i] < 0:
            cnao[:, i] *= -1
    return cnao


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
