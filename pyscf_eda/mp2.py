"""Energy density analysis (EDA) for closed-shell MP2.

Reference
---------
M. Kobayashi, Y. Imamura, H. Nakai, "Alternative linear-scaling methodology
for the second-order Moller-Plesset perturbation calculation based on the
divide-and-conquer method", J. Chem. Phys. 127, 074103 (2007), Sec. II B.

The DC-MP2 paper partitions the correlation energy into atomic contributions
by leaving the last AO -> MO integral transformation undone, in the same
spirit as the Mulliken-type partition of the HF energy.  With one atom per
subsystem and a buffer covering the whole molecule, the DC expressions
(Eqs. 20-22) reduce to those of the canonical MP2 calculation, Eq. (18):

    E_corr   = sum_{ij}^{occ} sum_{ab}^{vir} (ia|jb) (2 t_ijab - t_ijba)
    E_corr^A = sum_{ij} sum_{ab} [ w_occ sum_{mu in A} C_{mu i} (mu a|jb)
                                 + w_vir sum_{mu in A} C_{mu a} (i mu|jb) ]
               (2 t_ijab - t_ijba),                 w_occ + w_vir = 1,

where t_ijab = (ia|jb) / (e_i + e_j - e_a - e_b) are the MP2 amplitudes and
(pq|rs) are two-electron integrals in chemists' notation, so that
sum_A E_corr^A = E_corr exactly.  Following the numerical assessment of the
paper, w_occ = 1 (occupied-orbital partition only) is the default.

Orbital basis
-------------
As in the NAO-EDA of the HF energy, the AO index mu can be replaced by an
orthonormal one-centre basis phi_l = sum_mu X_{mu l} chi_mu (NAO, Loewdin):
C -> X^{-1} C for the MO coefficients and (mu a|jb) -> (l a|jb) for the
half-transformed integrals.  ``orbital_basis`` selects it, the HF part is
partitioned in the same basis, and E_MP2^A = E_HF^A + E_corr^A.

Frozen core orbitals (``mp.frozen``) are respected: only the active
occupied/virtual orbitals enter the correlation partition.
"""

import numpy
from pyscf import lib
from pyscf import mp as pyscf_mp
from pyscf.lib import logger
from pyscf_eda import corr
from pyscf_eda import rhf as eda_rhf
from pyscf_eda.corr import CorrEDAResult


class MP2EDAResult(CorrEDAResult):
    """Atomic energy densities of an MP2-EDA calculation (see ``CorrEDAResult``)."""
    method = 'MP2'


def corr_energy_by_atom(mp, x=None, t2=None, verbose=None):
    """Occupied- and virtual-orbital partitions of the MP2 correlation energy.

    Parameters
    ----------
    mp : pyscf.mp.mp2.RMP2 (or DFRMP2)
        Converged closed-shell MP2 object.
    x : ndarray, optional
        AO -> orthogonal atomic basis transformation matrix (default: AOs).
    t2 : ndarray, optional
        MP2 amplitudes t_ijab (default: ``mp.t2``, computed if needed).

    Returns
    -------
    e_occ, e_vir : ndarray (natm,)
        sum_{l in A} sum_{ijab} C'_{li} (l a|jb) T_ijab   and
        sum_{l in A} sum_{ijab} C'_{la} (i l|jb) T_ijab,  T = 2 t_ijab - t_ijba
    """
    if t2 is None:
        t2 = mp.t2
    if t2 is None:
        _, t2 = mp.kernel(with_t2=True)
    c_occ, c_vir = corr.active_orbitals(mp)
    return corr.partition_doubles(mp.mol, corr.eri_transformer(mp), c_occ, c_vir,
                                  t2, x=x, verbose=verbose)


class EDA(lib.StreamObject):
    """MP2 energy density analysis for a converged closed-shell MP2 object.

    Parameters
    ----------
    mp : pyscf.mp.mp2.RMP2
        Converged RMP2 (or DF-RMP2) object built on an RHF reference.
    ne_partition : {'half', 'mulliken', 'nuclear'}
        Partition of the nucleus-electron attraction in the HF part
        (see ``pyscf_eda.rhf.EDA``).
    orbital_basis : {'ao', 'nao', 'lso', 'lowdin', 'meta_lowdin'} or ndarray
        One-centre orbital basis used for both the HF and the correlation
        partitions ('nao': NAO-EDA (default), 'ao': conventional EDA, ...).
    w_occ : float
        Weight of the occupied-orbital partition of the correlation energy;
        w_vir = 1 - w_occ (Eq. 19).  Default 1.0 as adopted in the paper.

    Examples
    --------
    >>> from pyscf import gto, scf, mp
    >>> from pyscf_eda import mp2 as eda_mp2
    >>> mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> pt = mp.MP2(mf).run()
    >>> res = eda_mp2.EDA(pt).kernel()
    >>> print(res.summary())
    """

    result_class = MP2EDAResult

    def __init__(self, mp, ne_partition='half', orbital_basis='nao', w_occ=1.0):
        self._check_mp(mp)
        self._mp = mp
        self._scf = mp._scf
        self.mol = mp.mol
        self.verbose = mp.verbose
        self.stdout = mp.stdout
        self.ne_partition = ne_partition
        self.orbital_basis = orbital_basis
        self.w_occ = w_occ
        self.orth_coeff = None
        self.result = None
        self.tol_energy = 1e-8

    @staticmethod
    def _check_mp(mp):
        if not isinstance(mp, pyscf_mp.mp2.RMP2):
            raise TypeError('MP2-EDA requires a closed-shell pyscf.mp.mp2.RMP2 object, '
                            f'got {type(mp)}')
        eda_rhf.EDA._check_mf(mp._scf, allow_dft=False)

    def _ensure_amplitudes(self):
        mp = self._mp
        if mp.e_corr is None:
            logger.info(self, 'MP2 has not been run; running it now')
            mp.kernel(with_t2=True)
        elif mp.t2 is None:
            mp.kernel(with_t2=True)

    def _corr_partition(self):
        return corr_energy_by_atom(self._mp, x=self.orth_coeff, verbose=self.verbose)

    def _make_result(self, hf_res, e_occ, e_vir, obj):
        return self.result_class(hf_res, e_occ, e_vir, self.w_occ, e_tot_ref=self._e_tot_ref())

    def _e_corr_ref(self):
        return self._mp.e_corr

    def _e_tot_ref(self):
        return self._mp.e_tot

    def kernel(self):
        obj = self._mp
        log = logger.new_logger(self)
        if not 0.0 <= self.w_occ <= 1.0:
            raise ValueError('w_occ must lie in [0, 1]')
        self._ensure_amplitudes()

        # HF part (same orbital basis and nucleus-electron partition)
        hf_eda = eda_rhf.EDA(self._scf, ne_partition=self.ne_partition,
                             orbital_basis=self.orbital_basis)
        hf_eda.verbose = 0
        hf_res = hf_eda.kernel()
        self.orth_coeff = hf_eda.orth_coeff

        # correlation part
        e_occ, e_vir = self._corr_partition()
        self.result = self._make_result(hf_res, e_occ, e_vir, obj)
        method = self.result.method

        # sum-rule checks
        e_corr_ref = self._e_corr_ref()
        e_tot_ref = self._e_tot_ref()
        for name, arr in (('occupied', e_occ), ('virtual', e_vir)):
            diff = arr.sum() - e_corr_ref
            if abs(diff) > self.tol_energy:
                log.warn('Sum of the %s-partitioned atomic correlation energies '
                         '(%.10f) differs from E_corr (%.10f) by %.3e',
                         name, arr.sum(), e_corr_ref, diff)
        diff = self.result.e_tot.sum() - e_tot_ref
        if abs(diff) > self.tol_energy:
            log.warn('Sum of atomic %s energies (%.10f) differs from E(%s) '
                     '(%.10f) by %.3e', method, self.result.e_tot.sum(), method, e_tot_ref, diff)
        else:
            log.info('Sum of atomic %s energies %.10f reproduces E(%s) within %.1e',
                     method, self.result.e_tot.sum(), method, diff)
        if self.verbose >= logger.INFO:
            log.info('\n%s', self.result.summary())
        return self.result

    run = kernel


def kernel(mp, ne_partition='half', orbital_basis='nao', w_occ=1.0):
    """Perform the MP2-EDA and return an ``MP2EDAResult``."""
    return EDA(mp, ne_partition=ne_partition, orbital_basis=orbital_basis, w_occ=w_occ).kernel()


def atom_energies(mp, ne_partition='half', orbital_basis='nao', w_occ=1.0):
    """Atomic MP2 total energies E_MP2^A (hartree)."""
    return kernel(mp, ne_partition, orbital_basis, w_occ).e_tot


def nao_eda(mp, ne_partition='half', w_occ=1.0):
    """NAO-based MP2-EDA."""
    return kernel(mp, ne_partition=ne_partition, orbital_basis='nao', w_occ=w_occ)
