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
from pyscf import ao2mo
from pyscf import mp as pyscf_mp
from pyscf.lib import logger
from pyscf_eda import orth
from pyscf_eda import rhf as eda_rhf
from pyscf_eda.rhf import EDAResult


class MP2EDAResult(EDAResult):
    """Atomic energy densities of an MP2-EDA calculation.

    In addition to the HF components of ``EDAResult`` (``e_nn``, ``e_kin``,
    ``e_ne``, ``e_1el``, ``e_coul``, ``e_x``, ``e_elec``, ``pop``):

    e_hf       : HF atomic energy  E_HF^A = E_NN^A + E_ELC^A
    e_corr_occ : occupied-orbital partition of the MP2 correlation energy
    e_corr_vir : virtual-orbital partition of the MP2 correlation energy
    e_corr     : E_corr^A = w_occ * e_corr_occ + w_vir * e_corr_vir
    e_tot      : E_MP2^A = E_HF^A + E_corr^A
    """

    components = ('e_nn', 'e_kin', 'e_ne', 'e_1el', 'e_coul', 'e_x', 'e_elec',
                  'e_hf', 'e_corr', 'e_tot')
    labels = dict(EDAResult.labels, e_hf='E_HF', e_corr='E_corr', e_tot='E_MP2')
    ref_energy_label = 'MP2 total energy'

    def __init__(self, hf_result, e_corr_occ, e_corr_vir, w_occ, e_tot_scf=None):
        hf = hf_result
        self.mol = hf.mol
        self.ne_partition = hf.ne_partition
        self.orbital_basis = hf.orbital_basis
        self.orth_coeff = hf.orth_coeff
        self.pop = hf.pop
        for key in ('e_nn', 'e_kin', 'e_ne', 'e_other', 'e_1el', 'e_coul', 'e_x', 'e_elec'):
            setattr(self, key, getattr(hf, key))
        self.hf_result = hf
        self.e_hf = hf.e_tot
        self.w_occ = w_occ
        self.e_corr_occ = e_corr_occ
        self.e_corr_vir = e_corr_vir
        self.e_corr = w_occ * e_corr_occ + (1.0 - w_occ) * e_corr_vir
        self.e_tot = self.e_hf + self.e_corr
        self.e_tot_scf = e_tot_scf

    def summary(self, atoms=None):
        text = EDAResult.summary(self, atoms)
        return text.replace('Energy density analysis (RHF)',
                            f'Energy density analysis (MP2, w_occ={self.w_occ:g})')


def _eri_transformer(mp):
    """Return a function transforming (mo1, mo2, mo3, mo4) -> (12|34) integrals."""
    mf = mp._scf
    with_df = getattr(mp, 'with_df', None)
    if with_df is not None:
        def transform(coeffs):
            return with_df.ao2mo(coeffs, compact=False)
    elif getattr(mf, '_eri', None) is not None:
        eri = mf._eri

        def transform(coeffs):
            return ao2mo.general(eri, coeffs, compact=False)
    else:
        mol = mp.mol

        def transform(coeffs):
            return ao2mo.general(mol, coeffs, compact=False)
    return transform


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
    log = logger.new_logger(mp, verbose)
    mol = mp.mol
    if t2 is None:
        t2 = mp.t2
    if t2 is None:
        _, t2 = mp.kernel(with_t2=True)
    t2 = numpy.asarray(t2)
    tau = 2.0 * t2 - t2.transpose(0, 1, 3, 2)          # T_ijab = 2 t_ijab - t_ijba

    mo_coeff = numpy.asarray(mp.mo_coeff)[:, mp.get_frozen_mask()]
    nocc = mp.nocc
    c_occ = mo_coeff[:, :nocc]
    c_vir = mo_coeff[:, nocc:]
    nvir = c_vir.shape[1]
    assert t2.shape == (nocc, nocc, nvir, nvir)

    nao = mol.nao_nr()
    if x is None:
        x = numpy.eye(nao)
    xinv = numpy.linalg.inv(x)
    cp_occ = xinv.dot(c_occ)        # C' = X^{-1} C, MO coefficients in the phi basis
    cp_vir = xinv.dot(c_vir)

    transform = _eri_transformer(mp)
    aoslices = mol.aoslice_by_atom()
    e_occ = numpy.zeros(mol.natm)
    e_vir = numpy.zeros(mol.natm)
    for ia, (_, _, p0, p1) in enumerate(aoslices):
        if p1 == p0:
            continue
        x_a = x[:, p0:p1]
        na = p1 - p0
        # (l a|j b), l in A
        eri = transform((x_a, c_vir, c_occ, c_vir)).reshape(na, nvir, nocc, nvir)
        # sum_i C'_{li} T_ijab -> (l, j, a, b), then contract with (la|jb)
        w = lib.einsum('li,ijab->lajb', cp_occ[p0:p1], tau)
        e_occ[ia] = numpy.einsum('lajb,lajb->', w, eri)
        # (i l|j b), l in A
        eri = transform((c_occ, x_a, c_occ, c_vir)).reshape(nocc, na, nocc, nvir)
        w = lib.einsum('la,ijab->iljb', cp_vir[p0:p1], tau)
        e_vir[ia] = numpy.einsum('iljb,iljb->', w, eri)
        log.debug1('atom %d  E_corr(occ) = %.10f  E_corr(vir) = %.10f', ia, e_occ[ia], e_vir[ia])
    return e_occ, e_vir


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
        partitions ('ao': conventional EDA, 'nao': NAO-EDA, ...).
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

    def __init__(self, mp, ne_partition='half', orbital_basis='ao', w_occ=1.0):
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
        eda_rhf.EDA._check_mf(mp._scf)

    def kernel(self):
        mp = self._mp
        mol = self.mol
        log = logger.new_logger(self)
        if not 0.0 <= self.w_occ <= 1.0:
            raise ValueError('w_occ must lie in [0, 1]')
        if mp.e_corr is None:
            log.info('MP2 has not been run; running it now')
            mp.kernel(with_t2=True)
        elif mp.t2 is None:
            mp.kernel(with_t2=True)

        # HF part (same orbital basis and nucleus-electron partition)
        hf_eda = eda_rhf.EDA(self._scf, ne_partition=self.ne_partition,
                             orbital_basis=self.orbital_basis)
        hf_eda.verbose = 0
        hf_res = hf_eda.kernel()
        self.orth_coeff = hf_eda.orth_coeff

        # correlation part
        e_occ, e_vir = corr_energy_by_atom(mp, x=self.orth_coeff, verbose=self.verbose)

        e_tot_ref = mp.e_tot if mp.e_corr is not None else None
        self.result = MP2EDAResult(hf_res, e_occ, e_vir, self.w_occ, e_tot_scf=e_tot_ref)

        # sum-rule checks
        for name, arr, ref in (('occupied', e_occ, mp.e_corr),
                               ('virtual', e_vir, mp.e_corr)):
            diff = arr.sum() - ref
            if abs(diff) > self.tol_energy:
                log.warn('Sum of the %s-partitioned atomic correlation energies '
                         '(%.10f) differs from E_corr (%.10f) by %.3e',
                         name, arr.sum(), ref, diff)
        diff = self.result.e_tot.sum() - mp.e_tot
        if abs(diff) > self.tol_energy:
            log.warn('Sum of atomic MP2 energies (%.10f) differs from E(MP2) '
                     '(%.10f) by %.3e', self.result.e_tot.sum(), mp.e_tot, diff)
        else:
            log.info('Sum of atomic MP2 energies %.10f reproduces E(MP2) within %.1e',
                     self.result.e_tot.sum(), diff)
        if self.verbose >= logger.INFO:
            log.info('\n%s', self.result.summary())
        return self.result

    run = kernel


def kernel(mp, ne_partition='half', orbital_basis='ao', w_occ=1.0):
    """Perform the MP2-EDA and return an ``MP2EDAResult``."""
    return EDA(mp, ne_partition=ne_partition, orbital_basis=orbital_basis, w_occ=w_occ).kernel()


def atom_energies(mp, ne_partition='half', orbital_basis='ao', w_occ=1.0):
    """Atomic MP2 total energies E_MP2^A (hartree)."""
    return kernel(mp, ne_partition, orbital_basis, w_occ).e_tot


def nao_eda(mp, ne_partition='half', w_occ=1.0):
    """NAO-based MP2-EDA."""
    return kernel(mp, ne_partition=ne_partition, orbital_basis='nao', w_occ=w_occ)
