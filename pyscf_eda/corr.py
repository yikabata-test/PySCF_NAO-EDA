"""Atomic partition of correlation energies (shared by MP2-EDA and CCSD-EDA).

References
----------
M. Kobayashi, Y. Imamura, H. Nakai, J. Chem. Phys. 127, 074103 (2007)  [MP2]
M. Kobayashi, H. Nakai, J. Chem. Phys. 129, 044103 (2008)               [CCSD]

For a closed-shell correlated method whose correlation energy is written as

    E_corr = sum_{ij}^{occ} sum_{ab}^{vir} (ia|jb) (2 tau_ijab - tau_ijba)
             [+ 2 sum_{ia} f_ia t_ia]

with effective doubles amplitudes tau (MP2: tau = t2; CCSD: tau = t2 + t1 t1),
the atomic contribution is obtained by leaving the last AO -> MO integral
transformation undone (Eq. 10 of the 2008 paper):

    E_corr^A = sum_{ij} sum_{ab} [ w_occ sum_{l in A} C'_{l i} (l a|jb)
                                 + w_vir sum_{l in A} C'_{l a} (i l|jb) ]
               (2 tau_ijab - tau_ijba),            w_occ + w_vir = 1,

where l runs over the one-centre orbitals phi_l = sum_mu X_{mu l} chi_mu
(raw AOs for X = 1, NAOs / Loewdin orbitals otherwise), C' = X^{-1} C and
(l a|jb) are the half-transformed integrals in that basis.
"""

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf_eda.rhf import EDAResult


def eri_transformer(obj):
    """Function (mo1, mo2, mo3, mo4) -> (12|34), consistent with ``obj``'s integrals.

    ``obj`` is an MP2 or CCSD object; density-fitted objects use their
    ``with_df``, otherwise the cached AO integrals of the SCF object or,
    failing that, an out-of-core transformation from the molecule.
    """
    mf = obj._scf
    with_df = getattr(obj, 'with_df', None)
    if with_df is not None:
        def transform(coeffs):
            return with_df.ao2mo(coeffs, compact=False)
    elif getattr(mf, '_eri', None) is not None:
        eri = mf._eri

        def transform(coeffs):
            return ao2mo.general(eri, coeffs, compact=False)
    else:
        mol = obj.mol

        def transform(coeffs):
            return ao2mo.general(mol, coeffs, compact=False)
    return transform


def active_orbitals(obj):
    """(c_occ, c_vir) of the active (non-frozen) orbitals of an MP2/CCSD object."""
    mo_coeff = numpy.asarray(obj.mo_coeff)[:, obj.get_frozen_mask()]
    nocc = obj.nocc
    return mo_coeff[:, :nocc], mo_coeff[:, nocc:]


def partition_doubles(mol, transform, c_occ, c_vir, tau, x=None, verbose=None):
    """Occupied- and virtual-side atomic partitions of sum (ia|jb)(2 tau_ijab - tau_ijba).

    Parameters
    ----------
    mol : pyscf.gto.Mole
    transform : callable
        ``transform((mo1, mo2, mo3, mo4))`` -> (12|34) integrals, see
        ``eri_transformer``.
    c_occ, c_vir : ndarray
        AO coefficients of the active occupied / virtual orbitals.
    tau : ndarray (nocc, nocc, nvir, nvir)
        Effective doubles amplitudes tau_ijab.
    x : ndarray, optional
        AO -> orthogonal one-centre basis transformation (default: AOs).

    Returns
    -------
    e_occ, e_vir : ndarray (natm,)
    """
    log = logger.new_logger(mol, verbose)
    tau = numpy.asarray(tau)
    nocc = c_occ.shape[1]
    nvir = c_vir.shape[1]
    assert tau.shape == (nocc, nocc, nvir, nvir)
    tbar = 2.0 * tau - tau.transpose(0, 1, 3, 2)          # 2 tau_ijab - tau_ijba

    nao = mol.nao_nr()
    if x is None:
        x = numpy.eye(nao)
    xinv = numpy.linalg.inv(x)
    cp_occ = xinv.dot(c_occ)        # C' = X^{-1} C
    cp_vir = xinv.dot(c_vir)

    e_occ = numpy.zeros(mol.natm)
    e_vir = numpy.zeros(mol.natm)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        if p1 == p0:
            continue
        x_a = x[:, p0:p1]
        na = p1 - p0
        # occupied side: (l a|j b), l in A
        eri = transform((x_a, c_vir, c_occ, c_vir)).reshape(na, nvir, nocc, nvir)
        w = lib.einsum('li,ijab->lajb', cp_occ[p0:p1], tbar)
        e_occ[ia] = numpy.einsum('lajb,lajb->', w, eri)
        # virtual side: (i l|j b), l in A
        eri = transform((c_occ, x_a, c_occ, c_vir)).reshape(nocc, na, nocc, nvir)
        w = lib.einsum('la,ijab->iljb', cp_vir[p0:p1], tbar)
        e_vir[ia] = numpy.einsum('iljb,iljb->', w, eri)
        log.debug1('atom %d  E_corr(occ) = %.10f  E_corr(vir) = %.10f',
                   ia, e_occ[ia], e_vir[ia])
    return e_occ, e_vir


def partition_singles(mol, fock_ao, c_occ, c_vir, t1, x=None):
    """Atomic partitions of the singles term 2 sum_{ia} f_ia t_ia.

    occupied side: 2 sum_{l in A} sum_{ia} C'_{li} (X^T F C_vir)_{la} t_ia
    virtual side  : 2 sum_{l in A} sum_{ia} C'_{la} (X^T F C_occ)_{li} t_ia
    The term vanishes for canonical HF orbitals (Brillouin theorem).
    """
    nao = mol.nao_nr()
    if x is None:
        x = numpy.eye(nao)
    xinv = numpy.linalg.inv(x)
    cp_occ = xinv.dot(c_occ)
    cp_vir = xinv.dot(c_vir)
    f_lv = x.T.dot(fock_ao).dot(c_vir)     # (l, a)
    f_lo = x.T.dot(fock_ao).dot(c_occ)     # (l, i)
    d_occ = 2.0 * numpy.einsum('li,la,ia->l', cp_occ, f_lv, t1)
    d_vir = 2.0 * numpy.einsum('la,li,ia->l', cp_vir, f_lo, t1)
    e_occ = numpy.array([d_occ[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()])
    e_vir = numpy.array([d_vir[p0:p1].sum() for _, _, p0, p1 in mol.aoslice_by_atom()])
    return e_occ, e_vir


class CorrEDAResult(EDAResult):
    """Atomic energy densities of a correlated (MP2, CCSD, ...) EDA.

    In addition to the HF components of ``EDAResult`` (``e_nn``, ``e_kin``,
    ``e_ne``, ``e_1el``, ``e_coul``, ``e_x``, ``e_elec``, ``pop``):

    e_hf       : HF atomic energy  E_HF^A = E_NN^A + E_ELC^A
    e_corr_occ : occupied-orbital partition of the correlation energy
    e_corr_vir : virtual-orbital partition of the correlation energy
    e_corr     : E_corr^A = w_occ * e_corr_occ + w_vir * e_corr_vir
    e_tot      : E_HF^A + E_corr^A
    """

    method = 'corr'
    components = ('e_nn', 'e_kin', 'e_ne', 'e_1el', 'e_coul', 'e_x', 'e_elec',
                  'e_hf', 'e_corr', 'e_tot')

    def __init__(self, hf_result, e_corr_occ, e_corr_vir, w_occ, e_tot_ref=None):
        hf = hf_result
        self.mol = hf.mol
        self.ne_partition = hf.ne_partition
        self.orbital_basis = hf.orbital_basis
        self.orth_coeff = hf.orth_coeff
        self.pop = hf.pop
        for key in ('e_nn', 'e_kin', 'e_ne', 'e_other', 'e_1el', 'e_coul', 'e_x', 'e_elec',
                    'e_xc', 'xc'):
            setattr(self, key, getattr(hf, key, None))
        self.hf_result = hf
        self.e_hf = hf.e_tot
        self.w_occ = w_occ
        self.e_corr_occ = e_corr_occ
        self.e_corr_vir = e_corr_vir
        self.e_corr = w_occ * e_corr_occ + (1.0 - w_occ) * e_corr_vir
        self.e_tot = self.e_hf + self.e_corr
        self.e_tot_scf = e_tot_ref

    @property
    def labels(self):
        return dict(EDAResult.labels, e_hf='E_HF', e_corr='E_corr', e_tot=f'E_{self.method}')

    @property
    def ref_energy_label(self):
        return f'{self.method} total energy'

    def summary(self, atoms=None):
        text = EDAResult.summary(self, atoms)
        return text.replace('Energy density analysis (RHF)',
                            f'Energy density analysis ({self.method}, w_occ={self.w_occ:g})')
