"""Energy density analysis (EDA) for closed-shell CCSD.

Reference
---------
M. Kobayashi, H. Nakai, "Extension of linear-scaling divide-and-conquer-based
correlation method to coupled cluster theory with singles and doubles
excitations", J. Chem. Phys. 129, 044103 (2008), Sec. II B.

With one atom per subsystem and a buffer covering the whole molecule the
DC-CCSD expressions (Eqs. 11-14) reduce to the canonical CCSD ones,
Eqs. (6), (9) and (10):

    E_corr   = sum_{ij}^{occ} sum_{ab}^{vir} (ia|jb) (2 tau_ijab - tau_ijba)
    E_corr^A = sum_{ij} sum_{ab} [ w_occ sum_{mu in A} C_{mu i} (mu a|jb)
                                 + w_vir sum_{mu in A} C_{mu a} (i mu|jb) ]
               (2 tau_ijab - tau_ijba),                w_occ + w_vir = 1.

Effective amplitudes
--------------------
Eq. (8) of the paper writes the effective doubles amplitude as
t~_ij,ab = t_ia t_jb - t_ib t_ja + t_ij,ab.  That antisymmetrized form
belongs to the spin-orbital energy expression; inserted into the
closed-shell spatial-orbital formula (6) it does not reproduce the CCSD
correlation energy.  The closed-shell CCSD energy (as evaluated by PySCF)
is obtained with

    tau_ijab = t_ijab + t_ia t_jb ,

which is what this module uses, so that sum_A E_corr^A = E_corr holds
exactly.  The singles term 2 sum_ia f_ia t_ia of the general CCSD energy
vanishes for canonical HF orbitals and, as in Eq. (6) of the paper, is not
partitioned by default: a Mulliken-type partition of a vanishing total would
merely redistribute zero among the atoms.  ``with_singles=True`` includes it
(for non-canonical references).

The occupied-side weight w_occ = 1 (Eq. 14 of the paper) is the default.
Frozen core orbitals are respected.  The orbital basis of the partition
('ao', 'nao', 'lso', ...) is chosen as in ``pyscf_eda.rhf``.
"""

import numpy
from pyscf import cc as pyscf_cc
from pyscf import scf
from pyscf.lib import logger
from pyscf_eda import corr
from pyscf_eda import mp2 as eda_mp2
from pyscf_eda import rhf as eda_rhf
from pyscf_eda.corr import CorrEDAResult


class CCSDEDAResult(CorrEDAResult):
    """Atomic energy densities of a CCSD-EDA calculation (see ``CorrEDAResult``)."""
    method = 'CCSD'


def effective_amplitudes(t1, t2):
    """tau_ijab = t_ijab + t_ia t_jb (closed-shell effective doubles amplitudes)."""
    return t2 + numpy.einsum('ia,jb->ijab', t1, t1)


def corr_energy_by_atom(mycc, x=None, t1=None, t2=None, with_singles=False, verbose=None):
    """Occupied- and virtual-orbital partitions of the CCSD correlation energy.

    Parameters
    ----------
    mycc : pyscf.cc.ccsd.CCSD (or DF-CCSD)
        Converged closed-shell CCSD object.
    x : ndarray, optional
        AO -> orthogonal atomic basis transformation matrix (default: AOs).
    t1, t2 : ndarray, optional
        CCSD amplitudes (default: ``mycc.t1``, ``mycc.t2``).
    with_singles : bool
        Also partition the singles term 2 sum_ia f_ia t_ia (zero for
        canonical HF orbitals; default False).

    Returns
    -------
    e_occ, e_vir : ndarray (natm,)
    """
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if t1 is None or t2 is None:
        raise RuntimeError('CCSD amplitudes are not available; run the CCSD first')
    mol = mycc.mol
    c_occ, c_vir = corr.active_orbitals(mycc)
    tau = effective_amplitudes(numpy.asarray(t1), numpy.asarray(t2))
    e_occ, e_vir = corr.partition_doubles(mol, corr.eri_transformer(mycc),
                                          c_occ, c_vir, tau, x=x, verbose=verbose)
    if with_singles:
        fock_ao = mycc._scf.get_fock()
        s_occ, s_vir = corr.partition_singles(mol, fock_ao, c_occ, c_vir, t1, x=x)
        e_occ = e_occ + s_occ
        e_vir = e_vir + s_vir
    return e_occ, e_vir


class EDA(eda_mp2.EDA):
    """CCSD energy density analysis for a converged closed-shell CCSD object.

    Parameters
    ----------
    mycc : pyscf.cc.ccsd.CCSD
        Converged CCSD (or DF-CCSD) object built on an RHF reference.
    ne_partition : {'half', 'mulliken', 'nuclear'}
        Partition of the nucleus-electron attraction in the HF part.
    orbital_basis : {'ao', 'nao', 'lso', 'lowdin', 'meta_lowdin'} or ndarray
        One-centre orbital basis for the HF and correlation partitions.
    w_occ : float
        Weight of the occupied-orbital partition; w_vir = 1 - w_occ.
        Default 1.0 (Eq. 14 of the paper).
    with_singles : bool
        Partition the singles term 2 sum f_ia t_ia as well (default False;
        it vanishes for canonical HF orbitals).

    Examples
    --------
    >>> from pyscf import gto, scf, cc
    >>> from pyscf_eda import ccsd as eda_ccsd
    >>> mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> mycc = cc.CCSD(mf).run()
    >>> res = eda_ccsd.EDA(mycc).kernel()
    >>> print(res.summary())
    """

    result_class = CCSDEDAResult

    def __init__(self, mycc, ne_partition='half', orbital_basis='ao', w_occ=1.0,
                 with_singles=False):
        self._check_cc(mycc)
        self.with_singles = with_singles
        self._mp = mycc          # the correlated object driven by the base class
        self._cc = mycc
        self._scf = mycc._scf
        self.mol = mycc.mol
        self.verbose = mycc.verbose
        self.stdout = mycc.stdout
        self.ne_partition = ne_partition
        self.orbital_basis = orbital_basis
        self.w_occ = w_occ
        self.orth_coeff = None
        self.result = None
        self.tol_energy = 1e-8

    @staticmethod
    def _check_cc(mycc):
        if not isinstance(mycc, pyscf_cc.ccsd.CCSD) or \
           isinstance(mycc, (pyscf_cc.uccsd.UCCSD, pyscf_cc.gccsd.GCCSD)):
            raise TypeError('CCSD-EDA requires a closed-shell pyscf.cc.ccsd.CCSD object, '
                            f'got {type(mycc)}')
        if not isinstance(mycc._scf, scf.hf.RHF) or isinstance(mycc._scf, scf.uhf.UHF):
            raise TypeError('CCSD-EDA requires an RHF reference')
        eda_rhf.EDA._check_mf(mycc._scf)

    def _ensure_amplitudes(self):
        mycc = self._cc
        if mycc.e_corr is None or mycc.t1 is None or mycc.t2 is None:
            logger.info(self, 'CCSD has not been run; running it now')
            mycc.kernel()
        if not mycc.converged:
            logger.warn(self, 'CCSD is not converged; the EDA uses the current amplitudes')

    def _corr_partition(self):
        return corr_energy_by_atom(self._cc, x=self.orth_coeff,
                                   with_singles=self.with_singles, verbose=self.verbose)


def kernel(mycc, ne_partition='half', orbital_basis='ao', w_occ=1.0):
    """Perform the CCSD-EDA and return a ``CCSDEDAResult``."""
    return EDA(mycc, ne_partition=ne_partition, orbital_basis=orbital_basis, w_occ=w_occ).kernel()


def atom_energies(mycc, ne_partition='half', orbital_basis='ao', w_occ=1.0):
    """Atomic CCSD total energies E_CCSD^A (hartree)."""
    return kernel(mycc, ne_partition, orbital_basis, w_occ).e_tot


def nao_eda(mycc, ne_partition='half', w_occ=1.0):
    """NAO-based CCSD-EDA."""
    return kernel(mycc, ne_partition=ne_partition, orbital_basis='nao', w_occ=w_occ)
