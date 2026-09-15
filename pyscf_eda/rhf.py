"""Energy density analysis (EDA) for closed-shell (restricted) Hartree-Fock.

Reference
---------
H. Nakai, "Energy density analysis with Kohn-Sham orbitals",
Chem. Phys. Lett. 363, 73-79 (2002).

The total RHF energy

    E_TOT = E_NN + T_S + E_Ne + E_CLB + E_X

is partitioned into atomic contributions E_TOT^A such that
sum_A E_TOT^A = E_TOT holds exactly (to machine precision).

Partitioning schemes
--------------------
* Basis-function (Mulliken-type) partitioning, Eq. (5)-(9) of the paper:
  for an operator matrix M the atomic contribution is the partial trace

      E^A[M] = sum_{mu in A} (P M)_{mu mu}

  where P is the AO density matrix and "mu in A" runs over the AOs centred
  on atom A.  This is applied to the kinetic energy (T), the Coulomb energy
  (J/2) and the exact exchange energy (-K/4).

* Nucleus-based partitioning: the nuclear repulsion energy is assigned to
  nuclei by the partial sum of Eq. (11),

      E_NN^A = 1/2 sum_{B != A} Z_A Z_B / R_AB .

* Nucleus-electron attraction.  In this implementation one half of E_Ne is
  partitioned by basis functions and the other half by nuclei:

      E_Ne^A = 1/2 sum_{mu in A} (P V)_{mu mu}
             + 1/2 sum_{mu nu} P_{mu nu} (V_A)_{nu mu},
      V   = sum_A V_A,   (V_A)_{nu mu} = -Z_A <nu| 1/|r - R_A| |mu>.

  This is the default (``ne_partition='half'``).  The original scheme of
  the 2002 paper (pure basis-function partitioning) is available with
  ``ne_partition='mulliken'``; pure nuclear partitioning with
  ``ne_partition='nuclear'``.

  An effective core potential (ECP), if present, is a nucleus-electron
  interaction as well and is treated with the same rule.
"""

import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.lib import logger

NE_PARTITIONS = ('half', 'mulliken', 'nuclear')


def _partial_trace_by_atom(mol, pm):
    """sum_{mu in A} (P M)_{mu mu} for each atom A, given the matrix product PM."""
    diag = numpy.einsum('ii->i', pm)
    aoslices = mol.aoslice_by_atom()
    return numpy.array([diag[p0:p1].sum() for _, _, p0, p1 in aoslices])


def _ao_energy_density(mol, dm, mat):
    """(P M)_{mu mu} summed over the AOs of each atom (Mulliken-type partition)."""
    return _partial_trace_by_atom(mol, dm.dot(mat))


def nuc_attraction_by_nucleus(mol):
    """Per-nucleus attraction matrices V_A with sum_A V_A = int1e_nuc.

    Returns
    -------
    ndarray of shape (natm, nao, nao)
    """
    nao = mol.nao_nr()
    v_atoms = numpy.zeros((mol.natm, nao, nao))
    for ia in range(mol.natm):
        z = mol.atom_charge(ia)
        if z == 0:  # ghost atom
            continue
        with mol.with_rinv_at_nucleus(ia):
            v_atoms[ia] = -z * mol.intor('int1e_rinv')
    return v_atoms


def ecp_by_atom(mol):
    """Per-atom scalar ECP matrices, or ``None`` when the molecule has no ECP.

    Returns
    -------
    ndarray of shape (natm, nao, nao)
    """
    if not mol.has_ecp():
        return None
    nao = mol.nao_nr()
    v_atoms = numpy.zeros((mol.natm, nao, nao))
    ecp_atoms = mol._ecpbas[:, gto.ATOM_OF]
    for ia in set(ecp_atoms.tolist()):
        m = mol.copy()
        m._ecpbas = mol._ecpbas[ecp_atoms == ia]
        v_atoms[ia] = m.intor('ECPscalar')
    return v_atoms


def nuc_repulsion_by_atom(mol):
    """E_NN^A = 1/2 sum_{B != A} Z_A Z_B / R_AB  (Eq. 11)."""
    charges = mol.atom_charges().astype(float)
    coords = mol.atom_coords()
    e_nn = numpy.zeros(mol.natm)
    for ia in range(mol.natm):
        for ib in range(mol.natm):
            if ia == ib:
                continue
            rab = numpy.linalg.norm(coords[ia] - coords[ib])
            e_nn[ia] += 0.5 * charges[ia] * charges[ib] / rab
    return e_nn


class EDAResult:
    """Container for the atomic energy densities of one EDA calculation.

    All per-atom quantities are numpy arrays of length ``mol.natm`` in
    hartree.  ``e_tot`` sums to the SCF total energy.

    Attributes
    ----------
    e_nn   : nuclear repulsion         E_NN^A
    e_kin  : kinetic energy            T_S^A
    e_ne   : nucleus-electron attraction (incl. ECP) E_Ne^A
    e_1el  : one-electron energy       E_1EL^A = T_S^A + E_Ne^A (+ E_other^A)
    e_coul : Coulomb energy            E_CLB^A
    e_x    : exact exchange energy     E_X^A
    e_elec : electronic energy         E_ELC^A = E_1EL^A + E_CLB^A + E_X^A
    e_tot  : total energy              E_TOT^A = E_NN^A + E_ELC^A
    e_other: one-electron terms present in ``mf.get_hcore()`` that are not
             T + V_nuc + V_ecp (e.g. external fields), Mulliken-partitioned.
    """

    components = ('e_nn', 'e_kin', 'e_ne', 'e_1el', 'e_coul', 'e_x', 'e_elec', 'e_tot')
    labels = {
        'e_nn': 'E_NN', 'e_kin': 'T_S', 'e_ne': 'E_Ne', 'e_other': 'E_other',
        'e_1el': 'E_1EL', 'e_coul': 'E_CLB', 'e_x': 'E_X',
        'e_elec': 'E_ELC', 'e_tot': 'E_TOT',
    }

    def __init__(self, mol, ne_partition, **kwargs):
        self.mol = mol
        self.ne_partition = ne_partition
        self.e_nn = kwargs['e_nn']
        self.e_kin = kwargs['e_kin']
        self.e_ne = kwargs['e_ne']
        self.e_other = kwargs.get('e_other', numpy.zeros(mol.natm))
        self.e_coul = kwargs['e_coul']
        self.e_x = kwargs['e_x']
        self.e_1el = self.e_kin + self.e_ne + self.e_other
        self.e_elec = self.e_1el + self.e_coul + self.e_x
        self.e_tot = self.e_nn + self.e_elec
        self.e_tot_scf = kwargs.get('e_tot_scf')

    @property
    def atom_energies(self):
        """Alias of ``e_tot`` (atomic total energies)."""
        return self.e_tot

    def as_dict(self):
        keys = list(self.components)
        if numpy.any(self.e_other != 0):
            keys.insert(3, 'e_other')
        return {k: getattr(self, k) for k in keys}

    def summary(self, atoms=None):
        """Return a table of the atomic energy densities as a string.

        Parameters
        ----------
        atoms : sequence of int, optional
            Atom indices to show (default: all).
        """
        mol = self.mol
        if atoms is None:
            atoms = range(mol.natm)
        atoms = list(atoms)
        rows = self.as_dict()
        width = 16
        header = f"{'Component':<10}" + ''.join(
            f"{f'{mol.atom_symbol(ia)}{ia}':>{width}}" for ia in atoms)
        header += f"{'Sum':>{width}}"
        lines = [
            f"Energy density analysis (RHF), ne_partition='{self.ne_partition}'",
            'Energies in hartree', header, '-' * len(header)]
        for key, values in rows.items():
            line = f"{self.labels[key]:<10}" + ''.join(
                f"{values[ia]:>{width}.8f}" for ia in atoms)
            line += f"{values.sum():>{width}.8f}"
            lines.append(line)
        lines.append('-' * len(header))
        if self.e_tot_scf is not None:
            lines.append(f"SCF total energy        : {self.e_tot_scf:20.10f}")
            lines.append(f"Sum of atomic energies  : {self.e_tot.sum():20.10f}")
            lines.append(f"Difference              : {self.e_tot.sum() - self.e_tot_scf:20.3e}")
        return '\n'.join(lines)

    def __repr__(self):
        return self.summary()


class EDA(lib.StreamObject):
    """Energy density analysis for a converged closed-shell RHF object.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Converged RHF object (non-relativistic, no KS-DFT functional).
    ne_partition : {'half', 'mulliken', 'nuclear'}
        How the nucleus-electron attraction (and ECP) energy is partitioned.
        'half' (default): one half by basis functions, one half by nuclei.
        'mulliken'      : entirely by basis functions (original 2002 scheme).
        'nuclear'       : entirely by nuclei.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf_eda import rhf as eda_rhf
    >>> mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> res = eda_rhf.EDA(mf).kernel()
    >>> print(res.summary())
    """

    def __init__(self, mf, ne_partition='half'):
        self._check_mf(mf)
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.ne_partition = ne_partition
        self.dm = None       # AO density matrix used (default: mf.make_rdm1())
        self.result = None
        self.tol_energy = 1e-8   # tolerance for the sum-rule check

    @staticmethod
    def _check_mf(mf):
        if not isinstance(mf, scf.hf.RHF):
            raise TypeError('EDA for closed-shell RHF requires a pyscf.scf.hf.RHF '
                            f'object, got {type(mf)}')
        if isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.rohf.ROHF):
            raise NotImplementedError('Open-shell (UHF/ROHF) EDA is not implemented')
        if hasattr(mf, 'xc'):
            raise NotImplementedError(
                'KS-DFT (exchange-correlation) EDA is not implemented; '
                'use a pure Hartree-Fock object')

    def kernel(self, dm=None):
        mf = self._scf
        mol = self.mol
        log = logger.new_logger(self)
        if self.ne_partition not in NE_PARTITIONS:
            raise ValueError(f'ne_partition must be one of {NE_PARTITIONS}')

        if dm is None:
            dm = self.dm
        if dm is None:
            if not mf.converged:
                log.warn('SCF is not converged; the EDA is performed on the '
                         'current density matrix anyway')
            dm = mf.make_rdm1()
        dm = numpy.asarray(dm)
        if dm.ndim != 2:
            raise ValueError('EDA for RHF expects a single (spin-summed) density matrix')
        self.dm = dm

        # --- one-electron integrals -------------------------------------
        t_mat = mol.intor_symmetric('int1e_kin')
        v_nuc_atoms = nuc_attraction_by_nucleus(mol)
        v_ecp_atoms = ecp_by_atom(mol)
        v_ne_atoms = v_nuc_atoms
        if v_ecp_atoms is not None:
            v_ne_atoms = v_nuc_atoms + v_ecp_atoms
        v_ne = v_ne_atoms.sum(axis=0)

        hcore = mf.get_hcore(mol)
        h_other = hcore - t_mat - v_ne
        has_other = abs(h_other).max() > 1e-10
        if has_other:
            log.warn('mf.get_hcore() contains one-electron terms beyond T + V_nuc '
                     '(+ V_ecp), max |diff| = %.3e. They are partitioned by basis '
                     'functions and reported as E_other.', abs(h_other).max())

        # --- energy densities -------------------------------------------
        e_nn = nuc_repulsion_by_atom(mol)
        e_kin = _ao_energy_density(mol, dm, t_mat)

        e_ne_ao = _ao_energy_density(mol, dm, v_ne)                     # by basis functions
        e_ne_nuc = numpy.einsum('ij,aji->a', dm, v_ne_atoms)            # by nuclei
        if self.ne_partition == 'half':
            e_ne = 0.5 * e_ne_ao + 0.5 * e_ne_nuc
        elif self.ne_partition == 'mulliken':
            e_ne = e_ne_ao
        else:
            e_ne = e_ne_nuc

        e_other = _ao_energy_density(mol, dm, h_other) if has_other else numpy.zeros(mol.natm)

        vj, vk = mf.get_jk(mol, dm)
        e_coul = 0.5 * _ao_energy_density(mol, dm, vj)
        e_x = -0.25 * _ao_energy_density(mol, dm, vk)

        self.result = EDAResult(mol, self.ne_partition,
                                e_nn=e_nn, e_kin=e_kin, e_ne=e_ne, e_other=e_other,
                                e_coul=e_coul, e_x=e_x,
                                e_tot_scf=mf.e_tot if mf.converged else None)

        # --- sum-rule check ---------------------------------------------
        e_sum = self.result.e_tot.sum()
        if mf.converged and mf.e_tot is not None:
            diff = e_sum - mf.e_tot
            if abs(diff) > self.tol_energy:
                log.warn('Sum of atomic energies (%.10f) differs from the SCF '
                         'total energy (%.10f) by %.3e', e_sum, mf.e_tot, diff)
            else:
                log.info('Sum of atomic energies %.10f reproduces E(SCF) '
                         'within %.1e', e_sum, diff)

        if self.verbose >= logger.INFO:
            log.info('\n%s', self.result.summary())
        return self.result

    run = kernel


def kernel(mf, ne_partition='half', dm=None):
    """Perform the EDA for a converged RHF object and return an ``EDAResult``."""
    return EDA(mf, ne_partition=ne_partition).kernel(dm=dm)


def atom_energies(mf, ne_partition='half', dm=None):
    """Return the atomic total energies E_TOT^A as a numpy array (hartree)."""
    return kernel(mf, ne_partition=ne_partition, dm=dm).e_tot
