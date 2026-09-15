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

Kohn-Sham DFT (RKS)
-------------------
For a restricted KS-DFT object the exchange-correlation energy is
partitioned on the numerical quadrature grid (Eq. 3 of the 2002 paper),

    E_XC^A = sum_{g in A} w_g p_A(r_g) F_XC(r_g),

where the grid points of atom A carry the Becke-type partition function
p_A (PySCF stores the atom of every grid point in ``grids.atm_idx``, and
``grids.weights`` already contain w_g p_A).  The exact-exchange part of a
(range-separated) hybrid functional is partitioned by basis functions like
the HF exchange, scaled with the functional's coefficients.  Non-local
correlation (VV10) and dispersion corrections are not supported.

Orbital basis (conventional, LSO- and NAO-EDA)
----------------------------------------------
T. Baba, M. Takeuchi, H. Nakai, Chem. Phys. Lett. 424, 193 (2006).

The basis-function partition above can be carried out in any orthonormal
one-centre basis {phi_l = sum_m X_{m l} chi_m} instead of the raw AOs:

    P' = X^{-1} P X^{-1 dagger},   M' = X^dagger M X,
    E^A[M] = sum_{l in A} (P' M')_{l l}                          (Eqs. 6, 7)

With X = 1 this is the conventional (Mulliken-type) EDA, with X = S^{-1/2}
the LSO-EDA, and with the natural atomic orbitals (NAO) the NAO-EDA, which
has the weakest basis-set dependence.  Select with ``orbital_basis``;
the default is the NAO-EDA ('nao'), the conventional EDA is 'ao'.
"""

import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.lib import logger
from pyscf_eda import orth
from pyscf_eda.orth import ORBITAL_BASES

NE_PARTITIONS = ('half', 'mulliken', 'nuclear')


def _partial_trace_by_atom(mol, pm):
    """sum_{l in A} (P M)_{l l} for each atom A, given the matrix product PM."""
    diag = numpy.einsum('ii->i', pm)
    aoslices = mol.aoslice_by_atom()
    return numpy.array([diag[p0:p1].sum() for _, _, p0, p1 in aoslices])


def _ao_energy_density(mol, dm, mat):
    """(P M)_{mu mu} summed over the AOs of each atom (Mulliken-type partition)."""
    return _partial_trace_by_atom(mol, dm.dot(mat))


class _OrbitalPartition:
    """Atomic partial traces sum_{l in A} (P' M')_{ll} in the basis phi = chi X.

    P' = X^{-1} P X^{-1 dagger} and M' = X^dagger M X, hence
    P' M' = X^{-1} (P M) X.  For X = 1 this reduces to the Mulliken-type
    partition of the conventional EDA.
    """

    def __init__(self, mol, x):
        self.mol = mol
        self.x = x
        self.xinv = numpy.linalg.inv(x)

    def __call__(self, dm, mat):
        pm = self.xinv.dot(dm).dot(mat).dot(self.x)
        return _partial_trace_by_atom(self.mol, pm)


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


def becke_grids(mf_or_mol, level=None):
    """Grids with Becke's original partition and Bragg-radius size adjustment.

    This is the partition used by the EDA code of Nakai (HONDO99/GAMESS):
    with it, Table 1 of Chem. Phys. Lett. 363, 73 (2002) is reproduced to
    all printed digits (Cartesian cc-pVDZ, B3LYP with VWN-RPA).  PySCF's
    default grids use the Treutler-Ahlrichs size adjustment instead, which
    changes the atomic E_XC partition by up to ~0.1 hartree while leaving
    the total energy unchanged.

    Usage: ``mf.grids = becke_grids(mf)`` before running the SCF.
    """
    from pyscf import dft
    from pyscf.dft import gen_grid, radi
    mol = mf_or_mol.mol if hasattr(mf_or_mol, 'mol') else mf_or_mol
    grids = dft.Grids(mol)
    grids.becke_scheme = gen_grid.original_becke
    grids.radii_adjust = radi.becke_atomic_radii_adjust
    grids.atomic_radii = radi.BRAGG_RADII
    if level is not None:
        grids.level = level
    return grids


def is_dft(mf):
    """True for a Kohn-Sham DFT object with a non-HF functional."""
    from pyscf import dft
    return isinstance(mf, dft.rks.KohnShamDFT)


def xc_energy_by_atom(mf, xc_code=None, dm=None, grids=None):
    """Grid partition of a functional energy: E_XC^A = sum_{g in A} w_g F_XC(r_g).

    Parameters
    ----------
    mf : pyscf.dft.rks.RKS
        Converged RKS object (its density and grids are used by default).
    xc_code : str, optional
        Functional to evaluate on the density (default ``mf.xc``).  Any
        libxc expression is accepted, e.g. 'LDA_X', 'GGA_X_B88',
        'LDA_C_VWN_RPA', 'GGA_C_LYP', which allows the decomposition of a
        hybrid functional into its constituents.  The exact-exchange part
        of a hybrid is *not* included (it is not a grid quantity).
    dm : ndarray, optional
        Spin-summed density matrix (default ``mf.make_rdm1()``).
    grids : pyscf.dft.gen_grid.Grids, optional
        Grids with ``atm_idx`` (default ``mf.grids``).

    Returns
    -------
    ndarray (natm,)
    """
    mol = mf.mol
    ni = mf._numint
    if xc_code is None:
        xc_code = mf.xc
    if dm is None:
        dm = mf.make_rdm1()
    dm = numpy.asarray(dm)
    if grids is None:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    if getattr(grids, 'atm_idx', None) is None:
        raise RuntimeError('grids.atm_idx is not available; rebuild the grids with '
                           'grids.build() (PySCF >= 2.4)')
    if ni.libxc.is_nlc(xc_code):
        raise NotImplementedError('non-local correlation functionals are not supported')
    xctype = ni._xc_type(xc_code)
    if xctype == 'HF':
        return numpy.zeros(mol.natm)
    ao_deriv = 0 if xctype == 'LDA' else 1
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi=1, with_lapl=False)
    exc_atoms = numpy.zeros(mol.natm)
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv):
        p0, p1 = p1, p1 + weight.size
        rho = make_rho(0, ao, mask, xctype)
        exc = ni.eval_xc_eff(xc_code, rho, deriv=0, xctype=xctype, spin=0)[0]
        den = (rho if xctype == 'LDA' else rho[0]) * weight
        contrib = den * exc
        idx = grids.atm_idx[p0:p1]
        keep = idx >= 0                      # padding points carry atm_idx = -1
        exc_atoms += numpy.bincount(idx[keep], weights=contrib[keep], minlength=mol.natm)
    return exc_atoms


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
    e_x    : exact exchange energy     E_X^A (scaled by the hybrid coefficient for DFT)
    e_xc   : DFT exchange-correlation energy E_XC^A (grid partition; zero for HF)
    e_elec : electronic energy         E_ELC^A = E_1EL^A + E_CLB^A + E_X^A (+ E_XC^A)
    e_tot  : total energy              E_TOT^A = E_NN^A + E_ELC^A
    e_other: one-electron terms present in ``mf.get_hcore()`` that are not
             T + V_nuc + V_ecp (e.g. external fields), Mulliken-partitioned.
    pop    : atomic electron populations sum_{l in A} (P' S')_{ll} in the same
             orbital basis (Mulliken / Loewdin / natural populations for
             'ao' / 'lso' / 'nao')
    orbital_basis : name of the orbital basis ('ao', 'lso', 'nao', ...)
    orth_coeff    : transformation matrix X used
    """

    components = ('e_nn', 'e_kin', 'e_ne', 'e_1el', 'e_coul', 'e_x', 'e_elec', 'e_tot')
    ref_energy_label = 'SCF total energy'
    labels = {
        'e_nn': 'E_NN', 'e_kin': 'T_S', 'e_ne': 'E_Ne', 'e_other': 'E_other',
        'e_1el': 'E_1EL', 'e_coul': 'E_CLB', 'e_x': 'E_X', 'e_xc': 'E_XC',
        'e_elec': 'E_ELC', 'e_tot': 'E_TOT',
    }

    def __init__(self, mol, ne_partition, orbital_basis='nao', **kwargs):
        self.mol = mol
        self.ne_partition = ne_partition
        self.orbital_basis = orbital_basis
        self.orth_coeff = kwargs.get('orth_coeff')
        self.pop = kwargs.get('pop')
        self.e_nn = kwargs['e_nn']
        self.e_kin = kwargs['e_kin']
        self.e_ne = kwargs['e_ne']
        self.e_other = kwargs.get('e_other', numpy.zeros(mol.natm))
        self.e_coul = kwargs['e_coul']
        self.e_x = kwargs['e_x']
        self.e_xc = kwargs.get('e_xc')
        self.xc = kwargs.get('xc')            # functional name for DFT, None for HF
        self.e_1el = self.e_kin + self.e_ne + self.e_other
        self.e_elec = self.e_1el + self.e_coul + self.e_x
        if self.e_xc is not None:
            self.e_elec = self.e_elec + self.e_xc
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
        if self.e_xc is not None:
            keys.insert(keys.index('e_x') + 1, 'e_xc')
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
        method = f'RKS {self.xc}' if self.xc is not None else 'RHF'
        lines = [
            f"Energy density analysis ({method}), orbital_basis='{self.orbital_basis}', "
            f"ne_partition='{self.ne_partition}'",
            'Energies in hartree', header, '-' * len(header)]
        if self.pop is not None:
            line = f"{'Population':<10}" + ''.join(
                f"{self.pop[ia]:>{width}.5f}" for ia in atoms)
            line += f"{self.pop.sum():>{width}.5f}"
            lines += [line, '-' * len(header)]
        for key, values in rows.items():
            line = f"{self.labels[key]:<10}" + ''.join(
                f"{values[ia]:>{width}.8f}" for ia in atoms)
            line += f"{values.sum():>{width}.8f}"
            lines.append(line)
        lines.append('-' * len(header))
        if self.e_tot_scf is not None:
            lines.append(f"{self.ref_energy_label:<24}: {self.e_tot_scf:20.10f}")
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
    orbital_basis : {'ao', 'nao', 'lso', 'lowdin', 'meta_lowdin'} or ndarray
        One-centre orbital basis in which the basis-function partition is
        carried out.
        'nao' (default): natural atomic orbitals, NAO-EDA (Baba et al. 2006).
        'ao'           : raw AOs, conventional (Mulliken-type) EDA.
        'lso'/'lowdin' : Loewdin symmetrically orthogonalized AOs, LSO-EDA.
        'meta_lowdin'  : PySCF's meta-Loewdin orbitals.
        An explicit (nao, nao) transformation matrix X (phi = chi X) may
        also be given.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf_eda import rhf as eda_rhf
    >>> mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> res = eda_rhf.EDA(mf).kernel()                      # NAO-EDA (default)
    >>> res = eda_rhf.EDA(mf, orbital_basis='ao').kernel()  # conventional EDA
    >>> print(res.summary())
    """

    def __init__(self, mf, ne_partition='half', orbital_basis='nao'):
        self._check_mf(mf)
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.ne_partition = ne_partition
        self.orbital_basis = orbital_basis
        self.orth_coeff = None   # transformation matrix X (built in kernel)
        self.dm = None       # AO density matrix used (default: mf.make_rdm1())
        self.result = None
        self.tol_energy = 1e-8   # tolerance for the sum-rule check

    @staticmethod
    def _check_mf(mf, allow_dft=True):
        if not isinstance(mf, scf.hf.RHF):
            raise TypeError('EDA for closed-shell RHF/RKS requires a pyscf.scf.hf.RHF '
                            f'(or dft.rks.RKS) object, got {type(mf)}')
        if isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.rohf.ROHF):
            raise NotImplementedError('Open-shell (UHF/ROHF/UKS) EDA is not implemented')
        if is_dft(mf):
            if not allow_dft:
                raise TypeError('a Hartree-Fock reference is required, got a KS-DFT object')
            if getattr(mf, 'nlc', None) or mf._numint.libxc.is_nlc(mf.xc):
                raise NotImplementedError('non-local correlation (VV10) is not supported')
            if getattr(mf, 'disp', None) or getattr(mf, 'do_disp', lambda: False)():
                raise NotImplementedError('dispersion corrections are not supported')

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

        # --- orbital basis for the basis-function partition ---------------
        s_mat = mol.intor_symmetric('int1e_ovlp')
        if isinstance(self.orbital_basis, numpy.ndarray):
            basis_name = 'custom'
        else:
            basis_name = str(self.orbital_basis).lower()
            if basis_name not in ORBITAL_BASES:
                raise ValueError(f'orbital_basis must be one of {ORBITAL_BASES} '
                                 'or a transformation matrix')
        x = orth.orth_coeff(mol, self.orbital_basis, dm=dm, s=s_mat)
        if basis_name != 'ao':
            err = orth.check_orthonormal(x, s_mat)
            log.debug('orbital basis %s: max |X^T S X - 1| = %.3e', basis_name, err)
        self.orth_coeff = x
        partition = _OrbitalPartition(mol, x)

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
        e_kin = partition(dm, t_mat)

        e_ne_ao = partition(dm, v_ne)                                   # by basis functions
        e_ne_nuc = numpy.einsum('ij,aji->a', dm, v_ne_atoms)            # by nuclei
        if self.ne_partition == 'half':
            e_ne = 0.5 * e_ne_ao + 0.5 * e_ne_nuc
        elif self.ne_partition == 'mulliken':
            e_ne = e_ne_ao
        else:
            e_ne = e_ne_nuc

        e_other = partition(dm, h_other) if has_other else numpy.zeros(mol.natm)

        e_xc = None
        xc = None
        if is_dft(mf):
            xc = mf.xc
            ni = mf._numint
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj = mf.get_j(mol, dm)
            e_coul = 0.5 * partition(dm, vj)
            if abs(hyb) > 1e-10 or abs(alpha) > 1e-10:
                vk = mf.get_k(mol, dm) * hyb
                if abs(omega) > 1e-10:
                    vk += mf.get_k(mol, dm, omega=omega) * (alpha - hyb)
                e_x = -0.25 * partition(dm, vk)
            else:
                e_x = numpy.zeros(mol.natm)
            e_xc = xc_energy_by_atom(mf, mf.xc, dm)
            log.info('E_XC = %.10f (grid partition, %d points)', e_xc.sum(), mf.grids.weights.size)
        else:
            vj, vk = mf.get_jk(mol, dm)
            e_coul = 0.5 * partition(dm, vj)
            e_x = -0.25 * partition(dm, vk)

        pop = partition(dm, s_mat)   # Mulliken / Loewdin / natural populations

        self.result = EDAResult(mol, self.ne_partition, basis_name,
                                e_nn=e_nn, e_kin=e_kin, e_ne=e_ne, e_other=e_other,
                                e_coul=e_coul, e_x=e_x, e_xc=e_xc, xc=xc, pop=pop,
                                orth_coeff=x, e_tot_scf=mf.e_tot if mf.converged else None)

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


def kernel(mf, ne_partition='half', orbital_basis='nao', dm=None):
    """Perform the EDA for a converged RHF object and return an ``EDAResult``."""
    return EDA(mf, ne_partition=ne_partition, orbital_basis=orbital_basis).kernel(dm=dm)


def atom_energies(mf, ne_partition='half', orbital_basis='nao', dm=None):
    """Return the atomic total energies E_TOT^A as a numpy array (hartree)."""
    return kernel(mf, ne_partition=ne_partition, orbital_basis=orbital_basis, dm=dm).e_tot


def nao_eda(mf, ne_partition='half', dm=None):
    """NAO-EDA (Baba, Takeuchi, Nakai 2006) for a converged RHF object."""
    return kernel(mf, ne_partition=ne_partition, orbital_basis='nao', dm=dm)


def lso_eda(mf, ne_partition='half', dm=None):
    """LSO-EDA (Loewdin symmetric orthogonalization) for a converged RHF object."""
    return kernel(mf, ne_partition=ne_partition, orbital_basis='lso', dm=dm)
