"""Complete-basis-set (CBS) atomic energies from n-scheme linear fitting.

Reference
---------
J. Seino, H. Nakai, "Informatics-based energy fitting scheme for correlation
energy at complete basis set limit", J. Comput. Chem. 37, 2304-2315 (2016).

The three-scheme linear fitting (3SLF) models estimate the CCSD(T)
correlation energy at the CBS limit as a linear combination of correlation
energies obtained with the hierarchical cc-pVXZ / aug-cc-pVXZ basis sets
(Eq. 21):

    E_corr^CBS = sum_X c^L_X E_MP2[X] + sum_X c^M_X E_CCSD[X] + sum_X c^H_X E_CCSD(T)[X]

    QDD : MP2/D,T,Q  + CCSD/D    + CCSD(T)/D     (5 energies)
    QTD : MP2/D,T,Q  + CCSD/D,T  + CCSD(T)/D     (6 energies)
    QTT : MP2/D,T,Q  + CCSD/D,T  + CCSD(T)/D,T   (7 energies)
    QTN : MP2/D,T,Q  + CCSD/D,T                  (5 energies)

with the optimal coefficients of Table 11 (fitted to CCSD(T)/HKKN[3,4]
references of the Gaussian-2/3X sets, frozen-core correlation energies).

Because the model is linear in the correlation energies, and every EDA
partition implemented here is linear as well, the atomic correlation
energies at the CBS limit are obtained with the same coefficients:

    E_corr^{CBS,A} = sum c E_corr^A(method, X),    sum_A E_corr^{CBS,A} = E_corr^CBS.

All energies of one basis set come from a single chain of calculations
(HF -> MP2 -> CCSD -> (T)), so QDD needs the CCSD(T)/DZ, MP2/TZ and MP2/QZ
chains and QTD the CCSD(T)/DZ, CCSD/TZ and MP2/QZ chains: three basis sets
each, with the DZ chain also providing MP2/DZ and CCSD/DZ, and the TZ CCSD
chain also providing MP2/TZ.

The HF part is not covered by the fitting model; by default the atomic HF
energies of the largest basis set (QZ) are used (``hf_cbs='largest'``).
``hf_cbs='karton-martin'`` applies the two-point (T, Q) HF extrapolation
E_HF(X) = E_CBS + A (X+1) exp(-9 sqrt(X)) of Karton and Martin, Theor.
Chem. Acc. 115, 330 (2006), which is linear in the energies as well.
"""

import numpy
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import mp as pyscf_mp
from pyscf import cc as pyscf_cc
from pyscf.lib import logger
from pyscf_eda import rhf as eda_rhf
from pyscf_eda import mp2 as eda_mp2
from pyscf_eda import ccsd as eda_ccsd
from pyscf_eda import ccsd_t as eda_ccsd_t

CARDINAL = {'D': 2, 'T': 3, 'Q': 4}
BASIS_FAMILIES = {
    'cc-pv': {'D': 'cc-pvdz', 'T': 'cc-pvtz', 'Q': 'cc-pvqz'},
    'aug-cc-pv': {'D': 'aug-cc-pvdz', 'T': 'aug-cc-pvtz', 'Q': 'aug-cc-pvqz'},
}
METHOD_LEVEL = {'MP2': 0, 'CCSD': 1, 'CCSD(T)': 2}

# Table 11 of Seino & Nakai (2016): 3SLF coefficients for the CCSD(T)/CBS
# reference.  Keys are (method, cardinal letter).
SCHEMES = {
    'QTN': {
        'cc-pv': {('MP2', 'D'): 2.1938, ('MP2', 'T'): -4.5607, ('MP2', 'Q'): 2.5895,
                  ('CCSD', 'D'): -1.5489, ('CCSD', 'T'): 2.4508},
        'aug-cc-pv': {('MP2', 'D'): 3.1440, ('MP2', 'T'): -5.2405, ('MP2', 'Q'): 2.3602,
                      ('CCSD', 'D'): -2.2552, ('CCSD', 'T'): 3.1713},
    },
    'QDD': {
        'cc-pv': {('MP2', 'D'): -1.6740, ('MP2', 'T'): -1.0099, ('MP2', 'Q'): 2.3525,
                  ('CCSD', 'D'): -1.3047, ('CCSD(T)', 'D'): 2.4337},
        'aug-cc-pv': {('MP2', 'D'): -2.0882, ('MP2', 'T'): 0.6252, ('MP2', 'Q'): 1.2493,
                      ('CCSD', 'D'): -1.1398, ('CCSD(T)', 'D'): 2.1470},
    },
    'QTD': {
        'cc-pv': {('MP2', 'D'): 0.4236, ('MP2', 'T'): -1.9340, ('MP2', 'Q'): 1.5360,
                  ('CCSD', 'D'): -1.9603, ('CCSD', 'T'): 1.6230, ('CCSD(T)', 'D'): 1.2853},
        'aug-cc-pv': {('MP2', 'D'): 0.7664, ('MP2', 'T'): -2.6017, ('MP2', 'Q'): 1.8667,
                      ('CCSD', 'D'): -1.9429, ('CCSD', 'T'): 1.7591, ('CCSD(T)', 'D'): 1.1465},
    },
    'QTT': {
        'cc-pv': {('MP2', 'D'): 0.2595, ('MP2', 'T'): -2.0564, ('MP2', 'Q'): 1.8197,
                  ('CCSD', 'D'): -0.2910, ('CCSD', 'T'): 0.2509,
                  ('CCSD(T)', 'D'): -0.0433, ('CCSD(T)', 'T'): 1.0351},
        'aug-cc-pv': {('MP2', 'D'): 0.2476, ('MP2', 'T'): -2.0832, ('MP2', 'Q'): 1.8613,
                      ('CCSD', 'D'): -0.1268, ('CCSD', 'T'): 0.1255,
                      ('CCSD(T)', 'D'): -0.1665, ('CCSD(T)', 'T'): 1.1189},
    },
}


def chemical_core(mol):
    """Number of chemical core orbitals (frozen-core convention of the paper)."""
    from pyscf.data import elements
    return elements.chemcore(mol)


def hf_cbs_coefficients(method, cardinals):
    """Linear coefficients {X: c_X} of the HF CBS estimate from the given cardinals."""
    cardinals = sorted(cardinals, key=CARDINAL.get)
    if method == 'largest':
        return {cardinals[-1]: 1.0}
    if method == 'karton-martin':
        if len(cardinals) < 2:
            raise ValueError('karton-martin HF extrapolation needs two basis sets')
        x, y = cardinals[-2], cardinals[-1]

        def f(letter):
            n = CARDINAL[letter]
            return (n + 1) * numpy.exp(-9.0 * numpy.sqrt(n))
        denom = f(x) - f(y)
        return {y: f(x) / denom, x: -f(y) / denom}
    raise ValueError(f"unknown hf_cbs '{method}'; choose 'largest' or 'karton-martin'")


class CBSEDAResult:
    """Atomic energies at the CBS limit from an n-scheme fitting model.

    Attributes (per-atom arrays, hartree)
    -------------------------------------
    e_hf        : HF atomic energies of the HF reference (``hf_cbs``)
    e_corr      : CBS-fitted atomic correlation energies
    e_tot       : e_hf + e_corr
    corr        : dict {(method, X): atomic correlation energies of that level}
    hf          : dict {X: atomic HF energies}
    coeff       : dict {(method, X): fitting coefficient}
    hf_coeff    : dict {X: coefficient of the HF CBS estimate}
    e_corr_mol  : molecular CBS correlation energy (equals e_corr.sum())
    e_tot_mol   : molecular CBS total energy
    corr_mol    : dict of molecular correlation energies
    hf_mol      : dict of molecular HF energies
    eda_results : dict {(method, X): the underlying EDA result objects}
    """

    def __init__(self, mol, scheme, basis_family, coeff, hf_coeff, corr, hf,
                 corr_mol, hf_mol, eda_results, orbital_basis, ne_partition, w_occ, frozen):
        self.mol = mol
        self.scheme = scheme
        self.basis_family = basis_family
        self.coeff = coeff
        self.hf_coeff = hf_coeff
        self.corr = corr
        self.hf = hf
        self.corr_mol = corr_mol
        self.hf_mol = hf_mol
        self.eda_results = eda_results
        self.orbital_basis = orbital_basis
        self.ne_partition = ne_partition
        self.w_occ = w_occ
        self.frozen = frozen
        self.e_corr = sum(c * corr[key] for key, c in coeff.items())
        self.e_hf = sum(c * hf[x] for x, c in hf_coeff.items())
        self.e_tot = self.e_hf + self.e_corr
        self.e_corr_mol = sum(c * corr_mol[key] for key, c in coeff.items())
        self.e_hf_mol = sum(c * hf_mol[x] for x, c in hf_coeff.items())
        self.e_tot_mol = self.e_hf_mol + self.e_corr_mol

    @property
    def atom_energies(self):
        return self.e_tot

    def summary(self, atoms=None):
        mol = self.mol
        if atoms is None:
            atoms = range(mol.natm)
        atoms = list(atoms)
        width = 16
        header = f"{'Component':<20}" + ''.join(
            f"{f'{mol.atom_symbol(ia)}{ia}':>{width}}" for ia in atoms) + f"{'Sum':>{width}}"
        lines = [f"CBS energy density analysis: scheme {self.scheme} ({self.basis_family}XZ, "
                 f"3SLF, Seino & Nakai 2016), orbital_basis='{self.orbital_basis}', "
                 f"ne_partition='{self.ne_partition}', w_occ={self.w_occ:g}, frozen={self.frozen}",
                 'Energies in hartree', header, '-' * len(header)]

        def row(label, values):
            return f"{label:<20}" + ''.join(f"{values[ia]:>{width}.8f}" for ia in atoms) \
                + f"{values.sum():>{width}.8f}"

        for x in sorted(self.hf, key=CARDINAL.get):
            lines.append(row(f'E_HF[{x}Z]', self.hf[x]))
        for key in sorted(self.corr, key=lambda k: (CARDINAL[k[1]], METHOD_LEVEL[k[0]])):
            lines.append(row(f'E_corr {key[0]}[{key[1]}Z]', self.corr[key]))
        lines.append('-' * len(header))
        hf_label = ' + '.join(f'{c:+.4f} HF[{x}Z]' for x, c in self.hf_coeff.items())
        lines.append(f'HF reference : {hf_label}')
        lines.append('CBS fit      : ' + ' '.join(
            f'{c:+.4f} {k[0]}[{k[1]}Z]' for k, c in self.coeff.items()))
        lines.append(row('E_HF (ref)', self.e_hf))
        lines.append(row('E_corr (CBS)', self.e_corr))
        lines.append(row(f'E_{self.scheme} (CBS)', self.e_tot))
        lines.append('-' * len(header))
        lines.append(f"Molecular CBS correlation energy: {self.e_corr_mol:20.10f}")
        lines.append(f"Molecular CBS total energy      : {self.e_tot_mol:20.10f}")
        lines.append(f"Sum of atomic energies          : {self.e_tot.sum():20.10f}")
        return '\n'.join(lines)

    def __repr__(self):
        return self.summary()


class CompositeEDA(lib.StreamObject):
    """Atomic energies at the CBS limit with the QDD / QTD / QTT / QTN models.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule (its basis is replaced by the hierarchical basis sets).
    scheme : {'QDD', 'QTD', 'QTT', 'QTN'}
        Fitting model (Table 7 / Table 11 of the paper).  Default 'QTD'.
    basis_family : {'cc-pv', 'aug-cc-pv'} or dict
        Hierarchical basis sets; a dict {'D': ..., 'T': ..., 'Q': ...} may be
        given together with ``coeff_family`` naming the parameter set.
    frozen : int, sequence or 'auto'
        Frozen core specification passed to MP2/CCSD; 'auto' (default)
        freezes the chemical core orbitals, as in the paper.
    orbital_basis, ne_partition, w_occ :
        Passed to the EDA of every level (see ``pyscf_eda.rhf`` /
        ``pyscf_eda.mp2``).
    hf_cbs : {'largest', 'karton-martin'}
        HF reference for the total atomic energies (see module docstring).
    conv_tol, cc_conv_tol, cc_conv_tol_normt : float
        Convergence thresholds of the SCF and CCSD calculations.
    """

    def __init__(self, mol, scheme='QTD', basis_family='cc-pv', frozen='auto',
                 orbital_basis='ao', ne_partition='half', w_occ=1.0, hf_cbs='largest',
                 coeff_family=None, conv_tol=1e-10, cc_conv_tol=1e-9, cc_conv_tol_normt=1e-7,
                 verbose=None):
        scheme = scheme.upper()
        if scheme not in SCHEMES:
            raise ValueError(f'unknown scheme {scheme}; choose from {tuple(SCHEMES)}')
        if isinstance(basis_family, dict):
            if coeff_family is None:
                raise ValueError("coeff_family ('cc-pv' or 'aug-cc-pv') is required "
                                 'with a user-defined basis dict')
            self.basis_sets = dict(basis_family)
            self.basis_family = coeff_family
        else:
            if basis_family not in BASIS_FAMILIES:
                raise ValueError(f'unknown basis family {basis_family}; '
                                 f'choose from {tuple(BASIS_FAMILIES)}')
            self.basis_sets = dict(BASIS_FAMILIES[basis_family])
            self.basis_family = basis_family
        if self.basis_family not in SCHEMES[scheme]:
            raise ValueError(f'no {scheme} parameters for basis family {self.basis_family}')
        self.mol = mol
        self.scheme = scheme
        self.coeff = dict(SCHEMES[scheme][self.basis_family])
        self.frozen = frozen
        self.orbital_basis = orbital_basis
        self.ne_partition = ne_partition
        self.w_occ = w_occ
        self.hf_cbs = hf_cbs
        self.conv_tol = conv_tol
        self.cc_conv_tol = cc_conv_tol
        self.cc_conv_tol_normt = cc_conv_tol_normt
        self.verbose = mol.verbose if verbose is None else verbose
        self.stdout = mol.stdout
        self.mfs = {}
        self.eda_results = {}
        self.result = None

    # --- planning -------------------------------------------------------
    def required_levels(self):
        """{X: highest method needed with basis X} (lower levels come for free)."""
        levels = {}
        for (method, x) in self.coeff:
            levels[x] = max(levels.get(x, 0), METHOD_LEVEL[method])
        return {x: [m for m, lv in METHOD_LEVEL.items() if lv == lv_max][0]
                for x, lv_max in levels.items()}

    def _build_mol(self, x):
        mol = self.mol.copy()
        mol.basis = self.basis_sets[x]
        mol.verbose = self.verbose
        mol.build(dump_input=False, parse_arg=False)
        return mol

    def _frozen(self, mol):
        if isinstance(self.frozen, str) and self.frozen == 'auto':
            return chemical_core(mol) or None
        return self.frozen

    # --- driver ---------------------------------------------------------
    def kernel(self):
        log = logger.new_logger(self)
        plan = self.required_levels()
        log.info('CBS scheme %s (%sXZ): %s', self.scheme, self.basis_family,
                 ', '.join(f'{m}/{x}Z' for x, m in sorted(plan.items(), key=lambda t: CARDINAL[t[0]])))
        eda_kw = dict(ne_partition=self.ne_partition, orbital_basis=self.orbital_basis)
        corr, hf, corr_mol, hf_mol = {}, {}, {}, {}
        for x in sorted(plan, key=CARDINAL.get):
            top = plan[x]
            mol = self._build_mol(x)
            frozen = self._frozen(mol)
            log.info('--- basis %s (%s), highest level %s, frozen=%s, nao=%d',
                     x, self.basis_sets[x], top, frozen, mol.nao)
            mf = scf.RHF(mol)
            mf.conv_tol = self.conv_tol
            mf.verbose = self.verbose
            mf.kernel()
            if not mf.converged:
                log.warn('SCF with %s did not converge', self.basis_sets[x])
            self.mfs[x] = mf
            hf_res = eda_rhf.EDA(mf, **eda_kw)
            hf_res.verbose = 0
            hf_res = hf_res.kernel()
            hf[x] = hf_res.e_tot
            hf_mol[x] = mf.e_tot
            self.eda_results[('HF', x)] = hf_res

            pt = pyscf_mp.MP2(mf, frozen=frozen)
            pt.verbose = self.verbose
            pt.kernel()
            res = eda_mp2.EDA(pt, w_occ=self.w_occ, **eda_kw)
            res.verbose = 0
            res = res.kernel()
            corr[('MP2', x)] = res.e_corr
            corr_mol[('MP2', x)] = pt.e_corr
            self.eda_results[('MP2', x)] = res

            if METHOD_LEVEL[top] >= 1:
                mycc = pyscf_cc.CCSD(mf, frozen=frozen)
                mycc.conv_tol = self.cc_conv_tol
                mycc.conv_tol_normt = self.cc_conv_tol_normt
                mycc.verbose = self.verbose
                mycc.kernel()
                if not mycc.converged:
                    log.warn('CCSD with %s did not converge', self.basis_sets[x])
                if top == 'CCSD(T)':
                    res = eda_ccsd_t.EDA(mycc, w_occ=self.w_occ, **eda_kw)
                    res.verbose = 0
                    res = res.kernel()
                    corr[('CCSD', x)] = res.e_ccsd
                    corr_mol[('CCSD', x)] = mycc.e_corr
                    corr[('CCSD(T)', x)] = res.e_corr
                    corr_mol[('CCSD(T)', x)] = mycc.e_corr + res.e_t.sum()
                    self.eda_results[('CCSD', x)] = res
                    self.eda_results[('CCSD(T)', x)] = res
                else:
                    res = eda_ccsd.EDA(mycc, w_occ=self.w_occ, **eda_kw)
                    res.verbose = 0
                    res = res.kernel()
                    corr[('CCSD', x)] = res.e_corr
                    corr_mol[('CCSD', x)] = mycc.e_corr
                    self.eda_results[('CCSD', x)] = res
            log.info('basis %s done: E_HF = %.10f, ' % (x, mf.e_tot) + ', '.join(
                f'E_corr({m}) = {corr_mol[(m, x)]:.10f}' for m in METHOD_LEVEL if (m, x) in corr_mol))

        hf_coeff = hf_cbs_coefficients(self.hf_cbs, list(hf))
        self.result = CBSEDAResult(self.mol, self.scheme, self.basis_family, self.coeff, hf_coeff,
                                   corr, hf, corr_mol, hf_mol, self.eda_results,
                                   self.orbital_basis, self.ne_partition, self.w_occ, self.frozen)
        diff = self.result.e_tot.sum() - self.result.e_tot_mol
        if abs(diff) > 1e-7:
            log.warn('Sum of atomic CBS energies differs from the molecular value by %.3e', diff)
        if self.verbose >= logger.INFO:
            log.info('\n%s', self.result.summary())
        return self.result

    run = kernel


def kernel(mol, scheme='QTD', **kwargs):
    """Run the composite CBS-EDA and return a ``CBSEDAResult``."""
    return CompositeEDA(mol, scheme=scheme, **kwargs).kernel()


def QDD(mol, **kwargs):
    """QDD model: MP2/D,T,Q + CCSD/D + CCSD(T)/D (three basis-set chains)."""
    return CompositeEDA(mol, scheme='QDD', **kwargs)


def QTD(mol, **kwargs):
    """QTD model: MP2/D,T,Q + CCSD/D,T + CCSD(T)/D (three basis-set chains)."""
    return CompositeEDA(mol, scheme='QTD', **kwargs)


def QTT(mol, **kwargs):
    """QTT model: MP2/D,T,Q + CCSD/D,T + CCSD(T)/D,T."""
    return CompositeEDA(mol, scheme='QTT', **kwargs)


def QTN(mol, **kwargs):
    """QTN model: MP2/D,T,Q + CCSD/D,T (no triples)."""
    return CompositeEDA(mol, scheme='QTN', **kwargs)
