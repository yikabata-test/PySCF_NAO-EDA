"""Energy density analysis (EDA) for closed-shell CCSD(T).

Reference
---------
M. Kobayashi, H. Nakai, "Divide-and-conquer-based linear-scaling approach for
traditional and renormalized coupled cluster methods with single, double, and
noniterative triple excitations", J. Chem. Phys. 131, 114108 (2009),
Sec. II B.

The (T) correction consists of the fourth-order triples term E_T[4] and the
fifth-order singles-triples term E_ST[5] (Eqs. 9, 10).  Both contain the
common factor t(2)_ijk,abc D_ijk,abc, i.e. the connected-triples
"numerator"

    W_ijk^abc = P_ijk^abc [ sum_e t_ij^ae (be|ck) - sum_m t_im^ab (mj|ck) ]

(chemists' notation, P the six-fold symmetrizer of the pairs (ia),(jb),(kc)),
with D = e_i + e_j + e_k - e_a - e_b - e_c and

    Z_ijk^abc = t_ia (bj|ck) + t_jb (ai|ck) + t_kc (ai|bj)    (z = Z / D).

Writing Y[T] = [4/3 T_abc - 2 T_acb + 2/3 T_bca] / D,

    E_T[4]  = sum_{ijk abc} Y[W] W,      E_ST[5] = sum_{ijk abc} Y[Z] W.

Atomic partition (Eqs. 14, 15, 21-27)
-------------------------------------
The common factor W is partitioned by EDA: one MO in its 3/4-transformed
two-electron integrals is replaced by the partial sum over the basis
functions (or NAOs) of atom A.  With the raw (unsymmetrized) term

    R_ijk^abc = sum_e t_ij^ae (be|ck) - sum_m t_im^ab (mj|ck),   W = P R,

the paper's U^{0,0} partition (occupied orbital k in both integrals,
adopted in the paper) is

    R^A_ijk^abc = sum_e t_ij^ae (be|c k_A) - sum_m t_im^ab (mj|c k_A),
    (pq|r k_A) = sum_{l in A} C'_{l k} (pq|r l),

and the U^{2,2} partition (virtual orbital c in both integrals) uses
(pq|c_A k) = sum_{l in A} C'_{l c} (pq|l k) instead.  Since P Y = 2 r3(T)/D
with r3 the closed-shell triples permutation operator, the atomic energies
can be evaluated with the unsymmetrized R^A alone:

    E_(T)^A = 2 sum_{ijk abc} r3(W + Z)_ijk^abc R^A_ijk^abc / D_ijk^abc ,

with sum_A E_(T)^A = E_(T) exactly, and E_T[4]^A, E_ST[5]^A from r3(W)
and r3(Z) respectively.  ``w_occ`` mixes the occupied (U^{0,0}) and
virtual (U^{2,2}) partitions as for the CCSD part; w_occ = 1 is the default.
The U^{X,Y} variants with X = 1 or Y = 1 are not implemented.

Implementation
--------------
Instead of forming R^A for every atom, the atom-independent contraction of
Y = 2 r3(W + Z)/D (and of the E_T[4] part 2 r3(W)/D) with the amplitudes
is done first, virtual block a by virtual block a:

    P_beck = sum_{ija} Y_ijk^abc t_ij^ae,     Q_mjck = sum_{iab} Y_ijk^abc t_im^ab ,

after which the atomic energies are single contractions with the
three-quarter transformed integrals carrying the one-centre index l:

    U^{0,0}: E_(T)^A = sum_{l in A} sum_k C'_{lk} [ sum_{bec} P_beck (be|cl)
                                                 - sum_{mjc} Q_mjck (mj|cl) ]
    U^{2,2}: E_(T)^A = sum_{l in A} sum_c C'_{lc} [ sum_{bek} P_beck (be|lk)
                                                 - sum_{mjk} Q_mjck (mj|lk) ] .

The work is O(o^3 v^4) for P (the cost of one (T) evaluation) plus
O(o v^3 N) for the final contractions, independent of the number of atoms;
memory is O(o^3 v^2) per virtual block, O(o v^3) for P and the (be|ck)
integrals, and the (be|cl) integrals are generated in blocks of b bounded
by ``max_memory``.

The renormalized R-CCSD(T) variants of the paper are not implemented.
"""

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf_eda import corr
from pyscf_eda import ccsd as eda_ccsd
from pyscf_eda.corr import CorrEDAResult


class CCSDTEDAResult(CorrEDAResult):
    """Atomic energy densities of a CCSD(T)-EDA calculation.

    In addition to the attributes of ``CorrEDAResult``:

    e_ccsd      : CCSD correlation part E_corr(CCSD)^A
    e_t         : (T) correction E_(T)^A = E_T[4]^A + E_ST[5]^A
    e_t4, e_st5 : its fourth-order and fifth-order components
    e_corr      : E_corr^A = E_corr(CCSD)^A + E_(T)^A
    e_tot       : E_CCSD(T)^A = E_HF^A + E_corr^A
    """

    method = 'CCSD(T)'
    components = ('e_nn', 'e_kin', 'e_ne', 'e_1el', 'e_coul', 'e_x', 'e_elec',
                  'e_hf', 'e_ccsd', 'e_t', 'e_corr', 'e_tot')

    def __init__(self, hf_result, ccsd_occ, ccsd_vir, t_occ, t_vir, w_occ,
                 t4_occ=None, t4_vir=None, e_tot_ref=None):
        CorrEDAResult.__init__(self, hf_result, ccsd_occ + t_occ, ccsd_vir + t_vir,
                               w_occ, e_tot_ref=e_tot_ref)
        w_vir = 1.0 - w_occ
        self.e_ccsd_occ = ccsd_occ
        self.e_ccsd_vir = ccsd_vir
        self.e_t_occ = t_occ
        self.e_t_vir = t_vir
        self.e_ccsd = w_occ * ccsd_occ + w_vir * ccsd_vir
        self.e_t = w_occ * t_occ + w_vir * t_vir
        if t4_occ is not None:
            self.e_t4 = w_occ * t4_occ + w_vir * t4_vir
            self.e_st5 = self.e_t - self.e_t4
        else:
            self.e_t4 = self.e_st5 = None

    @property
    def labels(self):
        return dict(CorrEDAResult.labels.fget(self), e_ccsd='E_corr(CCSD)', e_t='E_(T)')


def _r3(w):
    """Closed-shell triples permutation operator acting on the (i, j, k) axes 0-2."""
    return (4 * w + w.transpose(1, 2, 0, 3, 4) + w.transpose(2, 0, 1, 3, 4)
            - 2 * w.transpose(2, 1, 0, 3, 4) - 2 * w.transpose(0, 2, 1, 3, 4)
            - 2 * w.transpose(1, 0, 2, 3, 4))


def _w_block(t2, g, h, a):
    """W_ijk^abc = P R for fixed a, from the six pair permutations of R."""
    t2a = t2[:, :, a, :]        # t_ij^{a e}
    t2xa = t2[:, :, :, a]       # t_ij^{e a}
    ga = g[a]                   # (ae|ck)
    gxa = g[:, :, a, :]         # (be|ak)
    hxa = h[:, :, a, :]         # (mj|ak)
    w = lib.einsum('ije,beck->ijkbc', t2a, g)          # R_ijk^abc
    w -= lib.einsum('imb,mjck->ijkbc', t2a, h)
    w += lib.einsum('ike,cebj->ijkbc', t2a, g)         # R_ikj^acb
    w -= lib.einsum('imc,mkbj->ijkbc', t2a, h)
    w += lib.einsum('jibe,eck->ijkbc', t2, ga)         # R_jik^bac
    w -= lib.einsum('jmb,mick->ijkbc', t2xa, h)
    w += lib.einsum('jkbe,cei->ijkbc', t2, gxa)        # R_jki^bca
    w -= lib.einsum('jmbc,mki->ijkbc', t2, hxa)
    w += lib.einsum('kice,ebj->ijkbc', t2, ga)         # R_kij^cab
    w -= lib.einsum('kmc,mibj->ijkbc', t2xa, h)
    w += lib.einsum('kjce,bei->ijkbc', t2, gxa)        # R_kji^cba
    w -= lib.einsum('kmcb,mji->ijkbc', t2, hxa)
    return w


def triples_by_atom(mycc, x=None, t1=None, t2=None, mo_energy=None, verbose=None,
                    max_memory=None):
    """Atomic partitions of the (T) correction.

    Returns
    -------
    dict with keys 'occ' and 'vir' (U^{0,0} and U^{2,2} partitions); each
    value is a tuple (e_t, e_t4) of arrays (natm,) with the total (T)
    correction and its fourth-order part per atom.
    """
    log = logger.new_logger(mycc, verbose)
    mol = mycc.mol
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if t1 is None or t2 is None:
        raise RuntimeError('CCSD amplitudes are not available; run the CCSD first')
    t1 = numpy.asarray(t1)
    t2 = numpy.asarray(t2)
    c_occ, c_vir = corr.active_orbitals(mycc)
    nocc = c_occ.shape[1]
    nvir = c_vir.shape[1]
    if mo_energy is None:
        mo_energy = numpy.asarray(mycc._scf.mo_energy)[mycc.get_frozen_mask()]
    e_occ, e_vir = mo_energy[:nocc], mo_energy[nocc:]

    nao = mol.nao_nr()
    if x is None:
        x = numpy.eye(nao)
    xinv = numpy.linalg.inv(x)
    cp_occ = xinv.dot(c_occ)
    cp_vir = xinv.dot(c_vir)
    transform = corr.eri_transformer(mycc)
    if max_memory is None:
        max_memory = mycc.max_memory

    ovov = transform((c_occ, c_vir, c_occ, c_vir)).reshape(nocc, nvir, nocc, nvir)
    g_tot = transform((c_vir, c_vir, c_vir, c_occ)).reshape(nvir, nvir, nvir, nocc)   # (be|ck)
    h_tot = transform((c_occ, c_occ, c_vir, c_occ)).reshape(nocc, nocc, nvir, nocc)   # (mj|ck)

    # --- atom-independent part: P_beck and Q_mjck for Y4 = 2 r3(W)/D and Y5 = 2 r3(Z)/D
    eijk = lib.direct_sum('i+j+k->ijk', e_occ, e_occ, e_occ)
    ebc = lib.direct_sum('b+c->bc', e_vir, e_vir)
    p4 = numpy.zeros((nvir, nvir, nvir, nocc))        # [b, e, c, k]
    p5 = numpy.zeros((nvir, nvir, nvir, nocc))
    q4 = numpy.zeros((nocc, nocc, nvir, nocc))        # [m, j, c, k]
    q5 = numpy.zeros((nocc, nocc, nvir, nocc))
    for a in range(nvir):
        d3 = lib.direct_sum('ijk-bc->ijkbc', eijk, ebc + e_vir[a])
        w = _w_block(t2, g_tot, h_tot, a)
        z = (numpy.einsum('i,jbkc->ijkbc', t1[:, a], ovov)
             + numpy.einsum('jb,ikc->ijkbc', t1, ovov[:, a])
             + numpy.einsum('kc,ijb->ijkbc', t1, ovov[:, a]))
        t2a = t2[:, :, a, :]                                          # [i, j, e]  (= t_ij^ae)
        t2a_ib = numpy.ascontiguousarray(t2a.transpose(0, 2, 1))      # [i, b, m]  (= t_im^ab)
        for y, p, q in ((2.0 * _r3(w) / d3, p4, q4), (2.0 * _r3(z) / d3, p5, q5)):
            # P_beck += sum_ij Y_ijkbc t_ij^ae
            pe = t2a.reshape(nocc * nocc, nvir).T.dot(y.reshape(nocc * nocc, -1))   # [e, (k b c)]
            p += pe.reshape(nvir, nocc, nvir, nvir).transpose(2, 0, 3, 1)           # -> [b, e, c, k]
            # Q_mjck += sum_ib Y_ijkbc t_im^ab
            y_ib = numpy.ascontiguousarray(y.transpose(1, 2, 4, 0, 3))              # [j, k, c, i, b]
            qm = y_ib.reshape(-1, nocc * nvir).dot(t2a_ib.reshape(nocc * nvir, nocc))  # [(j k c), m]
            q += qm.reshape(nocc, nocc, nvir, nocc).transpose(3, 0, 2, 1)           # -> [m, j, c, k]
        log.debug1('(T) partition: virtual block %d/%d done', a + 1, nvir)
    w = z = y = y_ib = d3 = None

    # --- contractions with the integrals carrying the one-centre index l
    def blocks(n, per_unit_mb):
        avail = max_memory - lib.current_memory()[0]
        blk = int(max(1, min(n, avail // max(per_unit_mb * 1.5, 1e-6))))
        return lib.prange(0, n, blk)

    pg = {name: numpy.zeros((nocc, nao)) for name in ('4', '5')}     # occ: [k, l]
    qh = {name: numpy.zeros((nocc, nao)) for name in ('4', '5')}
    pgv = {name: numpy.zeros((nvir, nao)) for name in ('4', '5')}    # vir: [c, l]
    qhv = {name: numpy.zeros((nvir, nao)) for name in ('4', '5')}
    mb_b = nvir * nvir * nao * 8 / 1e6
    for b0, b1 in blocks(nvir, 2 * mb_b):
        nb = b1 - b0
        g = transform((c_vir[:, b0:b1], c_vir, c_vir, x)).reshape(nb, nvir, nvir, nao)   # (be|cl)
        for name, p in (('4', p4), ('5', p5)):
            pg[name] += p[b0:b1].reshape(-1, nocc).T.dot(g.reshape(-1, nao))
        g = transform((c_vir[:, b0:b1], c_vir, x, c_occ)).reshape(nb, nvir, nao, nocc)   # (be|lk)
        g = numpy.ascontiguousarray(g.transpose(0, 1, 3, 2)).reshape(-1, nao)            # [(b e k), l]
        for name, p in (('4', p4), ('5', p5)):
            pb = numpy.ascontiguousarray(p[b0:b1].transpose(2, 0, 1, 3)).reshape(nvir, -1)  # [c, (b e k)]
            pgv[name] += pb.dot(g)
        g = pb = None
    mb_m = nocc * nvir * nao * 8 / 1e6
    for m0, m1 in blocks(nocc, 2 * mb_m):
        nm = m1 - m0
        h = transform((c_occ[:, m0:m1], c_occ, c_vir, x)).reshape(nm, nocc, nvir, nao)   # (mj|cl)
        for name, q in (('4', q4), ('5', q5)):
            qh[name] += q[m0:m1].reshape(-1, nocc).T.dot(h.reshape(-1, nao))
        h = transform((c_occ[:, m0:m1], c_occ, x, c_occ)).reshape(nm, nocc, nao, nocc)   # (mj|lk)
        h = numpy.ascontiguousarray(h.transpose(0, 1, 3, 2)).reshape(-1, nao)            # [(m j k), l]
        for name, q in (('4', q4), ('5', q5)):
            qm = numpy.ascontiguousarray(q[m0:m1].transpose(2, 0, 1, 3)).reshape(nvir, -1)  # [c, (m j k)]
            qhv[name] += qm.dot(h)
        h = qm = None

    aoslice = mol.aoslice_by_atom()

    def by_atom(e_l):
        return numpy.array([e_l[p0:p1].sum() for (_, _, p0, p1) in aoslice])

    result = {}
    for name, coeff, pp, qq in (('occ', cp_occ, pg, qh), ('vir', cp_vir, pgv, qhv)):
        e4 = by_atom(numpy.einsum('lk,kl->l', coeff, pp['4'] - qq['4']))
        e5 = by_atom(numpy.einsum('lk,kl->l', coeff, pp['5'] - qq['5']))
        result[name] = (e4 + e5, e4)
        log.debug('(T) partition %s: E_T[4] = %s  E_ST[5] = %s', name, e4, e5)
    return result


class EDA(eda_ccsd.EDA):
    """CCSD(T) energy density analysis for a converged closed-shell CCSD object.

    Parameters are those of ``pyscf_eda.ccsd.EDA``; ``w_occ`` weights the
    occupied-side partitions (CCSD: occupied orbital i; (T): U^{0,0}) against
    the virtual-side ones (CCSD: virtual orbital a; (T): U^{2,2}).

    Examples
    --------
    >>> from pyscf import gto, scf, cc
    >>> from pyscf_eda import ccsd_t as eda_ccsd_t
    >>> mycc = cc.CCSD(mf).run()
    >>> res = eda_ccsd_t.EDA(mycc).kernel()
    >>> print(res.e_t, res.e_tot)
    """

    result_class = CCSDTEDAResult

    def _corr_partition(self):
        ccsd_occ, ccsd_vir = eda_ccsd.corr_energy_by_atom(
            self._cc, x=self.orth_coeff, with_singles=self.with_singles, verbose=self.verbose)
        parts = triples_by_atom(self._cc, x=self.orth_coeff, verbose=self.verbose)
        self._parts = (ccsd_occ, ccsd_vir, parts)
        return ccsd_occ + parts['occ'][0], ccsd_vir + parts['vir'][0]

    def _make_result(self, hf_res, e_occ, e_vir, obj):
        ccsd_occ, ccsd_vir, parts = self._parts
        (t_occ, t4_occ), (t_vir, t4_vir) = parts['occ'], parts['vir']
        return self.result_class(hf_res, ccsd_occ, ccsd_vir, t_occ, t_vir, self.w_occ,
                                 t4_occ=t4_occ, t4_vir=t4_vir, e_tot_ref=self._e_tot_ref())

    def _e_corr_ref(self):
        mycc = self._cc
        if getattr(self, '_e_t_ref', None) is None:
            self._e_t_ref = mycc.ccsd_t()
        return mycc.e_corr + self._e_t_ref

    def _e_tot_ref(self):
        return self._scf.e_tot + self._e_corr_ref()

    @property
    def e_t(self):
        """(T) correction of the whole molecule (from PySCF's ccsd_t)."""
        return self._e_corr_ref() - self._cc.e_corr


def kernel(mycc, ne_partition='half', orbital_basis='nao', w_occ=1.0):
    """Perform the CCSD(T)-EDA and return a ``CCSDTEDAResult``."""
    return EDA(mycc, ne_partition=ne_partition, orbital_basis=orbital_basis, w_occ=w_occ).kernel()


def atom_energies(mycc, ne_partition='half', orbital_basis='nao', w_occ=1.0):
    """Atomic CCSD(T) total energies E_CCSD(T)^A (hartree)."""
    return kernel(mycc, ne_partition, orbital_basis, w_occ).e_tot


def nao_eda(mycc, ne_partition='half', w_occ=1.0):
    """NAO-based CCSD(T)-EDA."""
    return kernel(mycc, ne_partition=ne_partition, orbital_basis='nao', w_occ=w_occ)
