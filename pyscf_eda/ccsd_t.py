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
O(o v^3 N) for the final contractions, independent of the number of atoms.
The W blocks are built for one virtual index a and a block of b at a time
(memory O(o^3 v n_b), n_b from ``max_memory``); P and the (be|ck) integrals
take O(o v^3), and the (be|cl) integrals are generated in blocks of b.
P and Q are computed by the OpenMP C kernel ``pyscf_eda/lib/ccsd_t_eda.c``
(blocks a >= b >= c as in PySCF's (T) code, BLAS dgemm through SciPy,
per-row locks on P and Q; memory O(o^3 n_c) per thread) or, if it cannot be
compiled, by the numpy implementation of the same block scheme.  For
butane/cc-pVDZ the C kernel takes about twice the (T) energy evaluation of
PySCF (the numpy version about 15-20 times), independent of the number of
atoms.

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


def _r3(w, out=None):
    """Closed-shell triples permutation operator acting on the (i, j, k) axes 0-2.

    r3(w) = 4 w_ijk + w_jki + w_kij - 2 w_kji - 2 w_ikj - 2 w_jik, evaluated
    with in-place accumulation (the trailing axes stay contiguous).
    """
    if out is None:
        out = numpy.empty_like(w)
    numpy.multiply(w, 4.0, out=out)
    out += w.transpose(1, 2, 0, 3, 4)
    out += w.transpose(2, 0, 1, 3, 4)
    tmp = w.transpose(2, 1, 0, 3, 4) + w.transpose(0, 2, 1, 3, 4)
    tmp += w.transpose(1, 0, 2, 3, 4)
    tmp *= 2.0
    out -= tmp
    return out


class _TriplesOperands:
    """Amplitudes and integrals rearranged once so that every term of W is a
    plain matrix multiplication whose natural output layout ends with the
    virtual index c (or with b, c), see ``_w_block``."""

    def __init__(self, t2, g, h):
        # g = (be|ck) as [b, e, c, k];  h = (mj|ck) as [m, j, c, k]
        self.t2 = t2
        self.g = g
        self.h = h
        self.g1 = numpy.ascontiguousarray(g.transpose(1, 3, 0, 2))     # [e, k, b, c] = (be|ck)
        self.g3 = numpy.ascontiguousarray(g.transpose(1, 3, 2, 0))     # [e, j, b, c] = (ce|bj)
        self.h2 = numpy.ascontiguousarray(h.transpose(0, 1, 3, 2))     # [m, j, k, c] = (mj|ck)
        self.h4 = numpy.ascontiguousarray(h.transpose(1, 2, 3, 0))     # [k, b, j, m] = (mk|bj)
        self.t8 = numpy.ascontiguousarray(t2.transpose(1, 0, 2, 3))    # [m, j, b, c] = t_jm^bc
        self.t9 = numpy.ascontiguousarray(t2.transpose(3, 0, 1, 2))    # [e, k, i, c] = t_ki^ce
        self.t12 = numpy.ascontiguousarray(t2.transpose(1, 0, 3, 2))   # [m, k, b, c] = t_km^cb

    def for_a(self, a):
        """Per-a slices (all small, O(o^2 v) or O(o v^2))."""
        t2, g, h = self.t2, self.g, self.h
        t2a = numpy.ascontiguousarray(t2[:, :, a, :])                 # [i, j, e]  t_ij^ae
        t2xa = t2[:, :, :, a]                                          # [j, m, b]  t_jm^ba
        gxa = g[:, :, a, :]                                            # [b, e, i]  (be|ai)
        hxa = h[:, :, a, :]                                            # [m, k, i]  (mk|ai)
        return dict(
            t2a=t2a,
            ta2=numpy.ascontiguousarray(t2a.transpose(0, 2, 1)),       # [i, b, m]  t_im^ab
            ta4=numpy.ascontiguousarray(t2a.transpose(1, 0, 2)),       # [m, i, c]  t_im^ac
            ta6=numpy.ascontiguousarray(t2xa.transpose(0, 2, 1)),      # [j, b, m]  t_jm^ba
            ta10=numpy.ascontiguousarray(t2xa.transpose(1, 0, 2)),     # [m, k, c]  t_km^ca
            ga5=numpy.ascontiguousarray(g[a].transpose(0, 2, 1)),      # [e, k, c]  (ae|ck)
            ga7=numpy.ascontiguousarray(gxa.transpose(1, 2, 0)),       # [e, i, c]  (ce|ai)
            ga9=numpy.ascontiguousarray(g[a].transpose(2, 1, 0)),      # [j, b, e]  (ae|bj)
            ga11=numpy.ascontiguousarray(gxa.transpose(2, 0, 1)),      # [i, b, e]  (be|ai)
            ha8=numpy.ascontiguousarray(hxa.transpose(1, 2, 0)),       # [k, i, m]  (mk|ai) (= [j, i, m] (mj|ai))
        )


def _w_block(t2, g, h, a, b0=None, b1=None, ops=None, opa=None, out=None):
    """W_ijk^abc = P R for fixed a and b in [b0, b1), as [i, j, k, b, c].

    The twelve terms of the six pair permutations of
    R_ijk^abc = sum_e t_ij^ae (be|ck) - sum_m t_im^ab (mj|ck) are matrix
    multiplications with pre-arranged operands; each result is accumulated
    through a transposition that keeps the last axis (c) or the last two
    axes (b, c) contiguous.
    """
    nocc, nvir = t2.shape[1], t2.shape[3]
    if b0 is None:
        b0, b1 = 0, nvir
    nb = b1 - b0
    if ops is None:
        ops = _TriplesOperands(t2, g, h)
    if opa is None:
        opa = ops.for_a(a)
    o, v = nocc, nvir
    t2a, ta2, ta4, ta6, ta10 = (opa[k] for k in ('t2a', 'ta2', 'ta4', 'ta6', 'ta10'))
    ga5, ga7, ga9, ga11, ha8 = (opa[k] for k in ('ga5', 'ga7', 'ga9', 'ga11', 'ha8'))
    t2_b = numpy.ascontiguousarray(t2[:, :, b0:b1, :]).reshape(o * o * nb, v)   # rows (j, i, b) / (j, k, b)
    dot = numpy.dot

    # R_ijk^abc : + sum_e t_ij^ae (be|ck)            -> (i j | k b c)
    w = dot(t2a.reshape(o * o, v), ops.g1[:, :, b0:b1, :].reshape(v, -1)).reshape(o, o, o, nb, v)
    if out is not None:
        out[:] = w
        w = out
    # R_ikj^acb : + sum_e t_ik^ae (ce|bj)            -> (i k | j b c)
    r = dot(t2a.reshape(o * o, v), ops.g3[:, :, b0:b1, :].reshape(v, -1)).reshape(o, o, o, nb, v)
    w += r.transpose(0, 2, 1, 3, 4)
    # R_jik^bac : + sum_e t_ji^be (ae|ck)            -> (j i b | k c)
    r = dot(t2_b, ga5.reshape(v, -1)).reshape(o, o, nb, o, v)
    w += r.transpose(1, 0, 3, 2, 4)
    # R_jki^bca : + sum_e t_jk^be (ce|ai)            -> (j k b | i c)
    r = dot(t2_b, ga7.reshape(v, -1)).reshape(o, o, nb, o, v)
    w += r.transpose(3, 0, 1, 2, 4)
    # R_kij^cab : + sum_e t_ki^ce (ae|bj)            -> (j b | k i c)
    r = dot(ga9[:, b0:b1, :].reshape(o * nb, v), ops.t9.reshape(v, -1)).reshape(o, nb, o, o, v)
    w += r.transpose(3, 0, 2, 1, 4)
    # R_kji^cba : + sum_e t_kj^ce (be|ai)            -> (i b | k j c)
    r = dot(ga11[:, b0:b1, :].reshape(o * nb, v), ops.t9.reshape(v, -1)).reshape(o, nb, o, o, v)
    w += r.transpose(0, 3, 2, 1, 4)
    # R_ijk^abc : - sum_m t_im^ab (mj|ck)            -> (i b | j k c)
    r = dot(ta2[:, b0:b1, :].reshape(o * nb, o), ops.h2.reshape(o, -1)).reshape(o, nb, o, o, v)
    w -= r.transpose(0, 2, 3, 1, 4)
    # R_ikj^acb : - sum_m t_im^ac (mk|bj)            -> (k b j | i c)
    r = dot(ops.h4[:, b0:b1].reshape(o * nb * o, o), ta4.reshape(o, -1)).reshape(o, nb, o, o, v)
    w -= r.transpose(3, 2, 0, 1, 4)
    # R_jik^bac : - sum_m t_jm^ba (mi|ck)            -> (j b | i k c)
    r = dot(ta6[:, b0:b1, :].reshape(o * nb, o), ops.h2.reshape(o, -1)).reshape(o, nb, o, o, v)
    w -= r.transpose(2, 0, 3, 1, 4)
    # R_jki^bca : - sum_m t_jm^bc (mk|ai)            -> (k i | j b c)
    t8_b = numpy.ascontiguousarray(ops.t8[:, :, b0:b1, :]).reshape(o, -1)
    r = dot(ha8.reshape(o * o, o), t8_b).reshape(o, o, o, nb, v)
    w -= r.transpose(1, 2, 0, 3, 4)
    # R_kij^cab : - sum_m t_km^ca (mi|bj)            -> (i b j | k c)
    r = dot(ops.h4[:, b0:b1].reshape(o * nb * o, o), ta10.reshape(o, -1)).reshape(o, nb, o, o, v)
    w -= r.transpose(0, 2, 3, 1, 4)
    # R_kji^cba : - sum_m t_km^cb (mj|ai)            -> (j i | k b c)
    t12_b = numpy.ascontiguousarray(ops.t12[:, :, b0:b1, :]).reshape(o, -1)
    r = dot(ha8.reshape(o * o, o), t12_b).reshape(o, o, o, nb, v)
    w -= r.transpose(1, 0, 2, 3, 4)
    return w


def _pq_intermediates_numpy(t1, t2, g_tot, h_tot, ovov, e_occ, e_vir, with_t4, max_memory, log):
    """P and Q intermediates with the numpy implementation (blocked over a and b)."""
    nocc, nvir = t1.shape
    ops = _TriplesOperands(t2, g_tot, h_tot)
    ov_kc = numpy.ascontiguousarray(ovov.transpose(0, 2, 1, 3))         # [j, k, b, c] = (jb|kc)
    eijk = lib.direct_sum('i+j+k->ijk', e_occ, e_occ, e_occ)
    ebc = lib.direct_sum('b+c->bc', e_vir, e_vir)
    keys = ('t', '4') if with_t4 else ('t',)
    p = {k: numpy.zeros((nvir, nvir, nvir, nocc)) for k in keys}      # [b, e, c, k]
    q = {k: numpy.zeros((nocc, nocc, nvir, nocc)) for k in keys}      # [m, j, c, k]
    mem_per_b = nocc ** 3 * nvir * 8 / 1e6 * (7 if with_t4 else 5)   # W, Y, D, transposed copies
    avail = max_memory - lib.current_memory()[0]
    nb_max = int(max(1, min(nvir, avail // max(mem_per_b, 1e-6))))
    log.debug('(T) partition: %d virtual orbitals b per block (%.0f MB)', nb_max, nb_max * mem_per_b)
    t2_oo = t2.reshape(nocc * nocc, nvir * nvir)
    for a in range(nvir):
        opa = ops.for_a(a)
        t2a_t = opa['t2a'].reshape(nocc * nocc, nvir).T                # [e, (i j)]
        t1a = t1[:, a]
        for b0, b1 in lib.prange(0, nvir, nb_max):
            nb = b1 - b0
            w = _w_block(t2, g_tot, h_tot, a, b0, b1, ops, opa)          # [i, j, k, b, c]
            z = t1a[:, None, None, None, None] * ov_kc[None, :, :, b0:b1, :]
            z += t1[None, :, None, b0:b1, None] * ovov[:, a][:, None, :, None, :]
            z += t1[None, None, :, None, :] * ovov[:, a, :, b0:b1][:, :, None, :, None]
            inv_d = lib.direct_sum('ijk-bc->ijkbc', eijk, ebc[b0:b1] + e_vir[a])
            numpy.reciprocal(inv_d, out=inv_d)
            inv_d *= 2.0
            blocks = []
            if with_t4:
                y4 = _r3(w)
                y4 *= inv_d
                blocks.append(('4', y4))
            w += z
            y = _r3(w, out=z)
            y *= inv_d
            blocks.append(('t', y))
            w = inv_d = None
            for key, yb in blocks:
                # P_beck += sum_ij Y_ijkbc t_ij^ae                         -> [e, (k b c)]
                pe = t2a_t.dot(yb.reshape(nocc * nocc, -1)).reshape(nvir, nocc, nb, nvir)
                p[key][b0:b1] += pe.transpose(2, 0, 3, 1)
                # Q_mjck += sum_ib Y_ijkbc t_im^ab                          -> [(j k c), m]
                y_ib = numpy.ascontiguousarray(yb.transpose(1, 2, 4, 0, 3)).reshape(-1, nocc * nb)
                qm = y_ib.dot(opa['ta2'][:, b0:b1, :].reshape(nocc * nb, nocc))
                q[key] += qm.reshape(nocc, nocc, nvir, nocc).transpose(3, 0, 2, 1)
            blocks = y = y4 = z = y_ib = None
        log.debug1('(T) partition: virtual block %d/%d done', a + 1, nvir)
    return p, q


def _pq_intermediates_c(lib_c, t1, t2, g_tot, h_tot, ovov, e_occ, e_vir, with_t4,
                        max_memory, log):
    """P and Q intermediates from the compiled kernel (None if unavailable)."""
    import ctypes
    from pyscf_eda import lib as eda_lib
    nocc, nvir = t1.shape
    t1 = numpy.ascontiguousarray(t1, dtype=numpy.float64)
    t2 = numpy.ascontiguousarray(t2, dtype=numpy.float64)
    t2c = numpy.ascontiguousarray(t2.transpose(2, 0, 1, 3))        # [c][k][i][e] = t_ki^ce
    t2ackm = numpy.ascontiguousarray(t2.transpose(3, 2, 0, 1))     # [a][c][k][m] = t_km^ca
    g = numpy.ascontiguousarray(g_tot, dtype=numpy.float64)        # [b][e][c][k] = (be|ck)
    hT = numpy.ascontiguousarray(h_tot.transpose(0, 2, 1, 3))      # [m][c][j][k] = (mj|ck)
    hB = numpy.ascontiguousarray(h_tot.transpose(2, 0, 1, 3))      # [b][m][k][j] = (mk|bj)
    ovov = numpy.ascontiguousarray(ovov, dtype=numpy.float64)
    e_occ = numpy.ascontiguousarray(e_occ, dtype=numpy.float64)
    e_vir = numpy.ascontiguousarray(e_vir, dtype=numpy.float64)
    p = numpy.zeros((nvir, nvir, nvir, nocc))
    q = numpy.zeros((nocc, nocc, nvir, nocc))
    p4 = numpy.zeros((nvir, nvir, nvir, nocc)) if with_t4 else None
    q4 = numpy.zeros((nocc, nocc, nvir, nocc)) if with_t4 else None
    # block size in c: per thread 5-6 arrays of nc*o^3 doubles
    nthreads = eda_lib.num_threads()
    per_c_mb = nocc ** 3 * 8 / 1e6 * (6 if with_t4 else 5) * nthreads
    avail = max_memory - lib.current_memory()[0]
    nc_max = int(max(1, min(nvir, avail // max(per_c_mb, 1e-6))))
    nc_max = min(nc_max, 64)
    dgemm = eda_lib.dgemm_pointer()
    log.debug('(T) partition (C kernel): %d threads, c-block %d, BLAS dgemm %s',
              nthreads, nc_max, 'yes' if dgemm else 'no (C loops)')
    ptr = lambda arr: arr.ctypes.data_as(ctypes.c_void_p)
    null = ctypes.c_void_p()

    def run():
        return lib_c.ccsd_t_eda_partition(
            ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(int(with_t4)), ctypes.c_int(nc_max),
            ptr(e_occ), ptr(e_vir), ptr(t1), ptr(t2), ptr(t2c), ptr(t2ackm), ptr(g), ptr(hT), ptr(hB),
            ptr(ovov), ptr(p), ptr(q), ptr(p4) if with_t4 else null, ptr(q4) if with_t4 else null,
            ctypes.c_void_p(dgemm) if dgemm else null)

    if dgemm and nthreads > 1:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            threadpool_limits = None
            log.warn('threadpoolctl is not installed: BLAS threads are not limited inside the '
                     'OpenMP kernel; install threadpoolctl to avoid oversubscription')
        if threadpool_limits is not None:
            with threadpool_limits(limits=1, user_api='blas'):
                err = run()
        else:
            err = run()
    else:
        err = run()
    if err != 0:
        raise MemoryError('ccsd_t_eda_partition failed to allocate its buffers')
    out = {'t': (p, q)}
    if with_t4:
        out['4'] = (p4, q4)
    return out


def triples_by_atom(mycc, x=None, t1=None, t2=None, mo_energy=None, verbose=None,
                    max_memory=None, with_t4=True, backend='auto'):
    """Atomic partitions of the (T) correction.

    Parameters
    ----------
    with_t4 : bool
        Also split every atomic (T) energy into E_T[4]^A and E_ST[5]^A
        (default True; costs about 20 % more).
    backend : {'auto', 'c', 'numpy'}
        'c' uses the compiled OpenMP kernel (``pyscf_eda/lib``), 'numpy'
        the pure numpy implementation; 'auto' takes the kernel when it can
        be loaded (or compiled) and falls back to numpy otherwise.

    Returns
    -------
    dict with keys 'occ' and 'vir' (U^{0,0} and U^{2,2} partitions); each
    value is a tuple (e_t, e_t4) of arrays (natm,) with the total (T)
    correction and its fourth-order part (None if ``with_t4`` is False).
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
    t2 = numpy.ascontiguousarray(t2)
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

    lib_c = None
    if backend in ('auto', 'c'):
        from pyscf_eda import lib as eda_lib
        lib_c = eda_lib.load()
        if lib_c is None and backend == 'c':
            raise RuntimeError('the compiled (T) partition kernel is not available')
    elif backend != 'numpy':
        raise ValueError("backend must be 'auto', 'c' or 'numpy'")
    keys = ('t', '4') if with_t4 else ('t',)
    if lib_c is not None:
        pq = _pq_intermediates_c(lib_c, t1, t2, g_tot, h_tot, ovov, e_occ, e_vir, with_t4,
                                 max_memory, log)
        p = {k: pq[k][0] for k in keys}
        q = {k: pq[k][1] for k in keys}
    else:
        p, q = _pq_intermediates_numpy(t1, t2, g_tot, h_tot, ovov, e_occ, e_vir, with_t4,
                                       max_memory, log)

    # --- contractions with the integrals carrying the one-centre index l
    def block_size(n, per_unit_mb):
        avail = max_memory - lib.current_memory()[0]
        return int(max(1, min(n, avail // max(per_unit_mb * 1.5, 1e-6))))

    pg = {k: numpy.zeros((nocc, nao)) for k in keys}      # occ: [k, l]
    qh = {k: numpy.zeros((nocc, nao)) for k in keys}
    pgv = {k: numpy.zeros((nvir, nao)) for k in keys}     # vir: [c, l]
    qhv = {k: numpy.zeros((nvir, nao)) for k in keys}
    mb_b = nvir * nvir * nao * 8 / 1e6
    blk_b = block_size(nvir, 2 * mb_b)
    rows = corr.transform_by_rows
    for b0, b1, g in rows(transform, (c_vir, c_vir, c_vir, x), blk_b, max_memory):     # (be|cl)
        for k in keys:
            pg[k] += p[k][b0:b1].reshape(-1, nocc).T.dot(g.reshape(-1, nao))
    g = None
    for b0, b1, g in rows(transform, (c_vir, c_vir, x, c_occ), blk_b, max_memory):     # (be|lk)
        nb = b1 - b0
        g = numpy.ascontiguousarray(g.reshape(nb, nvir, nao, nocc).transpose(0, 1, 3, 2)).reshape(-1, nao)
        for k in keys:
            pb = numpy.ascontiguousarray(p[k][b0:b1].transpose(2, 0, 1, 3)).reshape(nvir, -1)  # [c, (b e k)]
            pgv[k] += pb.dot(g)
    g = pb = None
    mb_m = nocc * nvir * nao * 8 / 1e6
    blk_m = block_size(nocc, 2 * mb_m)
    for m0, m1, h in rows(transform, (c_occ, c_occ, c_vir, x), blk_m, max_memory):     # (mj|cl)
        for k in keys:
            qh[k] += q[k][m0:m1].reshape(-1, nocc).T.dot(h.reshape(-1, nao))
    h = None
    for m0, m1, h in rows(transform, (c_occ, c_occ, x, c_occ), blk_m, max_memory):     # (mj|lk)
        nm = m1 - m0
        h = numpy.ascontiguousarray(h.reshape(nm, nocc, nao, nocc).transpose(0, 1, 3, 2)).reshape(-1, nao)
        for k in keys:
            qm = numpy.ascontiguousarray(q[k][m0:m1].transpose(2, 0, 1, 3)).reshape(nvir, -1)  # [c, (m j k)]
            qhv[k] += qm.dot(h)
    h = qm = None

    aoslice = mol.aoslice_by_atom()

    def by_atom(e_l):
        return numpy.array([e_l[p0:p1].sum() for (_, _, p0, p1) in aoslice])

    result = {}
    for name, coeff, pp, qq in (('occ', cp_occ, pg, qh), ('vir', cp_vir, pgv, qhv)):
        e_t = by_atom(numpy.einsum('lk,kl->l', coeff, pp['t'] - qq['t']))
        e4 = by_atom(numpy.einsum('lk,kl->l', coeff, pp['4'] - qq['4'])) if with_t4 else None
        result[name] = (e_t, e4)
        if with_t4:
            log.debug('(T) partition %s: E_T[4] = %s  E_ST[5] = %s', name, e4, e_t - e4)
        else:
            log.debug('(T) partition %s: E_(T) = %s', name, e_t)
    return result


class EDA(eda_ccsd.EDA):
    """CCSD(T) energy density analysis for a converged closed-shell CCSD object.

    Parameters are those of ``pyscf_eda.ccsd.EDA``; ``w_occ`` weights the
    occupied-side partitions (CCSD: occupied orbital i; (T): U^{0,0}) against
    the virtual-side ones (CCSD: virtual orbital a; (T): U^{2,2}).
    ``with_t4`` (default True) also resolves E_(T)^A into E_T[4]^A and
    E_ST[5]^A; ``with_t4=False`` saves about 20 % of the partition time.

    Examples
    --------
    >>> from pyscf import gto, scf, cc
    >>> from pyscf_eda import ccsd_t as eda_ccsd_t
    >>> mycc = cc.CCSD(mf).run()
    >>> res = eda_ccsd_t.EDA(mycc).kernel()
    >>> print(res.e_t, res.e_tot)
    """

    result_class = CCSDTEDAResult

    def __init__(self, mycc, ne_partition='half', orbital_basis='nao', w_occ=1.0,
                 with_singles=False, with_t4=True):
        eda_ccsd.EDA.__init__(self, mycc, ne_partition=ne_partition, orbital_basis=orbital_basis,
                              w_occ=w_occ, with_singles=with_singles)
        self.with_t4 = with_t4

    def _corr_partition(self):
        ccsd_occ, ccsd_vir = eda_ccsd.corr_energy_by_atom(
            self._cc, x=self.orth_coeff, with_singles=self.with_singles, verbose=self.verbose)
        parts = triples_by_atom(self._cc, x=self.orth_coeff, verbose=self.verbose,
                                with_t4=self.with_t4)
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
