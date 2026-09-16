/* Atomic partition of the CCSD(T) triples correction: atom-independent
 * intermediates P_beck and Q_mjck (see pyscf_eda/ccsd_t.py).
 *
 * For every block (a >= b >= c0..c1) the symmetrised triples numerator
 * W_ijk^abc, the singles term Z and Y = 2 w r3(W [+ Z]) / D are formed and
 * contracted back with the amplitudes into P and Q, the adjoint of the
 * six pair permutations of R_ijk^abc = sum_e t_ij^ae (be|ck) - sum_m t_im^ab (mj|ck).
 * Matrix products go through a BLAS dgemm function pointer (Fortran
 * interface, 32-bit integers) when one is supplied, otherwise through a
 * plain C loop.  Blocks are distributed over OpenMP threads; the shared
 * P and Q arrays are updated under per-row locks.
 */
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef void (*dgemm_fn)(const char *, const char *, const int *, const int *, const int *,
                         const double *, const double *, const int *, const double *,
                         const int *, const double *, double *, const int *);

static dgemm_fn g_dgemm = NULL;

/* row-major C[M,N] = alpha * op(A) op(B) + beta * C, op(X) = X or X^T */
static void gemm_rm(char ta, char tb, int m, int n, int k, double alpha,
                    const double *a, int lda, const double *b, int ldb,
                    double beta, double *c, int ldc)
{
    if (g_dgemm != NULL) {
        g_dgemm(&tb, &ta, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
        return;
    }
    int i, j, l;
    for (i = 0; i < m; i++) {
        double *ci = c + (size_t)i * ldc;
        if (beta == 0.0) for (j = 0; j < n; j++) ci[j] = 0.0;
        else if (beta != 1.0) for (j = 0; j < n; j++) ci[j] *= beta;
        for (l = 0; l < k; l++) {
            double ail = (ta == 'N') ? a[(size_t)i * lda + l] : a[(size_t)l * lda + i];
            ail *= alpha;
            if (tb == 'N') {
                const double *bl = b + (size_t)l * ldb;
                for (j = 0; j < n; j++) ci[j] += ail * bl[j];
            } else {
                for (j = 0; j < n; j++) ci[j] += ail * b[(size_t)j * ldb + l];
            }
        }
    }
}

#ifdef _OPENMP
static omp_lock_t *plock = NULL, *qlock = NULL;
#define LOCK(L, r)   omp_set_lock(&(L)[r])
#define UNLOCK(L, r) omp_unset_lock(&(L)[r])
#else
#define LOCK(L, r)
#define UNLOCK(L, r)
#endif

typedef struct {
    int o, v, nc;
    double *w, *z, *y, *y4, *tmp, *yt1, *yt2, *yt3, *pacc, *qacc;
    double *t2ab, *t2ba, *tac, *tbc, *tca, *tcb;
} Buffers;

/* ---------------------------------------------------------------- W block */
static void build_w(const Buffers *B, int a, int b, int c0, int c1,
                    const double *t2, const double *t2c, const double *t2ackm,
                    const double *g, const double *hT, const double *hB)
{
    const int o = B->o, v = B->v, nc = c1 - c0;
    const size_t o2 = (size_t)o * o, o3 = o2 * o, v2 = (size_t)v * v;
    double *w = B->w, *tmp = B->tmp;
    int i, j, k, c, m;

    memset(w, 0, sizeof(double) * nc * o3);
    /* R1 and R3: t_ij^ae (be|ck), t_ik^ae (be|cj)  -> tmp[(i,j)][(c,k)] */
    gemm_rm('N', 'N', o * o, nc * o, v, 1.0, t2 + (size_t)a * v, (int)v2,
            g + ((size_t)b * v) * v * o + (size_t)c0 * o, v * o, 0.0, tmp, nc * o);
    for (i = 0; i < o; i++) for (j = 0; j < o; j++) for (c = 0; c < nc; c++) {
        const double *t = tmp + ((i * o + j) * (size_t)nc + c) * o;
        double *wc = w + (size_t)c * o3 + i * o2 + j * o;
        for (k = 0; k < o; k++) wc[k] += t[k];
    }
    /* R3 needs (ce|bj) with c varying: one product per c */
    for (c = 0; c < nc; c++) {
        gemm_rm('N', 'N', o * o, o, v, 1.0, t2 + (size_t)a * v, (int)v2,
                g + (((size_t)(c0 + c) * v) * v + b) * o, v * o, 0.0, tmp, o);
        for (i = 0; i < o; i++) for (k = 0; k < o; k++) {
            const double *t = tmp + (i * o + k) * (size_t)o;
            double *wc = w + (size_t)c * o3 + i * o2;
            for (j = 0; j < o; j++) wc[j * o + k] += t[j];
        }
    }
    /* R5: t_ji^be (ae|ck) -> tmp[(j,i)][(c,k)] */
    gemm_rm('N', 'N', o * o, nc * o, v, 1.0, t2 + (size_t)b * v, (int)v2,
            g + ((size_t)a * v) * v * o + (size_t)c0 * o, v * o, 0.0, tmp, nc * o);
    for (j = 0; j < o; j++) for (i = 0; i < o; i++) for (c = 0; c < nc; c++) {
        const double *t = tmp + ((j * o + i) * (size_t)nc + c) * o;
        double *wc = w + (size_t)c * o3 + i * o2 + j * o;
        for (k = 0; k < o; k++) wc[k] += t[k];
    }
    /* R7: t_jk^be (ce|ai): one product per c -> tmp[(j,k)][i] */
    for (c = 0; c < nc; c++) {
        gemm_rm('N', 'N', o * o, o, v, 1.0, t2 + (size_t)b * v, (int)v2,
                g + (((size_t)(c0 + c) * v) * v + a) * o, v * o, 0.0, tmp, o);
        for (j = 0; j < o; j++) for (k = 0; k < o; k++) {
            const double *t = tmp + (j * o + k) * (size_t)o;
            double *wc = w + (size_t)c * o3 + j * o + k;
            for (i = 0; i < o; i++) wc[i * o2] += t[i];
        }
    }
    /* R9: t_ki^ce (ae|bj) -> tmp[(c,k,i)][j] */
    gemm_rm('N', 'N', nc * o * o, o, v, 1.0, t2c + (size_t)c0 * o2 * v, v,
            g + (((size_t)a * v) * v + b) * o, v * o, 0.0, tmp, o);
    for (c = 0; c < nc; c++) for (k = 0; k < o; k++) for (i = 0; i < o; i++) {
        const double *t = tmp + ((c * o + k) * (size_t)o + i) * o;
        double *wc = w + (size_t)c * o3 + i * o2 + k;
        for (j = 0; j < o; j++) wc[j * o] += t[j];
    }
    /* R11: t_kj^ce (be|ai) -> tmp[(c,k,j)][i] */
    gemm_rm('N', 'N', nc * o * o, o, v, 1.0, t2c + (size_t)c0 * o2 * v, v,
            g + (((size_t)b * v) * v + a) * o, v * o, 0.0, tmp, o);
    for (c = 0; c < nc; c++) for (k = 0; k < o; k++) for (j = 0; j < o; j++) {
        const double *t = tmp + ((c * o + k) * (size_t)o + j) * o;
        double *wc = w + (size_t)c * o3 + j * o + k;
        for (i = 0; i < o; i++) wc[i * o2] += t[i];
    }
    /* R2: - t_im^ab (mj|ck) -> tmp[i][(c,j,k)] */
    gemm_rm('N', 'N', o, nc * o * o, o, 1.0, B->t2ab, o, hT + (size_t)c0 * o2, (int)(v * o2),
            0.0, tmp, nc * o * o);
    for (i = 0; i < o; i++) for (c = 0; c < nc; c++) {
        const double *t = tmp + (i * (size_t)nc + c) * o2;
        double *wc = w + (size_t)c * o3 + i * o2;
        for (m = 0; m < (int)o2; m++) wc[m] -= t[m];
    }
    /* R6: - t_jm^ba (mi|ck) -> tmp[j][(c,i,k)] */
    gemm_rm('N', 'N', o, nc * o * o, o, 1.0, B->t2ba, o, hT + (size_t)c0 * o2, (int)(v * o2),
            0.0, tmp, nc * o * o);
    for (j = 0; j < o; j++) for (c = 0; c < nc; c++) for (i = 0; i < o; i++) {
        const double *t = tmp + ((j * (size_t)nc + c) * o + i) * o;
        double *wc = w + (size_t)c * o3 + i * o2 + j * o;
        for (k = 0; k < o; k++) wc[k] -= t[k];
    }
    /* R4: - t_im^ac (mk|bj): per i, tmp[c][(k,j)] */
    for (i = 0; i < o; i++) {
        gemm_rm('T', 'N', nc, o * o, o, 1.0, t2 + ((size_t)i * o * v + a) * v + c0, (int)v2,
                hB + (size_t)b * o3, (int)o2, 0.0, tmp, o * o);
        for (c = 0; c < nc; c++) for (k = 0; k < o; k++) {
            const double *t = tmp + (c * (size_t)o + k) * o;
            double *wc = w + (size_t)c * o3 + i * o2 + k;
            for (j = 0; j < o; j++) wc[j * o] -= t[j];
        }
    }
    /* R8: - t_jm^bc (mk|ai): per j, tmp[c][(k,i)] */
    for (j = 0; j < o; j++) {
        gemm_rm('T', 'N', nc, o * o, o, 1.0, t2 + ((size_t)j * o * v + b) * v + c0, (int)v2,
                hB + (size_t)a * o3, (int)o2, 0.0, tmp, o * o);
        for (c = 0; c < nc; c++) for (k = 0; k < o; k++) {
            const double *t = tmp + (c * (size_t)o + k) * o;
            double *wc = w + (size_t)c * o3 + j * o + k;
            for (i = 0; i < o; i++) wc[i * o2] -= t[i];
        }
    }
    /* R10: - t_km^ca (mi|bj) -> tmp[(c,k)][(i,j)] */
    gemm_rm('N', 'N', nc * o, o * o, o, 1.0, t2ackm + ((size_t)a * v + c0) * o2, o,
            hB + (size_t)b * o3, (int)o2, 0.0, tmp, o * o);
    for (c = 0; c < nc; c++) for (k = 0; k < o; k++) for (i = 0; i < o; i++) {
        const double *t = tmp + ((c * (size_t)o + k) * o + i) * o;
        double *wc = w + (size_t)c * o3 + i * o2 + k;
        for (j = 0; j < o; j++) wc[j * o] -= t[j];
    }
    /* R12: - t_km^cb (mj|ai) -> tmp[(c,k)][(j,i)] */
    gemm_rm('N', 'N', nc * o, o * o, o, 1.0, t2ackm + ((size_t)b * v + c0) * o2, o,
            hB + (size_t)a * o3, (int)o2, 0.0, tmp, o * o);
    for (c = 0; c < nc; c++) for (k = 0; k < o; k++) for (j = 0; j < o; j++) {
        const double *t = tmp + ((c * (size_t)o + k) * o + j) * o;
        double *wc = w + (size_t)c * o3 + j * o + k;
        for (i = 0; i < o; i++) wc[i * o2] -= t[i];
    }
}

/* ------------------------------------------------------ Z, D and r3 -> Y */
static void build_z(const Buffers *B, int a, int b, int c0, int c1,
                    const double *t1, const double *ovov)
{
    const int o = B->o, v = B->v, nc = c1 - c0;
    const size_t o2 = (size_t)o * o, o3 = o2 * o;
    int i, j, k, c;
    for (c = 0; c < nc; c++) {
        const int cc = c0 + c;
        double *zc = B->z + (size_t)c * o3;
        for (i = 0; i < o; i++) for (j = 0; j < o; j++) {
            const double tia = t1[i * v + a], tjb = t1[j * v + b];
            const double *ov_jb = ovov + ((size_t)j * v + b) * o * v + cc;   /* (jb|kc), k stride v */
            const double *ov_ia = ovov + ((size_t)i * v + a) * o * v + cc;   /* (ia|kc) */
            const double ia_jb = ovov[(((size_t)i * v + a) * o + j) * v + b];
            double *zij = zc + i * o2 + j * o;
            for (k = 0; k < o; k++)
                zij[k] = tia * ov_jb[(size_t)k * v] + tjb * ov_ia[(size_t)k * v] + t1[k * v + cc] * ia_jb;
        }
    }
}

static void r3_scaled(const double *w, double *y, int o, double fac,
                      const double *eo, double eabc)
{
    /* y_ijk = fac * (4 w_ijk + w_jki + w_kij - 2 w_kji - 2 w_ikj - 2 w_jik) / d_ijk */
    const size_t o2 = (size_t)o * o;
    int i, j, k;
    for (i = 0; i < o; i++) for (j = 0; j < o; j++) for (k = 0; k < o; k++) {
        const double d = eo[i] + eo[j] + eo[k] - eabc;
        const double s = 4.0 * w[i * o2 + j * o + k] + w[j * o2 + k * o + i] + w[k * o2 + i * o + j]
                       - 2.0 * (w[k * o2 + j * o + i] + w[i * o2 + k * o + j] + w[j * o2 + i * o + k]);
        y[i * o2 + j * o + k] = fac * s / d;
    }
}

/* --------------------------------------------- adjoint accumulation P, Q */
static void add_p(double *P, int row, int e_stride_unused, const Buffers *B, int col, const double *acc)
{
    /* P[row][e][col][k] += acc[e][k] */
    const int o = B->o, v = B->v;
    int e, k;
    LOCK(plock, row);
    for (e = 0; e < v; e++) {
        double *p = P + (((size_t)row * v + e) * v + col) * o;
        const double *s = acc + (size_t)e * o;
        for (k = 0; k < o; k++) p[k] += s[k];
    }
    UNLOCK(plock, row);
    (void)e_stride_unused;
}

static void accumulate(const Buffers *B, const double *y, int a, int b, int c0, int c1,
                       const double *t2, double *P, double *Q)
{
    const int o = B->o, v = B->v, nc = c1 - c0;
    const size_t o2 = (size_t)o * o, o3 = o2 * o, v2 = (size_t)v * v;
    double *yt1 = B->yt1, *yt2 = B->yt2, *yt3 = B->yt3, *pacc = B->pacc, *qacc = B->qacc;
    int i, j, k, m, c;

    for (c = 0; c < nc; c++) {
        const int cc = c0 + c;
        const double *yc = y + (size_t)c * o3;                       /* [i][j][k] */
        for (i = 0; i < o; i++) for (j = 0; j < o; j++) for (k = 0; k < o; k++) {
            const double val = yc[i * o2 + j * o + k];
            yt1[i * o2 + k * o + j] = val;                           /* [i][k][j] */
            yt2[j * o2 + i * o + k] = val;                           /* [j][i][k] */
            yt3[k * o2 + i * o + j] = val;                           /* [k][i][j] */
        }
        /* per-c amplitude slices */
        for (i = 0; i < o; i++) for (m = 0; m < o; m++) {
            B->tac[i * o + m] = t2[(((size_t)i * o + m) * v + a) * v + cc];   /* t_im^ac */
            B->tbc[i * o + m] = t2[(((size_t)i * o + m) * v + b) * v + cc];   /* t_jm^bc (j=i) */
            B->tca[i * o + m] = t2[(((size_t)i * o + m) * v + cc) * v + a];   /* t_km^ca (k=i) */
            B->tcb[i * o + m] = t2[(((size_t)i * o + m) * v + cc) * v + b];   /* t_km^cb */
        }
        /* --- P terms: acc[e][k] = sum_(pair) t[(pair)][e] * Y[(pair)][k] */
        /* R1: P[b][e][c][k] += sum_ij t_ij^ae Y_ijk */
        gemm_rm('T', 'N', v, o, o * o, 1.0, t2 + (size_t)a * v, (int)v2, yc, o, 0.0, pacc, o);
        add_p(P, b, 0, B, cc, pacc);
        /* R3: P[c][e][b][j] += sum_ik t_ik^ae Y_ijk   (yt1 = [(i,k)][j]) */
        gemm_rm('T', 'N', v, o, o * o, 1.0, t2 + (size_t)a * v, (int)v2, yt1, o, 0.0, pacc, o);
        add_p(P, cc, 0, B, b, pacc);
        /* R5: P[a][e][c][k] += sum_ji t_ji^be Y_ijk   (yt2 = [(j,i)][k]) */
        gemm_rm('T', 'N', v, o, o * o, 1.0, t2 + (size_t)b * v, (int)v2, yt2, o, 0.0, pacc, o);
        add_p(P, a, 0, B, cc, pacc);
        /* R7: P[c][e][a][i] += sum_jk t_jk^be Y_ijk   (yc as [i][(j,k)], transposed) */
        gemm_rm('T', 'T', v, o, o * o, 1.0, t2 + (size_t)b * v, (int)v2, yc, o * o, 0.0, pacc, o);
        add_p(P, cc, 0, B, a, pacc);
        /* R9: P[a][e][b][j] += sum_ki t_ki^ce Y_ijk   (yt3 = [(k,i)][j]) */
        gemm_rm('T', 'N', v, o, o * o, 1.0, t2 + (size_t)cc * v, (int)v2, yt3, o, 0.0, pacc, o);
        add_p(P, a, 0, B, b, pacc);
        /* R11: P[b][e][a][i] += sum_kj t_kj^ce Y_ijk  (yt1 as [i][(k,j)], transposed) */
        gemm_rm('T', 'T', v, o, o * o, 1.0, t2 + (size_t)cc * v, (int)v2, yt1, o * o, 0.0, pacc, o);
        add_p(P, b, 0, B, a, pacc);

        /* --- Q terms (sign of the h-terms is applied by the caller):
           qacc[m][(x,y)] = sum_i T[i][m] Y[i][(x,y)] */
        /* R2: Q[m][j][c][k] += sum_i t_im^ab Y_ijk */
        gemm_rm('T', 'N', o, o * o, o, 1.0, B->t2ab, o, yc, o * o, 0.0, qacc, o * o);
        LOCK(qlock, cc);
        for (m = 0; m < o; m++) for (j = 0; j < o; j++) {
            double *q = Q + (((size_t)m * o + j) * v + cc) * o;
            const double *s = qacc + (m * o2 + j * o);
            for (k = 0; k < o; k++) q[k] += s[k];
        }
        UNLOCK(qlock, cc);
        /* R4: Q[m][k][b][j] += sum_i t_im^ac Y_ijk   (yt1 = [i][(k,j)]) */
        gemm_rm('T', 'N', o, o * o, o, 1.0, B->tac, o, yt1, o * o, 0.0, qacc, o * o);
        LOCK(qlock, b);
        for (m = 0; m < o; m++) for (k = 0; k < o; k++) {
            double *q = Q + (((size_t)m * o + k) * v + b) * o;
            const double *s = qacc + (m * o2 + k * o);
            for (j = 0; j < o; j++) q[j] += s[j];
        }
        UNLOCK(qlock, b);
        /* R6: Q[m][i][c][k] += sum_j t_jm^ba Y_ijk   (yt2 = [j][(i,k)]) */
        gemm_rm('T', 'N', o, o * o, o, 1.0, B->t2ba, o, yt2, o * o, 0.0, qacc, o * o);
        LOCK(qlock, cc);
        for (m = 0; m < o; m++) for (i = 0; i < o; i++) {
            double *q = Q + (((size_t)m * o + i) * v + cc) * o;
            const double *s = qacc + (m * o2 + i * o);
            for (k = 0; k < o; k++) q[k] += s[k];
        }
        UNLOCK(qlock, cc);
        /* R8: Q[m][k][a][i] += sum_j t_jm^bc Y_ijk   (yt2 = [j][(i,k)]) */
        gemm_rm('T', 'N', o, o * o, o, 1.0, B->tbc, o, yt2, o * o, 0.0, qacc, o * o);
        LOCK(qlock, a);
        for (m = 0; m < o; m++) for (i = 0; i < o; i++) {
            const double *s = qacc + (m * o2 + i * o);
            for (k = 0; k < o; k++) Q[(((size_t)m * o + k) * v + a) * o + i] += s[k];
        }
        UNLOCK(qlock, a);
        /* R10: Q[m][i][b][j] += sum_k t_km^ca Y_ijk   (yc as [(i,j)][k], transposed) */
        gemm_rm('T', 'T', o, o * o, o, 1.0, B->tca, o, yc, o, 0.0, qacc, o * o);
        LOCK(qlock, b);
        for (m = 0; m < o; m++) for (i = 0; i < o; i++) {
            double *q = Q + (((size_t)m * o + i) * v + b) * o;
            const double *s = qacc + (m * o2 + i * o);
            for (j = 0; j < o; j++) q[j] += s[j];
        }
        UNLOCK(qlock, b);
        /* R12: Q[m][j][a][i] += sum_k t_km^cb Y_ijk */
        gemm_rm('T', 'T', o, o * o, o, 1.0, B->tcb, o, yc, o, 0.0, qacc, o * o);
        LOCK(qlock, a);
        for (m = 0; m < o; m++) for (i = 0; i < o; i++) {
            const double *s = qacc + (m * o2 + i * o);
            for (j = 0; j < o; j++) Q[(((size_t)m * o + j) * v + a) * o + i] += s[j];
        }
        UNLOCK(qlock, a);
    }
}

/* ------------------------------------------------------------- driver */
int ccsd_t_eda_partition(int nocc, int nvir, int with_t4, int nc_max,
                         const double *e_occ, const double *e_vir,
                         const double *t1, const double *t2, const double *t2c,
                         const double *t2ackm, const double *g, const double *hT,
                         const double *hB, const double *ovov,
                         double *P, double *Q, double *P4, double *Q4, void *dgemm_ptr)
{
    const int o = nocc, v = nvir;
    const size_t o2 = (size_t)o * o, o3 = o2 * o;
    int a, b, c0, ntask = 0, it;
    int *tasks;

    g_dgemm = (dgemm_fn)dgemm_ptr;
    if (nc_max < 1) nc_max = 1;
    for (a = 0; a < v; a++) for (b = 0; b <= a; b++)
        for (c0 = 0; c0 <= b; c0 += nc_max) ntask++;
    tasks = (int *)malloc(sizeof(int) * 3 * (size_t)ntask);
    if (tasks == NULL) return 1;
    it = 0;
    for (a = 0; a < v; a++) for (b = 0; b <= a; b++)
        for (c0 = 0; c0 <= b; c0 += nc_max) {
            tasks[3 * it] = a; tasks[3 * it + 1] = b; tasks[3 * it + 2] = c0; it++;
        }
#ifdef _OPENMP
    plock = (omp_lock_t *)malloc(sizeof(omp_lock_t) * v);
    qlock = (omp_lock_t *)malloc(sizeof(omp_lock_t) * v);
    for (a = 0; a < v; a++) { omp_init_lock(&plock[a]); omp_init_lock(&qlock[a]); }
#endif
    int fail = 0;
#pragma omp parallel
    {
        Buffers B;
        B.o = o; B.v = v; B.nc = nc_max;
        const size_t blk = (size_t)nc_max * o3;
        B.w = (double *)malloc(sizeof(double) * blk);
        B.z = (double *)malloc(sizeof(double) * blk);
        B.y = (double *)malloc(sizeof(double) * blk);
        B.y4 = with_t4 ? (double *)malloc(sizeof(double) * blk) : NULL;
        B.tmp = (double *)malloc(sizeof(double) * blk);
        B.yt1 = (double *)malloc(sizeof(double) * o3);
        B.yt2 = (double *)malloc(sizeof(double) * o3);
        B.yt3 = (double *)malloc(sizeof(double) * o3);
        B.pacc = (double *)malloc(sizeof(double) * (size_t)v * o);
        B.qacc = (double *)malloc(sizeof(double) * o3);
        B.t2ab = (double *)malloc(sizeof(double) * o2 * 6);
        B.t2ba = B.t2ab + o2; B.tac = B.t2ba + o2; B.tbc = B.tac + o2;
        B.tca = B.tbc + o2; B.tcb = B.tca + o2;
        if (B.w == NULL || B.z == NULL || B.y == NULL || B.tmp == NULL || B.yt1 == NULL ||
            B.yt2 == NULL || B.yt3 == NULL || B.pacc == NULL || B.qacc == NULL ||
            B.t2ab == NULL || (with_t4 && B.y4 == NULL)) {
#pragma omp atomic write
            fail = 1;
        }
#pragma omp barrier
        if (!fail) {
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < ntask; it++) {
            const int ta = tasks[3 * it], tb = tasks[3 * it + 1], tc0 = tasks[3 * it + 2];
            const int tc1 = (tc0 + nc_max <= tb + 1) ? tc0 + nc_max : tb + 1;
            const int nc = tc1 - tc0;
            int i, m, c;
            for (i = 0; i < o; i++) for (m = 0; m < o; m++) {
                B.t2ab[i * o + m] = t2[(((size_t)i * o + m) * v + ta) * v + tb];
                B.t2ba[i * o + m] = t2[(((size_t)i * o + m) * v + tb) * v + ta];
            }
            build_w(&B, ta, tb, tc0, tc1, t2, t2c, t2ackm, g, hT, hB);
            build_z(&B, ta, tb, tc0, tc1, t1, ovov);
            for (c = 0; c < nc; c++) {
                const int cc = tc0 + c;
                double fac = 1.0;
                if (ta == tb && tb == cc) fac = 1.0 / 6.0;
                else if (ta == tb || tb == cc) fac = 0.5;
                const double eabc = e_vir[ta] + e_vir[tb] + e_vir[cc];
                double *wc = B.w + (size_t)c * o3, *zc = B.z + (size_t)c * o3;
                size_t n;
                if (with_t4)
                    r3_scaled(wc, B.y4 + (size_t)c * o3, o, 2.0 * fac, e_occ, eabc);
                for (n = 0; n < o3; n++) wc[n] += zc[n];
                r3_scaled(wc, B.y + (size_t)c * o3, o, 2.0 * fac, e_occ, eabc);
            }
            accumulate(&B, B.y, ta, tb, tc0, tc1, t2, P, Q);
            if (with_t4) accumulate(&B, B.y4, ta, tb, tc0, tc1, t2, P4, Q4);
        }
        }
        free(B.w); free(B.z); free(B.y); free(B.y4); free(B.tmp);
        free(B.yt1); free(B.yt2); free(B.yt3); free(B.pacc); free(B.qacc); free(B.t2ab);
    }
#ifdef _OPENMP
    for (a = 0; a < v; a++) { omp_destroy_lock(&plock[a]); omp_destroy_lock(&qlock[a]); }
    free(plock); free(qlock); plock = qlock = NULL;
#endif
    free(tasks);
    return fail;
}

int ccsd_t_eda_num_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
