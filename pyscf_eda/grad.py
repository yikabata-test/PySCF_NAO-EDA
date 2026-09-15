"""Analytic nuclear gradients of the levels used in the CBS composite models.

The n-scheme fitting models (``pyscf_eda.cbs``) are linear in the
correlation energies, so the gradient of the CBS energy is the same linear
combination of the analytic gradients of the individual levels:

    dE_CBS/dR = sum_X c^HF_X dE_HF[X]/dR + sum c_{m,X} dE_corr(m, X)/dR,
    dE_corr(m, X)/dR = dE_tot(m, X)/dR - dE_HF[X]/dR.

PySCF provides analytic gradients for RHF, RMP2, CCSD and CCSD(T)
(``pyscf.grad``), including frozen-core orbitals.  Note that the CCSD(T)
gradient needs the CCSD(T) lambda equations (``pyscf.cc.ccsd_t_lambda``);
calling ``pyscf.grad.ccsd_t.Gradients(mycc).kernel()`` without lambda
amplitudes silently uses the CCSD lambda amplitudes and gives a wrong
gradient, which ``ccsd_t_gradient`` below avoids.
"""

import numpy
from pyscf.lib import logger


def hf_gradient(mf, verbose=0):
    """Analytic RHF/RKS nuclear gradient (natm, 3), hartree/bohr."""
    g = mf.nuc_grad_method()
    g.verbose = verbose
    return g.kernel()


def mp2_gradient(pt, verbose=0):
    """Analytic RMP2 total-energy gradient (frozen core supported)."""
    g = pt.nuc_grad_method()
    g.verbose = verbose
    return g.kernel()


def ccsd_gradient(mycc, verbose=0):
    """Analytic CCSD total-energy gradient (solves the CCSD lambda equations)."""
    g = mycc.nuc_grad_method()
    g.verbose = verbose
    return g.kernel()


def ccsd_t_gradient(mycc, eris=None, lambda_tol=1e-9, verbose=0):
    """Analytic CCSD(T) total-energy gradient with the CCSD(T) lambda equations.

    Returns
    -------
    ndarray (natm, 3)
    """
    from pyscf.cc import ccsd_t_lambda
    from pyscf.grad import ccsd_t as ccsd_t_grad
    if eris is None:
        eris = mycc.ao2mo()
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2, tol=lambda_tol,
                                        verbose=verbose)
    if not conv:
        logger.warn(mycc, 'CCSD(T) lambda equations did not converge')
    g = ccsd_t_grad.Gradients(mycc)
    g.verbose = verbose
    return g.kernel(mycc.t1, mycc.t2, l1, l2, eris=eris)


def hf_cbs_gradient(method, energies, grads, alpha=None):
    """Gradient of the HF/CBS estimate given the gradients {X: dE/dR}.

    Every scheme is linear in the energies, so the gradient is the same
    linear combination of the gradients (``energies`` only selects the
    cardinal numbers).
    """
    from pyscf_eda import cbs
    if alpha is None:
        alpha = cbs.HALKIER_ALPHA
    coeff = cbs.hf_cbs_coefficients(method, list(energies), alpha=alpha)
    return sum(c * grads[x] for x, c in coeff.items())


def format_gradient(mol, grad, title='Gradient (hartree/bohr)'):
    lines = [title, f"{'atom':<8}{'x':>18}{'y':>18}{'z':>18}"]
    for ia in range(mol.natm):
        lines.append(f"{mol.atom_symbol(ia) + str(ia):<8}" + ''.join(f"{v:>18.10f}" for v in grad[ia]))
    return '\n'.join(lines)
