"""Energy density analysis (EDA) on top of PySCF.

EDA partitions the total energy of a molecule into atomic energies
(H. Nakai, Chem. Phys. Lett. 363, 73 (2002)).

Currently implemented
---------------------
* ``pyscf_eda.rhf``  : closed-shell (restricted) Hartree-Fock; conventional, LSO- and NAO-EDA
* ``pyscf_eda.orth`` : orthogonal one-centre bases (NAO, Loewdin) used by NAO-/LSO-EDA
* ``pyscf_eda.corr`` : atomic partition of correlation energies (shared machinery)
* ``pyscf_eda.mp2``  : closed-shell MP2 (HF part + partitioned correlation energy)
* ``pyscf_eda.ccsd`` : closed-shell CCSD (HF part + partitioned correlation energy)
* ``pyscf_eda.ccsd_t``: closed-shell CCSD(T) (adds the partitioned (T) correction)
"""

__version__ = '0.1.0'

from pyscf_eda import rhf  # noqa: F401
from pyscf_eda import orth  # noqa: F401
from pyscf_eda import corr  # noqa: F401
from pyscf_eda import mp2  # noqa: F401
from pyscf_eda import ccsd  # noqa: F401
from pyscf_eda import ccsd_t  # noqa: F401
from pyscf_eda.rhf import EDA, EDAResult, kernel, atom_energies, nao_eda, lso_eda  # noqa: F401
