"""Energy density analysis (EDA) on top of PySCF.

EDA partitions the total energy of a molecule into atomic energies
(H. Nakai, Chem. Phys. Lett. 363, 73 (2002)).

Currently implemented
---------------------
* ``pyscf_eda.rhf`` : closed-shell (restricted) Hartree-Fock
"""

__version__ = '0.1.0'

from pyscf_eda import rhf  # noqa: F401
from pyscf_eda.rhf import EDA, EDAResult, kernel, atom_energies  # noqa: F401
