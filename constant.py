# 物理定数・材料定数

import numpy as np

# physical constants
PV = 8.854187817620e-12                     # electric constant (F/m)
EC = 1.6021766208e-19                       # elementary charge (C)
KB = 1.38064852e-23                         # Boltzmann constant (J/K)


# material parameters and conditions
TL = 300.0                                  # lattice temperature (K)
VT = KB * TL / EC                           # thermal voltage (V)
EG = 3.0                                    # Band gap [eV]
NC = 2.8e25                                 # /m^3
NV = 2.8e25                                 # /m^3
NI = (NC*NV)**(1/2)*np.exp(-EG / (2*VT))    # intrinsic density (/m3)
DC = 10.0 * PV                              # dielectric constant (F/m)
MUP = 1e-4                                  # hole mobility (m2/Vs)
MUN = 1e-4                                  # electron mobility (m2/Vs)
