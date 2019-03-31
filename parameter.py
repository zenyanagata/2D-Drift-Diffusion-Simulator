# シミュレーションの条件・規格化用の定数

import numpy as np
import constant as C

# Conditions
L = 16.9e-6       # device length (m)
N = 169           # number of mesh points
Nd_ = 5e25        # doping density in n-region (/m3)
Nd0_ = 5e26
Na_ = 5e24        # doping density in p-region (/m3)
Vstp = 0.01       # applied bias step (V)
Vmax = 5          # number of bias points
TOLERANCE = 7e-7  # potential tolerance (V)
h = L / N         # mesh width (m)


# constants for normalization
ETA = C.EC / h / C.DC / C.VT       # for Poisson

UJP = C.EC * C.MUP * C.VT / h**4   # unit of hole current
UJN = C.EC * C.MUN * C.VT / h**4   # unit of electron current

PFP = h**5 / (C.MUP * C.VT)        # pre-factor for Gamma_p
PFN = h**5 / (C.MUN * C.VT)        # pre-factor for Gamma_n

NNI = C.NI * h**3                  # normalized intrinsic density

J0_ = C.EC * Nd0_ * C.MUN * C.VT / L

