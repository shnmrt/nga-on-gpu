from __future__ import absolute_import
from __future__ import print_function
# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import pyopencl as cl
import pyopencl.array


class CB14():
    """ CY14 Eq. 1
        Median Ground Motion Model
              [ ln PGA                                             PSA < PGA and T < 0.25s
        LnY = |
              [ Fmag + Fdis + Fflt + Fhng + Fsite + Fsed + Fhyp + Fdip + Fatn;   otherwise


              Fmag : earthquake magnitude
              Fdis : geometric attenuation
              Fflt : style of faulting
              Fhng : hanging wall geometry
              Fsite: shallow site response
              Fsed : basin response
              Fhyp : hypocentral depth
              Fdip : fault dip
              Fatn : anelastic attenuation
    """
    def __init__(self):
        self.coefCB = np.array([[ 1.0000e-02, -4.3650e+00,  9.7700e-01,  5.3300e-01, -1.4850e+00,
        -4.9900e-01, -2.7730e+00,  2.4800e-01,  6.7530e+00,  0.0000e+00,
        -2.1400e-01,  7.2000e-01,  1.0940e+00,  2.1910e+00,  1.4160e+00,
        -7.0000e-03, -2.0700e-01,  3.9000e-01,  9.8100e-02,  3.3400e-02,
         7.5500e-03, -5.5000e-03,  0.0000e+00, -3.5000e-03,  3.6000e-03,
         1.6800e-01,  2.4200e-01,  1.4710e+00, -7.1400e-01,  1.0000e+00,
        -3.3600e-01, -2.7000e-01,  8.6500e+02, -1.1860e+00,  1.8390e+00,
         1.8800e+00,  1.1800e+00,  7.3400e-01,  4.9200e-01,  4.0400e-01,
         3.2500e-01,  3.0000e-01,  1.6600e-01,  1.0000e+00],
       [ 2.0000e-02, -4.3480e+00,  9.7600e-01,  5.4900e-01, -1.4880e+00,
        -5.0100e-01, -2.7720e+00,  2.4700e-01,  6.5020e+00,  0.0000e+00,
        -2.0800e-01,  7.3000e-01,  1.1490e+00,  2.1890e+00,  1.4530e+00,
        -1.6700e-02, -1.9900e-01,  3.8700e-01,  1.0090e-01,  3.2700e-02,
         7.5900e-03, -5.5000e-03,  0.0000e+00, -3.5000e-03,  3.6000e-03,
         1.6600e-01,  2.4400e-01,  1.4670e+00, -7.1100e-01,  1.0000e+00,
        -3.3900e-01, -2.6300e-01,  8.6500e+02, -1.2190e+00,  1.8400e+00,
         1.8800e+00,  1.1800e+00,  7.3800e-01,  4.9600e-01,  4.1700e-01,
         3.2600e-01,  3.0000e-01,  1.6600e-01,  9.9800e-01],
       [ 3.0000e-02, -4.0240e+00,  9.3100e-01,  6.2800e-01, -1.4940e+00,
        -5.1700e-01, -2.7820e+00,  2.4600e-01,  6.2910e+00,  0.0000e+00,
        -2.1300e-01,  7.5900e-01,  1.2900e+00,  2.1640e+00,  1.4760e+00,
        -4.2200e-02, -2.0200e-01,  3.7800e-01,  1.0950e-01,  3.3100e-02,
         7.9000e-03, -5.7000e-03,  0.0000e+00, -3.4000e-03,  3.7000e-03,
         1.6700e-01,  2.4600e-01,  1.4670e+00, -7.1300e-01,  1.0000e+00,
        -3.3800e-01, -2.5900e-01,  9.0800e+02, -1.2730e+00,  1.8410e+00,
         1.8800e+00,  1.1800e+00,  7.4700e-01,  5.0300e-01,  4.4600e-01,
         3.4400e-01,  3.0000e-01,  1.6500e-01,  9.8600e-01],
       [ 5.0000e-02, -3.4790e+00,  8.8700e-01,  6.7400e-01, -1.3880e+00,
        -6.1500e-01, -2.7910e+00,  2.4000e-01,  6.3170e+00,  0.0000e+00,
        -2.4400e-01,  8.2600e-01,  1.4490e+00,  2.1380e+00,  1.5490e+00,
        -6.6300e-02, -3.3900e-01,  2.9500e-01,  1.2260e-01,  2.7000e-02,
         8.0300e-03, -6.3000e-03,  0.0000e+00, -3.7000e-03,  4.0000e-03,
         1.7300e-01,  2.5100e-01,  1.4490e+00, -7.0100e-01,  1.0000e+00,
        -3.3800e-01, -2.6300e-01,  1.0540e+03, -1.3460e+00,  1.8430e+00,
         1.8800e+00,  1.1800e+00,  7.7700e-01,  5.2000e-01,  5.0800e-01,
         3.7700e-01,  3.0000e-01,  1.6200e-01,  9.3800e-01],
       [ 8.0000e-02, -3.2930e+00,  9.0200e-01,  7.2600e-01, -1.4690e+00,
        -5.9600e-01, -2.7450e+00,  2.2700e-01,  6.8610e+00,  0.0000e+00,
        -2.6600e-01,  8.1500e-01,  1.5350e+00,  2.4460e+00,  1.7720e+00,
        -7.9400e-02, -4.0400e-01,  3.2200e-01,  1.1650e-01,  2.8800e-02,
         8.1100e-03, -7.0000e-03,  0.0000e+00, -3.7000e-03,  3.9000e-03,
         1.9800e-01,  2.6000e-01,  1.4350e+00, -6.9500e-01,  1.0000e+00,
        -3.4700e-01, -2.1900e-01,  1.0860e+03, -1.4710e+00,  1.8450e+00,
         1.8800e+00,  1.1800e+00,  7.8200e-01,  5.3500e-01,  5.0400e-01,
         4.1800e-01,  3.0000e-01,  1.5800e-01,  8.8700e-01],
       [ 1.0000e-01, -3.6660e+00,  9.9300e-01,  6.9800e-01, -1.5720e+00,
        -5.3600e-01, -2.6330e+00,  2.1000e-01,  7.2940e+00,  0.0000e+00,
        -2.2900e-01,  8.3100e-01,  1.6150e+00,  2.9690e+00,  1.9160e+00,
        -2.9400e-02, -4.1600e-01,  3.8400e-01,  9.9800e-02,  3.2500e-02,
         7.4400e-03, -7.3000e-03,  0.0000e+00, -3.4000e-03,  4.2000e-03,
         1.7400e-01,  2.5900e-01,  1.4490e+00, -7.0800e-01,  1.0000e+00,
        -3.9100e-01, -2.0100e-01,  1.0320e+03, -1.6240e+00,  1.8470e+00,
         1.8800e+00,  1.1800e+00,  7.6900e-01,  5.4300e-01,  4.4500e-01,
         4.2600e-01,  3.0000e-01,  1.7000e-01,  8.7000e-01],
       [ 1.5000e-01, -4.8660e+00,  1.2670e+00,  5.1000e-01, -1.6690e+00,
        -4.9000e-01, -2.4580e+00,  1.8300e-01,  8.0310e+00,  0.0000e+00,
        -2.1100e-01,  7.4900e-01,  1.8770e+00,  3.5440e+00,  2.1610e+00,
         6.4200e-02, -4.0700e-01,  4.1700e-01,  7.6000e-02,  3.8800e-02,
         7.1600e-03, -6.9000e-03,  0.0000e+00, -3.0000e-03,  4.2000e-03,
         1.9800e-01,  2.5400e-01,  1.4610e+00, -7.1500e-01,  1.0000e+00,
        -4.4900e-01, -9.9000e-02,  8.7800e+02, -1.9310e+00,  1.8520e+00,
         1.8800e+00,  1.1800e+00,  7.6900e-01,  5.4300e-01,  3.8200e-01,
         3.8700e-01,  3.0000e-01,  1.8000e-01,  8.7600e-01],
       [ 2.0000e-01, -5.4110e+00,  1.3660e+00,  4.4700e-01, -1.7500e+00,
        -4.5100e-01, -2.4210e+00,  1.8200e-01,  8.3850e+00,  0.0000e+00,
        -1.6300e-01,  7.6400e-01,  2.0690e+00,  3.7070e+00,  2.4650e+00,
         9.6800e-02, -3.1100e-01,  4.0400e-01,  5.7100e-02,  4.3700e-02,
         6.8800e-03, -6.0000e-03,  0.0000e+00, -3.1000e-03,  4.1000e-03,
         2.0400e-01,  2.3700e-01,  1.4840e+00, -7.2100e-01,  1.0000e+00,
        -3.9300e-01, -1.9800e-01,  7.4800e+02, -2.1880e+00,  1.8560e+00,
         1.8800e+00,  1.1800e+00,  7.6100e-01,  5.5200e-01,  3.3900e-01,
         3.3800e-01,  3.0000e-01,  1.8600e-01,  8.7000e-01],
       [ 2.5000e-01, -5.9620e+00,  1.4580e+00,  2.7400e-01, -1.7110e+00,
        -4.0400e-01, -2.3920e+00,  1.8900e-01,  7.5340e+00,  0.0000e+00,
        -1.5000e-01,  7.1600e-01,  2.2050e+00,  3.3430e+00,  2.7660e+00,
         1.4410e-01, -1.7200e-01,  4.6600e-01,  4.3700e-02,  4.6300e-02,
         5.5600e-03, -5.5000e-03,  0.0000e+00, -3.3000e-03,  3.6000e-03,
         1.8500e-01,  2.0600e-01,  1.5810e+00, -7.8700e-01,  1.0000e+00,
        -3.3900e-01, -2.1000e-01,  6.5400e+02, -2.3810e+00,  1.8610e+00,
         1.8800e+00,  1.1800e+00,  7.4400e-01,  5.4500e-01,  3.4000e-01,
         3.1600e-01,  3.0000e-01,  1.9100e-01,  8.5000e-01],
       [ 3.0000e-01, -6.4030e+00,  1.5280e+00,  1.9300e-01, -1.7700e+00,
        -3.2100e-01, -2.3760e+00,  1.9500e-01,  6.9900e+00,  0.0000e+00,
        -1.3100e-01,  7.3700e-01,  2.3060e+00,  3.3340e+00,  3.0110e+00,
         1.5970e-01, -8.4000e-02,  5.2800e-01,  3.2300e-02,  5.0800e-02,
         4.5800e-03, -4.9000e-03,  0.0000e+00, -3.5000e-03,  3.1000e-03,
         1.6400e-01,  2.1000e-01,  1.5860e+00, -7.9500e-01,  1.0000e+00,
        -4.4700e-01, -1.2100e-01,  5.8700e+02, -2.5180e+00,  1.8650e+00,
         1.8800e+00,  1.1800e+00,  7.2700e-01,  5.6800e-01,  3.4000e-01,
         3.0000e-01,  3.0000e-01,  1.9800e-01,  8.1900e-01],
       [ 4.0000e-01, -7.5660e+00,  1.7390e+00, -2.0000e-02, -1.5940e+00,
        -4.2600e-01, -2.3030e+00,  1.8500e-01,  7.0120e+00,  0.0000e+00,
        -1.5900e-01,  7.3800e-01,  2.3980e+00,  3.5440e+00,  3.2030e+00,
         1.4100e-01,  8.5000e-02,  5.4000e-01,  2.0900e-02,  4.3200e-02,
         4.0100e-03, -3.7000e-03,  0.0000e+00, -3.4000e-03,  2.8000e-03,
         1.6000e-01,  2.2600e-01,  1.5440e+00, -7.7000e-01,  1.0000e+00,
        -5.2500e-01, -8.6000e-02,  5.0300e+02, -2.6570e+00,  1.8740e+00,
         1.8800e+00,  1.1800e+00,  6.9000e-01,  5.9300e-01,  3.5600e-01,
         2.6400e-01,  3.0000e-01,  2.0600e-01,  7.4300e-01],
       [ 5.0000e-01, -8.3790e+00,  1.8720e+00, -1.2100e-01, -1.5770e+00,
        -4.4000e-01, -2.2960e+00,  1.8600e-01,  6.9020e+00,  0.0000e+00,
        -1.5300e-01,  7.1800e-01,  2.3550e+00,  3.0160e+00,  3.3330e+00,
         1.4740e-01,  2.3300e-01,  6.3800e-01,  9.2000e-03,  4.0500e-02,
         3.8800e-03, -2.7000e-03,  0.0000e+00, -3.4000e-03,  2.5000e-03,
         1.8400e-01,  2.1700e-01,  1.5540e+00, -7.7000e-01,  1.0000e+00,
        -4.0700e-01, -2.8100e-01,  4.5700e+02, -2.6690e+00,  1.8830e+00,
         1.8800e+00,  1.1800e+00,  6.6300e-01,  6.1100e-01,  3.7900e-01,
         2.6300e-01,  3.0000e-01,  2.0800e-01,  6.8400e-01],
       [ 7.5000e-01, -9.8410e+00,  2.0210e+00, -4.2000e-02, -1.7570e+00,
        -4.4300e-01, -2.2320e+00,  1.8600e-01,  5.5220e+00,  0.0000e+00,
        -9.0000e-02,  7.9500e-01,  1.9950e+00,  2.6160e+00,  3.0540e+00,
         1.7640e-01,  4.1100e-01,  7.7600e-01, -8.2000e-03,  4.2000e-02,
         4.2000e-03, -1.6000e-03,  0.0000e+00, -3.2000e-03,  1.6000e-03,
         2.1600e-01,  1.5400e-01,  1.6260e+00, -7.8000e-01,  1.0000e+00,
        -3.7100e-01, -2.8500e-01,  4.1000e+02, -2.4010e+00,  1.9060e+00,
         1.8800e+00,  1.1800e+00,  6.0600e-01,  6.3300e-01,  4.3000e-01,
         3.2600e-01,  3.0000e-01,  2.2100e-01,  5.6200e-01],
       [ 1.0000e+00, -1.1011e+01,  2.1800e+00, -6.9000e-02, -1.7070e+00,
        -5.2700e-01, -2.1580e+00,  1.6900e-01,  5.6500e+00,  0.0000e+00,
        -1.0500e-01,  5.5600e-01,  1.4470e+00,  2.4700e+00,  2.5620e+00,
         2.5930e-01,  4.7900e-01,  7.7100e-01, -1.3100e-02,  4.2600e-02,
         4.0900e-03, -6.0000e-04,  0.0000e+00, -3.0000e-03,  6.0000e-04,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02, -1.9550e+00,  1.9290e+00,
         1.8800e+00,  1.1800e+00,  5.7900e-01,  6.2800e-01,  4.7000e-01,
         3.5300e-01,  3.0000e-01,  2.2500e-01,  4.6700e-01],
       [ 1.5000e+00, -1.2469e+01,  2.2700e+00,  4.7000e-02, -1.6210e+00,
        -6.3000e-01, -2.0630e+00,  1.5800e-01,  5.7950e+00,  0.0000e+00,
        -5.8000e-02,  4.8000e-01,  3.3000e-01,  2.1080e+00,  1.4530e+00,
         2.8810e-01,  5.6600e-01,  7.4800e-01, -1.8700e-02,  3.8000e-02,
         4.2400e-03,  0.0000e+00,  0.0000e+00, -1.9000e-03,  0.0000e+00,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02, -1.0250e+00,  1.9740e+00,
         1.8800e+00,  1.1800e+00,  5.4100e-01,  6.0300e-01,  4.9700e-01,
         3.9900e-01,  3.0000e-01,  2.2200e-01,  3.6400e-01],
       [ 2.0000e+00, -1.2969e+01,  2.2710e+00,  1.4900e-01, -1.5120e+00,
        -7.6800e-01, -2.1040e+00,  1.5800e-01,  6.6320e+00,  0.0000e+00,
        -2.8000e-02,  4.0100e-01, -5.1400e-01,  1.3270e+00,  6.5700e-01,
         3.1120e-01,  5.6200e-01,  7.6300e-01, -2.5800e-02,  2.5200e-02,
         4.4800e-03,  0.0000e+00,  0.0000e+00, -5.0000e-04,  0.0000e+00,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02, -2.9900e-01,  2.0190e+00,
         1.8800e+00,  1.1800e+00,  5.2900e-01,  5.8800e-01,  4.9900e-01,
         4.0000e-01,  3.0000e-01,  2.2600e-01,  2.9800e-01],
       [ 3.0000e+00, -1.3306e+01,  2.1500e+00,  3.6800e-01, -1.3150e+00,
        -8.9000e-01, -2.0510e+00,  1.4800e-01,  6.7590e+00,  0.0000e+00,
         0.0000e+00,  2.0600e-01, -8.4800e-01,  6.0100e-01,  3.6700e-01,
         3.4780e-01,  5.3400e-01,  6.8600e-01, -3.1100e-02,  2.3600e-02,
         3.4500e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02,  0.0000e+00,  2.1100e+00,
         1.8800e+00,  1.1800e+00,  5.2700e-01,  5.7800e-01,  5.0000e-01,
         4.1700e-01,  3.0000e-01,  2.2900e-01,  2.3400e-01],
       [ 4.0000e+00, -1.4020e+01,  2.1320e+00,  7.2600e-01, -1.5060e+00,
        -8.8500e-01, -1.9860e+00,  1.3500e-01,  7.9780e+00,  0.0000e+00,
         0.0000e+00,  1.0500e-01, -7.9300e-01,  5.6800e-01,  3.0600e-01,
         3.7470e-01,  5.2200e-01,  6.9100e-01, -4.1300e-02,  1.0200e-02,
         6.0300e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02,  0.0000e+00,  2.2000e+00,
         1.8800e+00,  1.1800e+00,  5.2100e-01,  5.5900e-01,  5.4300e-01,
         3.9300e-01,  3.0000e-01,  2.3700e-01,  2.0200e-01],
       [ 5.0000e+00, -1.4558e+01,  2.1160e+00,  1.0270e+00, -1.7210e+00,
        -8.7800e-01, -2.0210e+00,  1.3500e-01,  8.5380e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -7.4800e-01,  3.5600e-01,  2.6800e-01,
         3.3820e-01,  4.7700e-01,  6.7000e-01, -2.8100e-02,  3.4000e-03,
         8.0500e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02,  0.0000e+00,  2.2910e+00,
         1.8800e+00,  1.1800e+00,  5.0200e-01,  5.5100e-01,  5.3400e-01,
         4.2100e-01,  3.0000e-01,  2.3700e-01,  1.8400e-01],
       [ 7.5000e+00, -1.5509e+01,  2.2230e+00,  1.6900e-01, -7.5600e-01,
        -1.0770e+00, -2.1790e+00,  1.6500e-01,  8.4680e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -6.6400e-01,  7.5000e-02,  3.7400e-01,
         3.7540e-01,  3.2100e-01,  7.5700e-01, -2.0500e-02,  5.0000e-03,
         2.8000e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02,  0.0000e+00,  2.5170e+00,
         1.8800e+00,  1.1800e+00,  4.5700e-01,  5.4600e-01,  5.2300e-01,
         4.3800e-01,  3.0000e-01,  2.7100e-01,  1.7600e-01],
       [ 1.0000e+01, -1.5975e+01,  2.1320e+00,  3.6700e-01, -8.0000e-01,
        -1.2820e+00, -2.2440e+00,  1.8000e-01,  6.5640e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -5.7600e-01, -2.7000e-02,  2.9700e-01,
         3.5060e-01,  1.7400e-01,  6.2100e-01,  9.0000e-04,  9.9000e-03,
         4.5800e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02,  0.0000e+00,  2.7440e+00,
         1.8800e+00,  1.1800e+00,  4.4100e-01,  5.4300e-01,  4.6600e-01,
         4.3800e-01,  3.0000e-01,  2.9000e-01,  1.5400e-01],
       [ 0.0000e+00, -4.4160e+00,  9.8400e-01,  5.3700e-01, -1.4990e+00,
        -4.9600e-01, -2.7730e+00,  2.4800e-01,  6.7680e+00,  0.0000e+00,
        -2.1200e-01,  7.2000e-01,  1.0900e+00,  2.1860e+00,  1.4200e+00,
        -6.4000e-03, -2.0200e-01,  3.9300e-01,  9.7700e-02,  3.3300e-02,
         7.5700e-03, -5.5000e-03,  0.0000e+00, -3.5000e-03,  3.6000e-03,
         1.6700e-01,  2.4100e-01,  1.4740e+00, -7.1500e-01,  1.0000e+00,
        -3.3700e-01, -2.7000e-01,  8.6500e+02, -1.1860e+00,  1.8390e+00,
         1.8800e+00,  1.1800e+00,  7.3400e-01,  4.9200e-01,  4.0900e-01,
         3.2200e-01,  3.0000e-01,  1.6600e-01,  1.0000e+00],
       [-1.0000e+00, -2.8950e+00,  1.5100e+00,  2.7000e-01, -1.2990e+00,
        -4.5300e-01, -2.4660e+00,  2.0400e-01,  5.8370e+00,  0.0000e+00,
        -1.6800e-01,  3.0500e-01,  1.7130e+00,  2.6020e+00,  2.4570e+00,
         1.0600e-01,  3.3200e-01,  5.8500e-01,  5.1700e-02,  3.2700e-02,
         6.1300e-03, -1.7000e-03,  0.0000e+00, -6.0000e-04,  1.7000e-03,
         5.9600e-01,  1.1700e-01,  1.6160e+00, -7.3300e-01,  1.0000e+00,
        -1.2800e-01, -7.5600e-01,  4.0000e+02, -1.9550e+00,  1.9290e+00,
         1.8800e+00,  1.1800e+00,  6.5500e-01,  4.9400e-01,  3.1700e-01,
         2.9700e-01,  3.0000e-01,  1.9000e-01,  6.8400e-01]])

    def getCoefs(self, period, varNames):
        varIndex = ['Period','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9',
        'c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20',
        'Dc20_CA','Dc20_JP','Dc20_CH','a2','h1','h2','h3','h4','h5','h6',
        'k1','k2','k3','c','n','f1','f2','t1','t2','flnAF','fC','rlnPGA,lnY']

        varDict = {}
        for v in varNames:
            varDict[v] = self.coefCB[self.coefCB[:,0]==period][0][varIndex.index(v)]
        return varDict
    # CB14 Eq. 2
    def magnitudeFunc(self, period, moment):
        coefMF = self.getCoefs(period, ['c0','c1','c2','c3','c4'])
        c0, c1 = coefMF['c0'], coefMF['c1']
        c2, c3 = coefMF['c2'], coefMF['c3']
        c4 = coefMF['c4']
        if moment <= 4.5:
            resMF = c0 + c1*moment
        elif 4.5 < moment <= 5.5:
            resMF = c0 + c1*moment + c2*(moment-4.5)
        elif 5.5 < moment <= 6.5:
            resMF = c0 + c1*moment + c2*(moment-4.5) + c3*(moment-5.5)
        else:
            resMF = c0 + c1*moment + c2*(moment-4.5) + c3*(moment-5.5) + c4*(moment-6.5)
        return resMF

    # CB14 Eq. 3
    def distanceFunc(self, period, moment, rrup):
        coefDF = self.getCoefs(period, ['c5','c6','c7'])
        c5, c6, c7 = coefDF['c5'], coefDF['c6'], coefDF['c7']
        rrup = np.array(rrup)
        return  (c5 + c6*moment)*np.log(np.sqrt(rrup**2+c7**2))


    # CB14 Eq. 4. 5. 6.
    def fltFunc(self, period, moment, faultType):
        coefFF = self.getCoefs(period, ['c8','c9'])
        c8, c9 = coefFF['c8'], coefFF['c9']

        if faultType == 'NR':
            Frv,Fnm = 0,1
        elif faultType == 'TH':
            Frv,Fnm = 1,0
        elif faultType == 'SS':
            Frv,Fnm = 0,0
        else:
            print("Normal: 'NR', Thrust/Reverse: 'TH', Strike-Slip: 'SS' ")
            raise ValueError

        if moment <= 4.5:
            fltM = 0
        elif 4.5 < moment <= 5.5:
            fltM = moment - 4.5
        else:
            fltM = 1

        return (c8*Frv+c9*Fnm)*fltM


    #CB14 Eq. 7,8,9,10,11,12,13,14,15,16
    def hangWallFun(self, period, moment, rjb, rrup, rx, width, ztor, dipAngle):
        """c10*FhngRx*FhngRrup*FhngM*FhngZ*FhngD"""
        coefHF = self.getCoefs(period, ['c10','a2', 'h1', 'h2', 'h3','h4','h5','h6'])
        c10, a2 = coefHF['c10'], coefHF['a2']
        h1, h2  = coefHF['h1'], coefHF['h2']
        h3, h4  = coefHF['h3'], coefHF['h4']
        h5, h6  = coefHF['h5'], coefHF['h6']
        # Eq. 16
        FhngD = (90-dipAngle)/45
        # Eq. 15
        if ztor <= 16.66:
            FhngZ=1-0.06*ztor
        else:
            FhngZ=0
        # Eq. 14
        if moment <= 5.5:
            FhngM = 0
        elif 5.5 < moment <= 6.5:
            FhngM = (moment-5.5)*(1+a2*(moment-6.5))
        elif 6.5 < moment:
            FhngM = 1+a2*(moment-6.5)
        #Eq. 13
        cHW = cl.create_some_context()
        qHW = cl.CommandQueue(cHW)
        rrup = np.array(rrup).astype(np.float32)
        rjb = np.array(rjb).astype(np.float32)
        rrupg = cl.array.to_device(qHW, rrup)
        rjbg = cl.array.to_device(qHW, rjb)
        FhngRrup = cl.array.empty_like(rrupg)
        getRrup = cl.elementwise.ElementwiseKernel(cHW,
        "float *rrupg, float *rjbg, float *FhngRrup",
        """
            if (rrupg[i]==0) {
                FhngRrup[i]=0;
            } else {
                FhngRrup[i]=(rrupg[i]-rjbg[i])/rrupg[i];
            }
        """,
        "getRrup"
        )
        getRrup(rrupg, rjbg, FhngRrup)
        FhngRrup=FhngRrup.get()

        #Eq. 12
        R2 = 62*moment-350
        #Eq. 11
        R1 = width*np.cos(np.radians(dipAngle))

        #Eq. 10
        rx = np.array(rx).astype(np.float32)
        rxg = cl.array.to_device(qHW,rx)
        F2rx = cl.array.empty_like(rxg)
        getf2rx = cl.elementwise.ElementwiseKernel(cHW,
        "float *F2rx, float *rxg, float R1, float R2, float h4, float h5, float h6",
        " F2rx[i] = h4 + h5*((rxg[i]-R1)/(R2-R1)) + h6*pow((rxg[i]-R1)/(R2-R1),2)",
        "getf2rx"
        )
        getf2rx(F2rx, rxg, R1, R2, h4, h5, h6)
        F2=F2rx.get()

        #Eq. 9
        F1rx = cl.array.empty_like(rxg)
        getf1rx = cl.elementwise.ElementwiseKernel(cHW,
        "float *F1rx, float *rxg, float h1, float h2, float h3, float R1",
        " F1rx[i] = h1 + h2*(rxg[i]/R1) + h3*pow((rxg[i]/R1),2) ",
        "getf1rx"
        )
        getf1rx(F1rx,rxg,h1,h2,h3,R1)
        F1=F1rx.get()

        F2 = np.array(F2).astype(np.float32)
        F1 = np.array(F1).astype(np.float32)
        f2rxG = cl.array.to_device(qHW, F2)
        f1rxG = cl.array.to_device(qHW, F1)
        FhngRx = cl.array.empty_like(rxg)
        getRx = cl.elementwise.ElementwiseKernel(cHW,
        "float *FhngRx, float *rxg, float *f2rxG, float *f1rxG, float R1, float elem",
        """
            if (rxg[i]<0) {
                FhngRx[i] = 0;
            } else if (0 <= rxg[i] < R1) {
                FhngRx[i] = f1rxG[i];
            } else if  (R1 <= rxg[i]) {
                FhngRx[i] = max(f2rxG[i],elem);
            }
        """,
        "getRx"
        )
        getRx(FhngRx, rxg, f2rxG, f1rxG, R1, 0)
        FhngRx=FhngRx.get()

        # Eq. 7
        resHW = c10 * FhngRx * FhngRrup * FhngM * FhngZ * FhngD
        return resHW

    #CB14 Eq. 17,18,19
    def siteResponseFunc(self, period, vs30, region, A1100):
        #initialize GPU computing with PyOpenCl
        cSR1 = cl.create_some_context()
        qSR1 = cl.CommandQueue(cSR1)

        "Fsite = FsiteG + Sj* FsiteJ"
        vs30 = np.array(vs30).astype(np.float32)
        A1100 = np.array(A1100).astype(np.float32)

        coefSR = self.getCoefs(period, ['c11','c12', 'c13', 'k1', 'k2', 'c', 'n'])
        c11, c12 = coefSR['c11'], coefSR['c12']
        c13, k1  = coefSR['c13'], coefSR['k1']
        k2, zc = coefSR['k2'], coefSR['c']
        zn = coefSR['n']

        #Eq 18
        vs30g = cl.array.to_device(qSR1, vs30)
        A1100g = cl.array.to_device(qSR1, A1100)
        FsiteG = cl.array.empty_like(vs30g)
        getSiteG= cl.elementwise.ElementwiseKernel(cSR1,
        "float *FsiteG, float *vs30g, float *A1100g, float c11, float k1, float k2, float zn, float zc",
        """
            if (vs30g[i] <= k1) {
                FsiteG[i] = c11 * log(vs30g[i]/k1) + k2 * ( log(A1100g[i] + zc * pow(vs30g[i]/k1,zn)) - log(A1100g[i]+zc) );
            } else {
                FsiteG[i] = (c11 + k2 * zn ) * log(vs30g[i]/k1);
            }
        """,
        "getSiteG"
        )
        getSiteG(FsiteG, vs30g, A1100g, c11, k1, k2, zn, zc)
        FsiteG=FsiteG.get()

        if region == 'JP':
            cSR2 = cl.create_some_context()
            qSR2 = cl.CommandQueue(cSR2)
            vs30g = cl.array.to_device(qSR2,vs30)
            FsiteJ = cl.array.empty_like(vs30g)
            getSiteJ = cl.elementwise.ElementwiseKernel(cSR,
            "float *FsiteJ, float *vs30g, float zn, float c12, float c13, float k1, float k2, float elem ",
            """
                if (vs30g[i] <= elem) {
                    FsiteJ[i] = (c12 + k2 * zn) * ( log(vs30g[i]/k1) - log(elem/k1));
                } else {
                    FsiteJ[i] = (c13 + k2 * zn) * log(vs30g[i]/k1);
                }
            """,
            "getSiteJ"
            )
            getSiteJ(FsiteJ, vs30g, zn, c12, c13, k1, k2, 200)
            FsiteJ=FsiteJ.get()
            return FsiteG+FsiteJ
        else:
            return FsiteG

    def basinResponseFunc(self, period, vs30, region):
        vs30 = np.array(vs30)
        if region == 'CA': # CB14 Eq. 33
            Sj = 0
            Z25 = np.exp(7.089-1.144 * np.log(vs30))
        elif region == 'JP': # CB14 Eq. 34
            Sj = 1
            Z25 = np.exp(5.359-1.102 * np.log(vs30))
        elif region == 'TR': # Gulerce 2014 Eq. 1 and 2
            Sj = 0
            Z25 = (0.519 + 3.595 * np.exp(28.5-0.4775*np.log(vs30**8 + 378.7**8)))/1000 # in kms

        coefBR = self.getCoefs(period, ['c14','c15','c16','k3'])
        c14, c15 = coefBR['c14'], coefBR['c15']
        c16, k3  = coefBR['c16'], coefBR['k3']

        cBR = cl.create_some_context()
        qBR = cl.CommandQueue(cBR)
        Z25 = np.array(Z25).astype(np.float32)

        Z25g = cl.array.to_device(qBR, Z25)
        Fsed = cl.array.empty_like(Z25g)
        getFsed = cl.elementwise.ElementwiseKernel(cBR,
        "float *Fsed, float *Z25g, float c14, float c15, float c16, float Sj, float k3, float elem",
        """
            if (Z25g[i] <= 1) {
                Fsed[i] = (c14 + c15 * Sj) * (Z25g[i]-1);
            } else if (1 < Z25g[i] <= 3) {
                Fsed[i] = 0;
            } else {
                Fsed[i] = c16*k3*exp(elem) * (1-exp(elem/3 * (Z25g[i]-3)));
            }
        """,
        "getFsed"
        )
        getFsed(Fsed, Z25g, c14,c15,c16,Sj,k3,-0.75)
        Fsed = Fsed.get()
        return Fsed

    def calcZhyp(self, moment, dipAngle, ztor, width):
        # Eqs. 35 36 37
        if dipAngle <= 40:
            fazs = 0.0445*(dipAngle-40)
        else:
            fazs = 0
        if moment < 6.75:
            fazm =-4.317+0.98*moment
        else:
            fazm = 2.325
        lnDZ = min(fazm+fazs, np.log(0.9*width))
        Zhyp = np.exp(lnDZ)+ztor
        return Zhyp

    def hypoDepthFunc(self, period, moment, ztor, width, dipAngle):
        "Fhyp = FhypH * FhypM" # Eq. 21
        coefHD = self.getCoefs(period, ['c17','c18'])
        c17, c18 = coefHD['c17'],coefHD['c18']

        Zhyp = self.calcZhyp(moment,dipAngle, ztor, width)
        # Eq. 23
        if moment <= 5.5:
            FhypM = c17
        elif 5.5 < moment <= 6.5:
            FhypM = c17 + (c18-c17) * (moment-5.5)
        else:
            FhypM = c18
        # Eq. 22
        if Zhyp <= 7:
            FhypH = 0
        elif 7 < Zhyp <= 20:
            FhypH = Zhyp-7
        else:
            FhypH = 13

        resHD = FhypM * FhypH
        return np.array(resHD).astype(np.float32)

    # CB14 Eq. 24
    def dipFunc(self, period, moment, dipAngle):
        c19 = self.getCoefs(period,['c19'])['c19']
        if moment <=4.5:
            dF = c19 * dipAngle
        elif 4.5 < moment <= 5.5:
            dF = c19*(5.5-moment)*dipAngle
        else:
            dF = 0
        return dF

    def aneslasticFunc(self, period, rrup, region):
        c20 = self.getCoefs(period, ['c20'])['c20']
        if region == 'CA':   # California
            Dc20 = self.getCoefs(period, ['Dc20_CA'])['Dc20_CA']
        elif region == 'JP': # Japan
            Dc20 = self.getCoefs(period, ['Dc20_JP'])['Dc20_JP']
        elif region == 'CH': # China
            Dc20 = self.getCoefs(period, ['Dc20_CH'])['Dc20_CH']
        else:
            print("'CA': California, 'JP': Japan, 'CH': China")
            raise ValueError
        rrup = np.array(rrup).astype(np.float32)

        cAF = cl.create_some_context()
        qAF = cl.CommandQueue(cAF)

        rrupg = cl.array.to_device(qAF, rrup)
        Fatn = cl.array.empty_like(rrupg)
        getFatn = cl.elementwise.ElementwiseKernel(cAF,
        "float *Fatn, float *rrupg, float c20, float Dc20",
        """
            if ( rrupg[i]>80) {
                Fatn[i] = (c20+Dc20)*(rrupg[i]-80);
            } else {
                Fatn[i] = 0;
            }
        """,
        "getFatn"
        )
        getFatn(Fatn,rrupg,c20,Dc20)
        Fatn=Fatn.get()
        return Fatn

    # Eqs. 27 28
    def getTauPhi(self, period, moment):
        t1 = self.getCoefs(period,['t1'])['t1']
        t2 = self.getCoefs(period,['t2'])['t2']
        f1 = self.getCoefs(period,['f1'])['f1']
        f2 = self.getCoefs(period,['f2'])['f2']

        if moment <= 4.5:
            return t1, f1
        elif 4.5 < moment < 5.5:
            return t2+(t1-t2)*(5.5-moment), f2+(f1-f2)*(5.5-moment)
        else:
            return t2, f2

    def sigmaTau(self, period, moment, vs30, A1100):
        # Eq. 31
        coefST = self.getCoefs(period,['k1','k2','c','n'])
        k1, k2 = coefST['k1'], coefST['k2']
        zc, zn = coefST['c'], coefST['n']

        cST = cl.create_some_context()
        qST = cl.CommandQueue(cST)

        vs30 = np.array(vs30).astype(np.float32)
        A1100 = np.array(A1100).astype(np.float32)
        vs30g = cl.array.to_device(qST, vs30)
        A1100g = cl.array.to_device(qST, A1100)
        alpha = cl.array.empty_like(vs30g)
        getAlpha = cl.elementwise.ElementwiseKernel(cST,
        "float *alpha, float *vs30g, float *A1100g, float k1, float k2 ,float zc, float zn",
        """
            if (vs30g[i]<k1) {
                alpha[i] = k2 * A1100g[i] * (1/(A1100g[i]+zc*pow(vs30g[i]/k1,zn))-1/(A1100g[i]+zc));
            } else {
                alpha[i] = 0;
            }
        """,
        "getAlpha"
        )
        getAlpha(alpha,vs30g,A1100g,k1,k2,zc,zn)
        alpha=alpha.get()
        # End of Eq. 31
        FlnAF = self.getCoefs(period,['flnAF'])['flnAF']
        rho = self.getCoefs(period,['rlnPGA,lnY'])['rlnPGA,lnY']
        tLNpga,fLNpga = self.getTauPhi(0,moment)
        tLNy, fLNy  = self.getTauPhi(period, moment)
        fLNy = np.sqrt(fLNy**2-FlnAF**2)
        fLNpga = np.sqrt(fLNpga**2-FlnAF**2)

        # Eq. 29
        tau = np.sqrt(tLNy**2 + alpha**2 * tLNpga**2 + 2*alpha*rho*tLNy*tLNpga)

        # Eq. 30
        sigma = np.sqrt(fLNy**2 + FlnAF**2 + alpha**2 * fLNpga**2 + 2*rho*alpha*fLNy*fLNpga)

        # Eq. 31
        sigmaT = np.sqrt(sigma**2+tau**2)
        return sigmaT, sigma, tau

    def getA1100(self, moment, rjb, rrup, rx, width, dipAngle, ztor, faultType, region):
        lnA1100 = self.magnitudeFunc(0,moment) +\
        self.distanceFunc(0, moment, rrup) +\
        self.fltFunc(0, moment, faultType) +\
        self.siteResponseFunc(0, 1100, region, 0) +\
        self.basinResponseFunc(0, 1100, region) +\
        self.hangWallFun(0, moment, rjb, rrup, rx, width, ztor, dipAngle) +\
        self.hypoDepthFunc(0,moment,ztor,width,dipAngle) +\
        self.dipFunc(0, moment, dipAngle) +\
        self.aneslasticFunc(0, rrup, region)

        return np.exp(lnA1100)

    def calcIM(self, period, moment, vs30, rjb, rrup, rx, width, dipAngle, ztor, faultType, region):
        a1100 = self.getA1100(moment, rjb, rrup, rx, width, dipAngle, ztor, faultType, region)
        IM = self.magnitudeFunc(period, moment)+\
        self.distanceFunc(period, moment,rrup)+\
        self.dipFunc(period, moment, dipAngle)+\
        self.fltFunc(period, moment, faultType)+\
        self.siteResponseFunc(period, vs30, region, a1100)+\
        self.basinResponseFunc(period, vs30, region)+\
        self.hangWallFun(period, moment, rjb, rrup, rx, width, ztor, dipAngle)+\
        self.hypoDepthFunc(period, moment, ztor, width, dipAngle)+\
        self.aneslasticFunc(period, rrup, region)

        sT,s,t = self.sigmaTau(period, moment, vs30, a1100)
        return np.exp(IM), sT ,s, t


    def calc_NGA(self, period, moment, vs30, rjb, rrup, rx, width, dipAngle, ztor, faultType, region):
        """
        User defined variables:
        period      : period of the event
        moment      : moment magnitude of the event
        vs30        : shear wave velocity (in m/sec)
        rjb         : closest distamce to the surface projection of the fault rupture plane (in kms)
        rrup        : closest distance to the coseismic fault rupture (in kms)
        rx          : closest distance to the surface projection of the top edge of the coseismic fault rupture (in kms)
        width       : the width of the rupture plane (in kms)
        dipAngle    : average dip angle of the fault rupture plane (in degrees)
        ztor        : depth to the top of the fault (in kms)
        faultType   : 'NR' for Normal/Oblique Normal, 'TH' for Thrust/Reverse/Oblique Reverse, 'SS' for Strike-Slip
        region      : 'CA' for California, 'JP' for Japan and 'CN' for China

        """
        IM  = self.calcIM(period, moment, vs30, rjb, rrup, rx, width, dipAngle, ztor, faultType, region)
        if period == 0:
            return IM
        elif period < 0.25 and period != -1:
            IM1 = self.calcIM(0, moment, vs30, rjb, rrup, rx, width, dipAngle, ztor, faultType, region)

            cFin = cl.create_some_context()
            qFin = cl.CommandQueue(cFin)
            IM = np.array(IM).astype(np.float32)
            IM1 = np.array(IM1).astype(np.float32)
            IM_g = cl.array.to_device(qFin, IM)
            IM1_g = cl.array.to_device(qFin, IM1)
            resIM = cl.array.empty_like(IM_g)
            getresIM = cl.elementwise.ElementwiseKernel(cFin,
            "float *resIM, float *IM_g, float *IM1_g",
            " resIM[i] = max(IM_g[i],IM1_g[i])",
            "getresIM"
            )
            getresIM(resIM, IM_g, IM1_g)
            resIMx = resIM.get()
            return resIMx



def performance(X):
    import time
    st = time.time()
    elements = [i for i in range(1,X+1)]
    a=CB14().calc_NGA(0.02, 7, elements, elements, elements, elements, 1, 50, 1, 'SS', 'CA')
    completeTime = time.time()-st
    print('Computation of %s nodes completed in %s seconds' %(len(elements), completeTime))
