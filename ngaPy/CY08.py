from __future__ import absolute_import
from __future__ import print_function
# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import pyopencl as cl
import pyopencl.array


class CY08():
    def __init__(self):
        self.coefCY = np.array([[ 1.00000e-02,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.26870e+00,  1.00000e-01, -2.55000e-01,  2.99600e+00,
         4.18400e+00,  6.16000e+00,  4.89300e-01,  5.12000e-02,
         8.60000e-02,  7.90000e-01,  1.50050e+00, -3.21800e-01,
        -8.04000e-03, -7.85000e-03, -4.41700e-01, -1.41700e-01,
        -7.01000e-03,  1.02151e-01,  2.28900e-01,  1.49960e-02,
         5.80000e+02,  7.00000e-02,  3.43700e-01,  2.63700e-01,
         4.45800e-01,  3.45900e-01,  8.00000e-01,  6.63000e-02],
       [ 2.00000e-02,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.25150e+00,  1.00000e-01, -2.55000e-01,  3.29200e+00,
         4.18790e+00,  6.15800e+00,  4.89200e-01,  5.12000e-02,
         8.60000e-02,  8.12900e-01,  1.50280e+00, -3.32300e-01,
        -8.11000e-03, -7.92000e-03, -4.34000e-01, -1.36400e-01,
        -7.27900e-03,  1.08360e-01,  2.28900e-01,  1.49960e-02,
         5.80000e+02,  6.99000e-02,  3.47100e-01,  2.67100e-01,
         4.45800e-01,  3.45900e-01,  8.00000e-01,  6.63000e-02],
       [ 3.00000e-02,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.17440e+00,  1.00000e-01, -2.55000e-01,  3.51400e+00,
         4.15560e+00,  6.15500e+00,  4.89000e-01,  5.11000e-02,
         8.60000e-02,  8.43900e-01,  1.50710e+00, -3.39400e-01,
        -8.39000e-03, -8.19000e-03, -4.17700e-01, -1.40300e-01,
        -7.35400e-03,  1.19888e-01,  2.28900e-01,  1.49960e-02,
         5.80000e+02,  7.01000e-02,  3.60300e-01,  2.80300e-01,
         4.53500e-01,  3.53700e-01,  8.00000e-01,  6.63000e-02],
       [ 4.00000e-02,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.06710e+00,  1.00000e-01, -2.55000e-01,  3.56300e+00,
         4.12260e+00,  6.15080e+00,  4.88800e-01,  5.08000e-02,
         8.60000e-02,  8.74000e-01,  1.51380e+00, -3.45300e-01,
        -8.75000e-03, -8.55000e-03, -4.00000e-01, -1.59100e-01,
        -6.97700e-03,  1.33641e-01,  2.28900e-01,  1.49960e-02,
         5.79900e+02,  7.02000e-02,  3.71800e-01,  2.91800e-01,
         4.58900e-01,  3.59200e-01,  8.00000e-01,  6.63000e-02],
       [ 5.00000e-02,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -9.46400e-01,  1.00000e-01, -2.55000e-01,  3.54700e+00,
         4.10110e+00,  6.14410e+00,  4.88400e-01,  5.04000e-02,
         8.60000e-02,  8.99600e-01,  1.52300e+00, -3.50200e-01,
        -9.12000e-03, -8.91000e-03, -3.90300e-01, -1.86200e-01,
        -6.46700e-03,  1.48927e-01,  2.29000e-01,  1.49960e-02,
         5.79900e+02,  7.01000e-02,  3.84800e-01,  3.04800e-01,
         4.63000e-01,  3.63500e-01,  8.00000e-01,  6.63000e-02],
       [ 7.50000e-02,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -7.05100e-01,  1.00000e-01, -2.54000e-01,  3.44800e+00,
         4.08600e+00,  6.12000e+00,  4.87200e-01,  4.95000e-02,
         8.60000e-02,  9.44200e-01,  1.55970e+00, -3.57900e-01,
        -9.73000e-03, -9.50000e-03, -4.04000e-01, -2.53800e-01,
        -5.73400e-03,  1.90596e-01,  2.29200e-01,  1.49960e-02,
         5.79600e+02,  6.86000e-02,  3.87800e-01,  3.12900e-01,
         4.70200e-01,  3.71300e-01,  8.00000e-01,  6.63000e-02],
       [ 1.00000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -5.74700e-01,  1.00000e-01, -2.53000e-01,  3.31200e+00,
         4.10300e+00,  6.08500e+00,  4.85400e-01,  4.89000e-02,
         8.60000e-02,  9.67700e-01,  1.61040e+00, -3.60400e-01,
        -9.75000e-03, -9.52000e-03, -4.42300e-01, -2.94300e-01,
        -5.60400e-03,  2.30662e-01,  2.29700e-01,  1.49960e-02,
         5.79200e+02,  6.46000e-02,  3.83500e-01,  3.15200e-01,
         4.74700e-01,  3.76900e-01,  8.00000e-01,  6.63000e-02],
       [ 1.50000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -5.30900e-01,  1.00000e-01, -2.50000e-01,  3.04400e+00,
         4.17170e+00,  5.98710e+00,  4.80800e-01,  4.79000e-02,
         8.60000e-02,  9.66000e-01,  1.75490e+00, -3.56500e-01,
        -8.83000e-03, -8.62000e-03, -5.16200e-01, -3.11300e-01,
        -5.84500e-03,  2.66468e-01,  2.32600e-01,  1.49880e-02,
         5.77200e+02,  4.94000e-02,  3.71900e-01,  3.12800e-01,
         4.79800e-01,  3.84700e-01,  8.00000e-01,  6.12000e-02],
       [ 2.00000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -6.35200e-01,  1.00000e-01, -2.44900e-01,  2.83100e+00,
         4.24760e+00,  5.86990e+00,  4.75500e-01,  4.71000e-02,
         8.60000e-02,  9.33400e-01,  1.91570e+00, -3.47000e-01,
        -7.78000e-03, -7.59000e-03, -5.69700e-01, -2.92700e-01,
        -6.14100e-03,  2.55253e-01,  2.38600e-01,  1.49640e-02,
         5.73900e+02, -1.90000e-03,  3.60100e-01,  3.07600e-01,
         4.81600e-01,  3.90200e-01,  8.00000e-01,  5.30000e-02],
       [ 2.50000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -7.76600e-01,  1.00000e-01, -2.38200e-01,  2.65800e+00,
         4.31840e+00,  5.75470e+00,  4.70600e-01,  4.64000e-02,
         8.60000e-02,  8.94600e-01,  2.07090e+00, -3.37900e-01,
        -6.88000e-03, -6.71000e-03, -6.10900e-01, -2.66200e-01,
        -6.43900e-03,  2.31541e-01,  2.49700e-01,  1.48810e-02,
         5.68500e+02, -4.79000e-02,  3.52200e-01,  3.04700e-01,
         4.81500e-01,  3.94600e-01,  7.99900e-01,  4.57000e-02],
       [ 3.00000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -9.27800e-01,  9.99000e-02, -2.31300e-01,  2.50500e+00,
         4.38440e+00,  5.65270e+00,  4.66500e-01,  4.58000e-02,
         8.60000e-02,  8.59000e-01,  2.20050e+00, -3.31400e-01,
        -6.12000e-03, -5.98000e-03, -6.44400e-01, -2.40500e-01,
        -6.70400e-03,  2.07277e-01,  2.67400e-01,  1.46390e-02,
         5.60500e+02, -7.56000e-02,  3.43800e-01,  3.00500e-01,
         4.80100e-01,  3.98100e-01,  7.99700e-01,  3.98000e-02],
       [ 4.00000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.21760e+00,  9.97000e-02, -2.14600e-01,  2.26100e+00,
         4.49790e+00,  5.49970e+00,  4.60700e-01,  4.45000e-02,
         8.50000e-02,  8.01900e-01,  2.38860e+00, -3.25600e-01,
        -4.98000e-03, -4.86000e-03, -6.93100e-01, -1.97500e-01,
        -7.12500e-03,  1.65464e-01,  3.12000e-01,  1.34930e-02,
         5.40000e+02, -9.60000e-02,  3.35100e-01,  2.98400e-01,
         4.75800e-01,  4.03600e-01,  7.98800e-01,  3.12000e-02],
       [ 5.00000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.46950e+00,  9.91000e-02, -1.97200e-01,  2.08700e+00,
         4.58810e+00,  5.40290e+00,  4.57100e-01,  4.29000e-02,
         8.30000e-02,  7.57800e-01,  2.50000e+00, -3.18900e-01,
        -4.20000e-03, -4.10000e-03, -7.24600e-01, -1.63300e-01,
        -7.43500e-03,  1.33828e-01,  3.61000e-01,  1.11330e-02,
         5.12900e+02, -9.98000e-02,  3.35300e-01,  3.03600e-01,
         4.71000e-01,  4.07900e-01,  7.96600e-01,  2.55000e-02],
       [ 7.50000e-01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.92780e+00,  9.36000e-02, -1.62000e-01,  1.81200e+00,
         4.75710e+00,  5.29000e+00,  4.53100e-01,  3.87000e-02,
         6.90000e-02,  6.78800e-01,  2.62240e+00, -2.70200e-01,
        -3.08000e-03, -3.01000e-03, -7.70800e-01, -1.02800e-01,
        -8.12000e-03,  8.51530e-02,  4.35300e-01,  6.73900e-03,
         4.41900e+02, -7.65000e-02,  3.42900e-01,  3.20500e-01,
         4.62100e-01,  4.15700e-01,  7.79200e-01,  1.75000e-02],
       [ 1.00000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -2.24530e+00,  7.66000e-02, -1.40000e-01,  1.64800e+00,
         4.88200e+00,  5.24800e+00,  4.51700e-01,  3.50000e-02,
         4.50000e-02,  6.19600e-01,  2.66900e+00, -2.05900e-01,
        -2.46000e-03, -2.41000e-03, -7.99000e-01, -6.99000e-02,
        -8.44400e-03,  5.85950e-02,  4.62900e-01,  5.74900e-03,
         3.91800e+02, -4.12000e-02,  3.57700e-01,  3.41900e-01,
         4.58100e-01,  4.21300e-01,  7.50400e-01,  1.33000e-02],
       [ 1.50000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -2.73070e+00,  2.20000e-03, -1.18400e-01,  1.51100e+00,
         5.06970e+00,  5.21940e+00,  4.50700e-01,  2.80000e-02,
         1.34000e-02,  5.10100e-01,  2.69850e+00, -8.52000e-02,
        -1.80000e-03, -1.76000e-03, -8.38200e-01, -4.25000e-02,
        -7.70700e-03,  3.17870e-02,  4.75600e-01,  5.54400e-03,
         3.48100e+02,  1.40000e-02,  3.76900e-01,  3.70300e-01,
         4.49300e-01,  4.21300e-01,  7.13600e-01,  9.00000e-03],
       [ 2.00000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -3.14130e+00, -5.91000e-02, -1.10000e-01,  1.47000e+00,
         5.21730e+00,  5.20990e+00,  4.50400e-01,  2.13000e-02,
         4.00000e-03,  3.91700e-01,  2.70850e+00,  1.60000e-02,
        -1.47000e-03, -1.43000e-03, -8.66300e-01, -3.02000e-02,
        -4.79200e-03,  1.97160e-02,  4.78500e-01,  5.52100e-03,
         3.32500e+02,  5.44000e-02,  4.02300e-01,  4.02300e-01,
         4.45900e-01,  4.21300e-01,  7.03500e-01,  6.80000e-03],
       [ 3.00000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -3.74130e+00, -9.31000e-02, -1.04000e-01,  1.45600e+00,
         5.43850e+00,  5.20400e+00,  4.50100e-01,  1.06000e-02,
         1.00000e-03,  1.24400e-01,  2.71450e+00,  1.87600e-01,
        -1.17000e-03, -1.15000e-03, -9.03200e-01, -1.29000e-02,
        -1.82800e-03,  9.64300e-03,  4.79600e-01,  5.51700e-03,
         3.24100e+02,  1.23200e-01,  4.40600e-01,  4.40600e-01,
         4.43300e-01,  4.21300e-01,  7.00600e-01,  4.50000e-03],
       [ 4.00000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -4.18140e+00, -9.82000e-02, -1.02000e-01,  1.46500e+00,
         5.59770e+00,  5.20200e+00,  4.50100e-01,  4.10000e-03,
         0.00000e+00,  8.60000e-03,  2.71640e+00,  3.37800e-01,
        -1.07000e-03, -1.04000e-03, -9.23100e-01, -1.60000e-03,
        -1.52300e-03,  5.37900e-03,  4.79900e-01,  5.51700e-03,
         3.21700e+02,  1.85900e-01,  4.78400e-01,  4.78400e-01,
         4.42400e-01,  4.21300e-01,  7.00100e-01,  3.40000e-03],
       [ 5.00000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -4.51870e+00, -9.94000e-02, -1.01000e-01,  1.47800e+00,
         5.72760e+00,  5.20100e+00,  4.50000e-01,  1.00000e-03,
         0.00000e+00,  0.00000e+00,  2.71720e+00,  4.57900e-01,
        -1.02000e-03, -9.90000e-04, -9.22200e-01,  0.00000e+00,
        -1.44000e-03,  3.22300e-03,  4.79900e-01,  5.51700e-03,
         3.20900e+02,  2.29500e-01,  5.07400e-01,  5.07400e-01,
         4.42000e-01,  4.21300e-01,  7.00000e-01,  2.70000e-03],
       [ 7.50000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -5.12240e+00, -9.99000e-02, -1.01000e-01,  1.49800e+00,
         5.98910e+00,  5.20000e+00,  4.50000e-01,  0.00000e+00,
         0.00000e+00,  0.00000e+00,  2.71770e+00,  7.51400e-01,
        -9.60000e-04, -9.40000e-04, -8.34600e-01,  0.00000e+00,
        -1.36900e-03,  1.13400e-03,  4.80000e-01,  5.51700e-03,
         3.20300e+02,  2.66000e-01,  5.32800e-01,  5.32800e-01,
         4.41600e-01,  4.21300e-01,  7.00000e-01,  1.80000e-03],
       [ 1.00000e+01,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -5.58720e+00, -1.00000e-01, -1.00000e-01,  1.50200e+00,
         6.19300e+00,  5.20000e+00,  4.50000e-01,  0.00000e+00,
         0.00000e+00,  0.00000e+00,  2.71800e+00,  1.18560e+00,
        -9.40000e-04, -9.10000e-04, -7.33200e-01,  0.00000e+00,
        -1.36100e-03,  5.15000e-04,  4.80000e-01,  5.51700e-03,
         3.20100e+02,  2.68200e-01,  5.54200e-01,  5.54200e-01,
         4.41400e-01,  4.21300e-01,  7.00000e-01,  1.40000e-03],
       [ 0.00000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
        -1.26870e+00,  1.00000e-01, -2.55000e-01,  2.99600e+00,
         4.18400e+00,  6.16000e+00,  4.89300e-01,  5.12000e-02,
         8.60000e-02,  7.90000e-01,  1.50050e+00, -3.21800e-01,
        -8.04000e-03, -7.85000e-03, -4.41700e-01, -1.41700e-01,
        -7.01000e-03,  1.02151e-01,  2.28900e-01,  1.49960e-02,
         5.80000e+02,  7.00000e-02,  3.43700e-01,  2.63700e-01,
         4.45800e-01,  3.45900e-01,  8.00000e-01,  6.63000e-02],
       [-1.00000e+00,  1.06000e+00,  3.45000e+00, -2.10000e+00,
        -5.00000e-01,  5.00000e+01,  3.00000e+00,  4.00000e+00,
         2.28840e+00,  1.09400e-01, -6.26000e-02,  1.64800e+00,
         4.29790e+00,  5.17000e+00,  4.40700e-01,  2.07000e-02,
         4.37000e-02,  3.07900e-01,  2.66900e+00, -1.16600e-01,
        -2.75000e-03, -6.25000e-03, -7.86100e-01, -6.99000e-02,
        -8.44400e-03,  5.41000e+00,  2.89900e-01,  6.71800e-03,
         4.59000e+02,  1.13800e-01,  2.53900e-01,  2.38100e-01,
         4.49600e-01,  3.55400e-01,  7.50400e-01,  1.33000e-02]])

    def __call__(self, period, Moment, Vs30, Rrup, Rjb, Rx, DipAngle, Azimuth, Ztor, FaultType, shockType, VsCondition):
        self.period = period
        self.ztor = Ztor
        self.moment = Moment
        self.vs30 = Vs30
        self.rrup = Rrup
        self.rjb = Rjb
        self.rx = Rx
        self.dip = DipAngle
        self.azimuth = Azimuth

        if FaultType == 'NR':
            self.Fnm, self.Frv = 1,0
        elif FaultType == 'TH':
            self.Fnm, self.Frv = 0,1
        elif FaultType == 'SS':
            self.Fnm, self.Frv = 0,0
        else:
            print("""
            Fault Type must be following format
            NR : Normal Fault 
            TH : Thrust/Reverse Fault
            SS : Strike-Slip Fault            
            """)
            raise ValueError

        if shockType == 'M':
            self.Fas = 0
        elif shockType == 'A':
            self.Fas = 1
        else:
            print("Shock type must be 'M' for Main or 'A' for Aftershock")
            raise ValueError

        if VsCondition == 'M':
            self.Finf, self.Fmes = 0,1
        elif VsCondition == 'E':
            self.Finf, self.Fmes = 1,0
        else:
            print("""
            Vs30 Condition should be in following format:
            M for Measured
            E f??r Estimated """)
            raise ValueError


        #PyOpenCL Implementation
        self.CTX = cl.create_some_context()
        self.QUE = cl.CommandQueue(self.CTX)


    def get_coefs(self, varNames, periodTerm=None):
        if periodTerm is None:
            periodTerm = self.period
        
        varIndex = ['Period', 'c2', 'c3', 'c4', 'c4a', 'cRB', 'cHM', 'cg3', 'c1', 'c1a', 'c1b', 'cn', 'cM', 'c5', 'c6', 'c7', 'c7a', 'c9', 'c9a', 'c10', 'cg1', 'cg2', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 't1', 't2', 's1', 's2', 's3', 's4']

        varDict = {}
        for v in varNames:
            varDict[v] = self.coefCY[self.coefCY[:,0]==periodTerm][0][varIndex.index(v)]
        return varDict

    def styleFault(self, periodTerm=None):
        coefSF = self.get_coefs(['c1','c1a','c1b','c7','c7a','c10'],periodTerm)
        term0 = coefSF['c1'] + (coefSF['c1a'] * self.Frv + coefSF['c1b'] * self.Fnm + coefSF['c7'] * (self.ztor-4)) * (1-self.Fas)
        term1 = (coefSF['c10']+coefSF['c7a']*(self.ztor-4))*self.Fas
        return (term0 + term1)

    def momentModel(self, periodTerm=None):
        coefMF = self.get_coefs(['c2','c3','cn','cM'],periodTerm)
        term3 = coefMF['c2']*(self.moment-6) + ((coefMF['c2']-coefMF['c3'])/coefMF['cn'])*np.log(1+np.exp(coefMF['cn']*(coefMF['cM']-self.moment)))
        return term3

    def distanceModel(self,periodTerm=None):
        coefDM = self.get_coefs(['c4','c5','c6','cHM','c4a','cRB','cg1','cg2','cg3'],periodTerm)

        term4 = coefDM['c4'] * np.log(self.rrup+coefDM['c5']*np.cosh(coefDM['c6']*max(self.moment-coefDM['cHM'],0)))
        term5 = (coefDM['c4a']-coefDM['c4']) * np.log(np.sqrt(self.rrup**2 + coefDM['cRB']**2))
        term6 = (coefDM['cg1']+coefDM['cg2']/np.cosh(max(self.moment-coefDM['cg3'],0)))*self.rrup
        return term4+term5+term6
    
    def hangingModel(self, periodTerm=None):
        coefHW = self.get_coefs(['c9','c9a'])

        rx = np.array(self.rx).astype(np.float32)
        azi = np.array(self.azimuth).astype(np.float32)
        rx = cl.array.to_device(self.QUE, rx)
        azi = cl.array.to_device(self.QUE, azi)
        Fhw = cl.array.empty_like(rx)
        getFhw = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *fhw, float *rx, float *azi",
        """
            if (rx[i]<0) {
                fhw[i] = 0;
            } else if (azi[i]<0) {
                fhw[i]=0;
            } else {
                fhw[i] = 1;
            }
        """,
        "getFhw")
        getFhw(Fhw, rx, azi)

        d = np.radians(self.dip)

        term7 = coefHW['c9']*Fhw.get()*np.tanh(self.rx*np.cos(d)*np.cos(d)/coefHW['c9a']) * (1 - np.sqrt(self.rjb**2 + self.ztor**2)/(self.rrup+0.001))

        return term7

    def lnYref(self):
        return self.momentModel()+self.distanceModel()+self.styleFault()+self.hangingModel()

    def siteModel(self,periodTerm=None):
        coefSM = self.get_coefs(['f1','f2','f3','f4'],periodTerm)
        lnYref = self.lnYref()
        lnYref = np.array(lnYref).astype(np.float32)
        lnYref = cl.array.to_device(self.QUE,lnYref)
        vs = np.array(self.vs30).astype(np.float32)
        vs = cl.array.to_device(self.QUE, vs)
        sm = cl.array.empty_like(vs)

        getSM = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *sm, float *vs, float *lnY, float f1, float f2, float f3, float f4",
        """
        float const1 = 0;
        float const2 = 1130;
        sm[i] = (f1 * min(log(vs[i]/const2),const1)) + (f2*( exp(f3*(min(vs[i],const2)-360)) - exp(f3*(const2-360)) )* log((exp(lnY[i])+f4)/f4));

        //( f2 * (exp(f3*(min(vs[i],const2)-360)) - exp(f3*(const2-360)))*log(exp(lnY[i])+f4)/f4);""",
        "getSM")

        getSM(sm, vs, lnYref, coefSM['f1'], coefSM['f2'], coefSM['f3'], coefSM['f4'])
        return sm.get()

    def basinModel(self,periodTerm=None):
        coefBM = self.get_coefs(['f5','f6','f7','f8'],periodTerm)
        z1 = np.exp(28.5-0.4775*np.log(self.vs30**8+378.7**8))
        #print(z1)
        z1 = np.array(z1).astype(np.float32)
        z1 = cl.array.to_device(self.QUE, z1)
        bm = cl.array.empty_like(z1)

        getBM = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *bm, float *z1, float f5, float f6, float f7, float f8",
        """
        float const1 = 0;
        bm[i] = (f5*(1-1/cosh(f6*max(const1, z1[i]-f7))))  +  (f8/cosh(0.15*max(const1, z1[i]-15)));
        """,
        "getBM")
        getBM(bm, z1, coefBM['f5'], coefBM['f6'], coefBM['f7'], coefBM['f8'])
        return bm.get()

    def get_SA(self):
        return np.exp(\
            self.momentModel()+\
                self.distanceModel()+\
                    self.styleFault()+\
                        self.hangingModel()+\
                            self.siteModel()+\
                                self.basinModel())


    def get_NL(self,periodTerm=None):
        coefNL = self.get_coefs(['f2','f3','f4'],periodTerm)
        yref = np.exp(self.lnYref())

        vs = np.array(self.vs30).astype(np.float32)
        vs = cl.array.to_device(self.QUE, vs)
        b = cl.array.empty_like(vs)
        getB = cl.elementwise.ElementwiseKernel(self.CTX, 
        "float *cnstB, float *vs, float f2, float f3, float c1, float c2",
        " cnstB[i] = f2 * (exp(f3*(min(vs[i],c1)-c2)) - exp(f3*(c1-c2)) ); ",
        "getB")
        getB(b, vs, coefNL['f2'], coefNL['f3'], 1130, 360)
        return b.get()*yref / (yref+coefNL['f4'])


    def get_sigmaTau(self,periodTerm=None):
        coefST = self.get_coefs(['t1', 't2', 's1', 's2', 's3', 's4'], periodTerm)
        NL = self.get_NL()
        tau = coefST['t1'] + 0.5 * (coefST['t2']-coefST['t1']) * (min(max(self.moment,5),7)-5)
        sigma = (coefST['s1'] + 0.5 *(coefST['s2']-coefST['s1'])* (min(max(self.moment,5),7)-5) + coefST['s4'] * self.Fas ) * np.sqrt((coefST['s3']*self.Finf+0.7*self.Fmes)+(1+NL)**2)

        tauNL = (1+NL)*tau
        sigmaT = np.sqrt(sigma**2 + tauNL**2)

        return (sigma, tauNL, sigmaT)



test = CY08()
test(0,7,180,10,10,10,30,45,0,'NR','M','M')

test.get_SA()
