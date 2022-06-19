from __future__ import absolute_import
from __future__ import print_function

# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl


class BSSA14():

    # BSSA14 GMPE Model Class
    """ lnY = Fe(M,mech) + Fp(Rjb,M,region) + Fs(Vs30, Rjb, M, z1) + EnS(M, Rjb,Vs30)
    E for event
    P for Path
    S for Site
    """

    def __init__(self):
        self.coefBSSA = np.array([[ 1.00000e-02,  4.53400e-01,  4.91600e-01,  2.51900e-01,
         4.59900e-01,  1.42100e+00,  4.93200e-02, -1.65900e-01,
         5.50000e+00, -1.13400e+00,  1.91600e-01, -8.08800e-03,
         4.50000e+00,  1.00000e+00,  4.50000e+00,  0.00000e+00,
         2.82000e-03, -2.44000e-03, -6.03700e-01,  1.50020e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.48330e-01,
        -7.01000e-03, -9.90000e+00, -9.90000e+00,  1.11670e+02,
         2.70000e+02,  9.60000e-02,  7.00000e-02,  2.25000e+02,
         3.00000e+02,  6.98000e-01,  4.99000e-01,  4.02000e-01,
         3.45000e-01],
       [ 2.00000e-02,  4.85980e-01,  5.23590e-01,  2.97070e-01,
         4.88750e-01,  1.43310e+00,  5.33880e-02, -1.65610e-01,
         5.50000e+00, -1.13940e+00,  1.89620e-01, -8.07400e-03,
         4.50000e+00,  1.00000e+00,  4.50000e+00,  0.00000e+00,
         2.78000e-03, -2.34000e-03, -5.73900e-01,  1.50036e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.47100e-01,
        -7.28000e-03, -9.90000e+00, -9.90000e+00,  1.13100e+02,
         2.70000e+02,  9.20000e-02,  3.00000e-02,  2.25000e+02,
         3.00000e+02,  7.02000e-01,  5.02000e-01,  4.09000e-01,
         3.46000e-01],
       [ 3.00000e-02,  5.69160e-01,  6.09200e-01,  4.03910e-01,
         5.57830e-01,  1.42610e+00,  6.14440e-02, -1.66900e-01,
         5.50000e+00, -1.14210e+00,  1.88420e-01, -8.33600e-03,
         4.50000e+00,  1.00000e+00,  4.49000e+00,  0.00000e+00,
         2.76000e-03, -2.17000e-03, -5.34100e-01,  1.50295e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.54850e-01,
        -7.35000e-03, -9.90000e+00, -9.90000e+00,  1.12130e+02,
         2.70000e+02,  8.10000e-02,  2.90000e-02,  2.25000e+02,
         3.00000e+02,  7.21000e-01,  5.14000e-01,  4.45000e-01,
         3.64000e-01],
       [ 5.00000e-02,  7.54360e-01,  7.99050e-01,  6.06520e-01,
         7.27260e-01,  1.39740e+00,  6.73570e-02, -1.80820e-01,
         5.50000e+00, -1.11590e+00,  1.87090e-01, -9.81900e-03,
         4.50000e+00,  1.00000e+00,  4.20000e+00,  0.00000e+00,
         2.96000e-03, -1.99000e-03, -4.58000e-01,  1.50142e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.92000e-01,
        -6.47000e-03, -9.90000e+00, -9.90000e+00,  9.79300e+01,
         2.70000e+02,  6.30000e-02,  3.00000e-02,  2.25000e+02,
         3.00000e+02,  7.53000e-01,  5.32000e-01,  5.03000e-01,
         4.26000e-01],
       [ 7.50000e-02,  9.64470e-01,  1.00770e+00,  7.76780e-01,
         9.56300e-01,  1.41740e+00,  7.35490e-02, -1.96650e-01,
         5.50000e+00, -1.08310e+00,  1.82250e-01, -1.05800e-02,
         4.50000e+00,  1.00000e+00,  4.04000e+00,  0.00000e+00,
         2.96000e-03, -2.16000e-03, -4.44100e-01,  1.49400e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -2.35000e-01,
        -5.73000e-03, -9.90000e+00, -9.90000e+00,  8.59900e+01,
         2.70040e+02,  6.40000e-02,  2.20000e-02,  2.25000e+02,
         3.00000e+02,  7.45000e-01,  5.42000e-01,  4.74000e-01,
         4.66000e-01],
       [ 1.00000e-01,  1.12680e+00,  1.16690e+00,  8.87100e-01,
         1.14540e+00,  1.42930e+00,  5.52310e-02, -1.98380e-01,
         5.54000e+00, -1.06520e+00,  1.72030e-01, -1.02000e-02,
         4.50000e+00,  1.00000e+00,  4.13000e+00,  0.00000e+00,
         2.88000e-03, -2.44000e-03, -4.87200e-01,  1.47912e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -2.49160e-01,
        -5.60000e-03, -9.90000e+00, -9.90000e+00,  7.95900e+01,
         2.70090e+02,  8.70000e-02,  1.40000e-02,  2.25000e+02,
         3.00000e+02,  7.28000e-01,  5.41000e-01,  4.15000e-01,
         4.58000e-01],
       [ 1.50000e-01,  1.30950e+00,  1.34810e+00,  1.06480e+00,
         1.33240e+00,  1.28440e+00, -4.20650e-02, -1.82340e-01,
         5.74000e+00, -1.05320e+00,  1.54010e-01, -8.97700e-03,
         4.50000e+00,  1.00000e+00,  4.39000e+00,  0.00000e+00,
         2.79000e-03, -2.71000e-03, -5.79600e-01,  1.44285e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -2.57130e-01,
        -5.85000e-03, -9.90000e+00, -9.90000e+00,  8.13300e+01,
         2.70160e+02,  1.20000e-01,  1.50000e-02,  2.25000e+02,
         3.00000e+02,  7.20000e-01,  5.37000e-01,  3.54000e-01,
         3.88000e-01],
       [ 2.00000e-01,  1.32550e+00,  1.35900e+00,  1.12200e+00,
         1.34140e+00,  1.13490e+00, -1.10960e-01, -1.58520e-01,
         5.92000e+00, -1.06070e+00,  1.44890e-01, -7.71700e-03,
         4.50000e+00,  1.00000e+00,  4.61000e+00,  0.00000e+00,
         2.61000e-03, -2.97000e-03, -6.87600e-01,  1.39261e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -2.46580e-01,
        -6.14000e-03, -9.90000e+00, -9.90000e+00,  9.09100e+01,
         2.70000e+02,  1.36000e-01,  4.50000e-02,  2.25000e+02,
         3.00000e+02,  7.11000e-01,  5.39000e-01,  3.44000e-01,
         3.09000e-01],
       [ 2.50000e-01,  1.27660e+00,  1.30170e+00,  1.08280e+00,
         1.30520e+00,  1.01660e+00, -1.62130e-01, -1.27840e-01,
         6.05000e+00, -1.07730e+00,  1.39250e-01, -6.51700e-03,
         4.50000e+00,  1.00000e+00,  4.78000e+00,  0.00000e+00,
         2.44000e-03, -3.14000e-03, -7.71800e-01,  1.35621e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -2.35740e-01,
        -6.44000e-03, -9.90000e+00, -9.90000e+00,  9.70400e+01,
         2.69450e+02,  1.41000e-01,  5.50000e-02,  2.25000e+02,
         3.00000e+02,  6.98000e-01,  5.47000e-01,  3.50000e-01,
         2.66000e-01],
       [ 3.00000e-01,  1.22170e+00,  1.24010e+00,  1.02460e+00,
         1.26530e+00,  9.56760e-01, -1.95900e-01, -9.28550e-02,
         6.14000e+00, -1.09480e+00,  1.33880e-01, -5.47500e-03,
         4.50000e+00,  1.00000e+00,  4.93000e+00,  0.00000e+00,
         2.20000e-03, -3.30000e-03, -8.41700e-01,  1.30847e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -2.19120e-01,
        -6.70000e-03, -9.90000e+00, -9.90000e+00,  1.03150e+02,
         2.68590e+02,  1.38000e-01,  5.00000e-02,  2.25000e+02,
         3.00000e+02,  6.75000e-01,  5.61000e-01,  3.63000e-01,
         2.29000e-01],
       [ 4.00000e-01,  1.10460e+00,  1.12140e+00,  8.97650e-01,
         1.15520e+00,  9.67660e-01, -2.26080e-01, -2.31890e-02,
         6.20000e+00, -1.12430e+00,  1.25120e-01, -4.05300e-03,
         4.50000e+00,  1.00000e+00,  5.16000e+00,  0.00000e+00,
         2.11000e-03, -3.21000e-03, -9.10900e-01,  1.25266e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.95820e-01,
        -7.13000e-03, -9.90000e+00, -9.90000e+00,  1.06020e+02,
         2.66540e+02,  1.22000e-01,  4.90000e-02,  2.25000e+02,
         3.00000e+02,  6.43000e-01,  5.80000e-01,  3.81000e-01,
         2.10000e-01],
       [ 5.00000e-01,  9.69910e-01,  9.91060e-01,  7.61500e-01,
         1.01200e+00,  1.03840e+00, -2.35220e-01,  2.91190e-02,
         6.20000e+00, -1.14590e+00,  1.20150e-01, -3.22000e-03,
         4.50000e+00,  1.00000e+00,  5.34000e+00,  0.00000e+00,
         2.35000e-03, -2.91000e-03, -9.69300e-01,  1.20391e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.75000e-01,
        -7.44000e-03, -9.90000e+00, -9.90000e+00,  1.05540e+02,
         2.65000e+02,  1.09000e-01,  6.00000e-02,  2.25000e+02,
         3.00000e+02,  6.15000e-01,  5.99000e-01,  4.10000e-01,
         2.24000e-01],
       [ 7.50000e-01,  6.69030e-01,  6.97370e-01,  4.75230e-01,
         6.91730e-01,  1.28710e+00, -2.15910e-01,  1.08290e-01,
         6.20000e+00, -1.17770e+00,  1.10540e-01, -1.93100e-03,
         4.50000e+00,  1.00000e+00,  5.60000e+00,  0.00000e+00,
         2.69000e-03, -2.53000e-03, -1.01540e+00,  1.14759e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.38660e-01,
        -8.12000e-03,  9.20000e-02,  5.90000e-02,  1.08390e+02,
         2.66510e+02,  1.00000e-01,  7.00000e-02,  2.25000e+02,
         3.00000e+02,  5.81000e-01,  6.22000e-01,  4.57000e-01,
         2.66000e-01],
       [ 1.00000e+00,  3.93200e-01,  4.21800e-01,  2.07000e-01,
         4.12400e-01,  1.50040e+00, -1.89830e-01,  1.78950e-01,
         6.20000e+00, -1.19300e+00,  1.02480e-01, -1.21000e-03,
         4.50000e+00,  1.00000e+00,  5.74000e+00,  0.00000e+00,
         2.92000e-03, -2.09000e-03, -1.05000e+00,  1.10995e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.05210e-01,
        -8.44000e-03,  3.67000e-01,  2.08000e-01,  1.16390e+02,
         2.70000e+02,  9.80000e-02,  2.00000e-02,  2.25000e+02,
         3.00000e+02,  5.53000e-01,  6.25000e-01,  4.98000e-01,
         2.98000e-01],
       [ 1.50000e+00, -1.49540e-01, -1.18660e-01, -3.13800e-01,
        -1.43700e-01,  1.76220e+00, -1.46700e-01,  3.38960e-01,
         6.20000e+00, -1.20630e+00,  9.64500e-02, -3.65000e-04,
         4.50000e+00,  1.00000e+00,  6.18000e+00,  0.00000e+00,
         3.04000e-03, -1.52000e-03, -1.04540e+00,  1.07239e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -6.20000e-02,
        -7.71000e-03,  6.38000e-01,  3.09000e-01,  1.25380e+02,
         2.62410e+02,  1.04000e-01,  1.00000e-02,  2.25000e+02,
         3.00000e+02,  5.32000e-01,  6.19000e-01,  5.25000e-01,
         3.15000e-01],
       [ 2.00000e+00, -5.86690e-01, -5.50030e-01, -7.14660e-01,
        -6.06580e-01,  1.91520e+00, -1.12370e-01,  4.47880e-01,
         6.20000e+00, -1.21590e+00,  9.63600e-02,  0.00000e+00,
         4.50000e+00,  1.00000e+00,  6.54000e+00,  0.00000e+00,
         2.92000e-03, -1.17000e-03, -1.03920e+00,  1.00949e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -3.61360e-02,
        -4.79000e-03,  8.71000e-01,  3.82000e-01,  1.30370e+02,
         2.40140e+02,  1.05000e-01,  8.00000e-03,  2.25000e+02,
         3.00000e+02,  5.26000e-01,  6.18000e-01,  5.32000e-01,
         3.29000e-01],
       [ 3.00000e+00, -1.18980e+00, -1.14200e+00, -1.23000e+00,
        -1.26640e+00,  2.13230e+00, -4.33200e-02,  6.26940e-01,
         6.20000e+00, -1.21790e+00,  9.76400e-02,  0.00000e+00,
         4.50000e+00,  1.00000e+00,  6.93000e+00,  0.00000e+00,
         2.62000e-03, -1.19000e-03, -1.01120e+00,  9.22430e+02,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.35770e-02,
        -1.83000e-03,  1.13500e+00,  5.16000e-01,  1.30360e+02,
         1.95000e+02,  8.80000e-02,  0.00000e+00,  2.25000e+02,
         3.00000e+02,  5.34000e-01,  6.19000e-01,  5.37000e-01,
         3.44000e-01],
       [ 4.00000e+00, -1.63880e+00, -1.57480e+00, -1.66730e+00,
        -1.75160e+00,  2.20400e+00, -1.46420e-02,  7.63030e-01,
         6.20000e+00, -1.21620e+00,  1.02180e-01, -5.20000e-05,
         4.50000e+00,  1.00000e+00,  7.32000e+00,  0.00000e+00,
         2.61000e-03, -1.08000e-03, -9.69400e-01,  8.44480e+02,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -3.21230e-03,
        -1.52000e-03,  1.27100e+00,  6.29000e-01,  1.29490e+02,
         1.99450e+02,  7.00000e-02,  0.00000e+00,  2.25000e+02,
         3.00000e+02,  5.36000e-01,  6.16000e-01,  5.43000e-01,
         3.49000e-01],
       [ 5.00000e+00, -1.96600e+00, -1.88820e+00, -2.02450e+00,
        -2.09280e+00,  2.22990e+00, -1.48550e-02,  8.73140e-01,
         6.20000e+00, -1.21890e+00,  1.03530e-01,  0.00000e+00,
         4.50000e+00,  1.00000e+00,  7.78000e+00,  0.00000e+00,
         2.60000e-03, -5.70000e-04, -9.19500e-01,  7.93130e+02,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -2.54800e-04,
        -1.44000e-03,  1.32900e+00,  7.38000e-01,  1.30220e+02,
         2.30000e+02,  6.10000e-02,  0.00000e+00,  2.25000e+02,
         3.00000e+02,  5.28000e-01,  6.22000e-01,  5.32000e-01,
         3.35000e-01],
       [ 7.50000e+00, -2.58650e+00, -2.48740e+00, -2.81760e+00,
        -2.68540e+00,  2.11870e+00, -8.16060e-02,  1.01210e+00,
         6.20000e+00, -1.25430e+00,  1.25070e-01,  0.00000e+00,
         4.50000e+00,  1.00000e+00,  9.48000e+00,  0.00000e+00,
         2.60000e-03,  3.80000e-04, -7.76600e-01,  7.71010e+02,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -5.46000e-05,
        -1.37000e-03,  1.32900e+00,  8.09000e-01,  1.30720e+02,
         2.50390e+02,  5.80000e-02,  0.00000e+00,  2.25000e+02,
         3.00000e+02,  5.12000e-01,  6.34000e-01,  5.11000e-01,
         2.70000e-01],
       [ 1.00000e+01, -3.07020e+00, -2.95370e+00, -3.37760e+00,
        -3.17260e+00,  1.88370e+00, -1.50960e-01,  1.06510e+00,
         6.20000e+00, -1.32530e+00,  1.51830e-01,  0.00000e+00,
         4.50000e+00,  1.00000e+00,  9.66000e+00,  0.00000e+00,
         3.03000e-03,  1.49000e-03, -6.55800e-01,  7.75000e+02,
         7.60000e+02,  0.00000e+00,  1.00000e-01,  0.00000e+00,
        -1.36000e-03,  1.18300e+00,  7.03000e-01,  1.30000e+02,
         2.10000e+02,  6.00000e-02,  0.00000e+00,  2.25000e+02,
         3.00000e+02,  5.10000e-01,  6.04000e-01,  4.87000e-01,
         2.39000e-01],
       [ 0.00000e+00,  4.47300e-01,  4.85600e-01,  2.45900e-01,
         4.53900e-01,  1.43100e+00,  5.05300e-02, -1.66200e-01,
         5.50000e+00, -1.13400e+00,  1.91700e-01, -8.08800e-03,
         4.50000e+00,  1.00000e+00,  4.50000e+00,  0.00000e+00,
         2.86000e-03, -2.55000e-03, -6.00000e-01,  1.50000e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.50000e-01,
        -7.01000e-03, -9.90000e+00, -9.90000e+00,  1.10000e+02,
         2.70000e+02,  1.00000e-01,  7.00000e-02,  2.25000e+02,
         3.00000e+02,  6.95000e-01,  4.95000e-01,  3.98000e-01,
         3.48000e-01],
       [-1.00000e+00,  5.03700e+00,  5.07800e+00,  4.84900e+00,
         5.03300e+00,  1.07300e+00, -1.53600e-01,  2.25200e-01,
         6.20000e+00, -1.24300e+00,  1.48900e-01, -3.44000e-03,
         4.50000e+00,  1.00000e+00,  5.30000e+00,  0.00000e+00,
         4.35000e-03, -3.30000e-04, -8.40000e-01,  1.30000e+03,
         7.60000e+02,  0.00000e+00,  1.00000e-01, -1.00000e-01,
        -8.44000e-03, -9.90000e+00, -9.90000e+00,  1.05000e+02,
         2.72000e+02,  8.20000e-02,  8.00000e-02,  2.25000e+02,
         3.00000e+02,  6.44000e-01,  5.52000e-01,  4.01000e-01,
         3.46000e-01]])

    def getCoefs(self,period, varNames):
        """
        varNames is a list of variables
        """
        varIndex = ['Period','e0','e1','e2','e3','e4','e5','e6','Mh','c1','c2','c3',
        'Mref','Rref','h','Dc3GLCATW','Dc3CNTR','Dc3ITJP','c','Vc','Vref','f1','f3',
        'f4','f5','f6','f7','R1','R2','DfR','DfV','V1','V2','phi1','phi2','t1','t2']

        varDict = {}
        for v in varNames:
            varDict[v] = self.coefBSSA[self.coefBSSA[:,0]==period][0][varIndex.index(v)]
        return varDict

    def eventFunction(self, period, moment, faultType): # BSSA 14 Eq. 2
        if faultType == 'U':
            U,SS,NS,RS = 1,0,0,0
        elif faultType == 'SS':
            U,SS,NS,RS = 0,1,0,0
        elif faultType == 'NR':
            U,SS,NS,RS = 0,0,1,0
        elif faultType == 'TH':
            U,SS,NS,RS = 0,0,0,1

        coefSF = self.getCoefs(period, ['Mh','e0','e1','e2','e3','e4','e5','e6'])
        Mh, e0 = coefSF['Mh'], coefSF['e0']
        e1, e2 = coefSF['e1'], coefSF['e2']
        e3, e4 = coefSF['e3'], coefSF['e4']
        e5, e6 = coefSF['e5'], coefSF['e6']

        if Mh >= moment:
            return e0*U + e1*SS + e2*NS + e3*RS + e4*(moment-Mh) + e5*(moment-Mh)**2
        else:
            return e0*U + e1*SS + e2*NS + e3*RS + e6*(moment-Mh)

    def pathFunction(self, period, moment, rjb, region): # BSSA 14 Eq. 3 and 4
        if region == 'TR' or region == 'CN':
            dc3 = self.getCoefs(period, ['Dc3CNTR'])['Dc3CNTR']
        elif region == 'IT' or region == 'JP':
            dc3 = self.getCoefs(period, ['Dc3ITJP'])['Dc3ITJP']
        elif region == 'GL' or region == 'CA' or region == 'TW':
            dc3 = self.getCoefs(period, ['Dc3GLCATW'])['Dc3GLCATW']

        coefPF = self.getCoefs(period, ['c1','c2','c3','Mref','Rref','h'])
        c1, c2 = coefPF['c1'], coefPF['c2']
        c3, h = coefPF['c3'], coefPF['h']
        Mref, Rref = coefPF['Mref'], coefPF['Rref']

        rrup = np.array(rjb).astype(np.float32)
        rLocal = np.sqrt(np.power(rrup,2)+np.power(h,2))
        rLocal = np.array(rLocal).astype(np.float32)

        ctxPF = cl.create_some_context()
        queuePF = cl.CommandQueue(ctxPF)

        rLocalg = cl.array.to_device(queuePF,rLocal)
        resPF = cl.array.empty_like(rLocalg)
        pathFunc = cl.elementwise.ElementwiseKernel(ctxPF,
        """float *resPF, float *rLocalg, float moment, float c1, float c2,
        float c3, float dc3, float Mref, float Rref""",
        " resPF[i]= (c1+c2*(moment-Mref)) * log(rLocalg[i]/Rref) + (c3 + dc3)*(rLocalg[i]-Rref) ",
        "pathFunc"
        )
        pathFunc(resPF, rLocalg, moment, c1, c2, c3, dc3, Mref, Rref)
        return resPF.get()

    def linearAmp(self, period, vs30):
        coefLA = self.getCoefs(period, ['c','Vc','Vref'])
        c, Vc, Vref = coefLA['c'], coefLA['Vc'], coefLA['Vref']

        vs30 = np.array(vs30).astype(np.float32)
        ctxLA = cl.create_some_context()
        qLA = cl.CommandQueue(ctxLA)

        vs30g = cl.array.to_device(qLA, vs30)
        lnFlin = cl.array.empty_like(vs30g)

        Flin = cl.elementwise.ElementwiseKernel(ctxLA,
        " float *lnFlin, float *vs30g, float Vc, float Vref, float c",
        " lnFlin[i] = c * log(min(vs30g[i],Vc)/Vref) ",
        "Flin"
        )
        Flin(lnFlin, vs30g, Vc, Vref, c)
        return lnFlin.get()

    def nonlinearAmp(self, period, vs30, pgaR):
        coefNL = self.getCoefs(period, ['f1','f3','f4','f5'])
        f1, f3 = coefNL['f1'], coefNL['f3']
        f4, f5 = coefNL['f4'], coefNL['f5']

        vs30 = np.array(vs30).astype(np.float32)
        pgaR = np.array(vs30).astype(np.float32)

        cNL = cl.create_some_context()
        qNL = cl.CommandQueue(cNL)

        # BSSA 14 Eq. 8
        vs30g = cl.array.to_device(qNL,vs30)
        f2g = cl.array.empty_like(vs30g)
        getF2 = cl.elementwise.ElementwiseKernel(cNL,
        "float *f2g, float *vs30g, float f4, float f5, float cnst",
        " f2g[i]= f4 * (exp(f5*(min(vs30g[i],cnst)-360)) - exp(f5*400)) ",
        "getF2"
        )
        getF2(f2g, vs30g, f5, f5, 760)
        f2 = f2g.get()


        pgaR = np.array(pgaR).astype(np.float32)
        pgaRg = cl.array.to_device(qNL, pgaR)
        f2 = np.array(f2).astype(np.float32)
        f2g = cl.array.to_device(qNL, f2)
        resNL = cl.array.empty_like(f2g)
        getNL = cl.elementwise.ElementwiseKernel(cNL,
        "float *pgaRg, float *f2g, float *resNL, float f1, float f3",
        " resNL[i] = f1 + f2g[i] * log((pgaRg[i]+f3)/f3) ",
        "getNL")
        getNL(pgaRg, f2g, resNL, f1, f3)
        return resNL.get()

    def calcZ1(self, vs30): # Gulerce 2014 Eq. 1 and also CY08 Eq. 1
        "lnZ1 = 28.5-0.4775 * ln(vs30**8,378.7**8)"
        vs30 = np.array(vs30)
        z1 = np.exp(28.5 - 0.4775 * np.log(vs30**8 + 378.7**8))
        return z1/1000 # returns in kms


    def basinDepth(self, period, vs30, region):
        coefBD = self.getCoefs(period, ['f6','f7'])
        f6, f7 = coefBD['f6'], coefBD['f7']
        vs30 = np.array(vs30)
        if region == 'CA' or 'TR':
            MZ1 = np.exp( -7.15/4 * np.log( (vs30**4 + 570.94**4)/(1360**2 + 570.94**2)))
        elif region == 'JP':
            MZ1 = np.exp( -5.23/2 * np.log( (vs30**2 + 412.39**2)/(1360**2 + 412.39**2)))
        else:
            MZ1 = np.zeros_like(vs30)
        Z1 = self.calcZ1(vs30)
        SZ1 = Z1-MZ1
        clBD = cl.create_some_context()
        qBD = cl.CommandQueue(clBD)
        SZ1 = np.array(SZ1).astype(np.float32)
        SZ1g = cl.array.to_device(qBD,SZ1)
        resBD = cl.array.empty_like(SZ1g)
        calBD = cl.elementwise.ElementwiseKernel(clBD,
        "float *resBD, float *SZ1g, float period, float f6, float f7",
        """
            if (period < 0.65) {
                resBD[i] = 0;
            } else if (period >= 0.65 && SZ1g[i] <= f7/f6) {
                resBD[i]= f6 * SZ1g[i];
            } else if (period >= 0.65 && SZ1g[i] > f7/f6) {
                resBD[i] = f7;
            }
        """,
        "calBD"
        )
        calBD(resBD, SZ1g, period, f6, f7)
        return resBD.get()


    def getPGAr(self, moment, rjb, faultType, region):
        lnSA=self.eventFunction(0,moment,faultType) +\
        self.pathFunction(0,moment,rjb,region) +\
        self.linearAmp(0,760) +\
        self.nonlinearAmp(0,760,0) +\
        self.basinDepth(0,760,region)
        return np.exp(lnSA)

    def calcSigmaTau(self, period, moment, rjb, vs30):
        coefST = self.getCoefs(period, ['R1','R2','DfR','DfV','V1','V2','phi1','phi2','t1','t2'])
        R1, R2 = coefST['R1'], coefST['R2']
        DfR, DfV = coefST['DfR'], coefST['DfV']
        V1, V2 = coefST['V1'], coefST['V2']
        phi1, phi2 = coefST['phi1'], coefST['phi2']
        t1, t2 = coefST['t1'], coefST['t2']
        # BSSA 14 Eq. 17 and Eq. 14
        if moment <= 4.5:
            fM = phi1
            tM = t1
        elif 4.5 < moment < 5.5:
            fM = phi1+(phi2-phi1)*(moment-4.5)
            tM = t1+(t2-t1)*(moment-4.5)
        else:
            fM = phi2
            tM = t2

        # BSSA Eq. 15 and 16
        clST = cl.create_some_context()
        qST  = cl.CommandQueue(clST)
        rjb = np.array(rjb).astype(np.float32)
        rjbg = cl.array.to_device(qST,rjb)
        fMRg = cl.array.empty_like(rjbg)
        # BSSA 14 Eq. 16 GPU implementation
        c_fMR = cl.elementwise.ElementwiseKernel(clST,
        "float *fMRg, float *rjbg, float fM, float DfR, float R1, float R2",
        """
            if (rjbg[i] <= R1) {
                fMRg[i] = fM;
            } else if (R1 < rjbg[i] <= R2) {
                fMRg[i] = fM + DfR * (log(rjbg[i]/R1)/log(R2/R1));
            } else if (rjbg[i]>R2) {
                fMRg[i] = fM + DfR;
            }
        """,
        "c_fMR"
        )
        c_fMR(fMRg, rjbg, fM, DfR, R1, R2)
        fMR = fMRg.get()

        #BSSA 14 Eq. 15
        clST2 = cl.create_some_context()
        qST2 = cl.CommandQueue(clST2)

        vs30 = np.array(vs30).astype(np.float32)
        vs30g = cl.array.to_device(qST2, vs30)
        fMR = np.array(fMR).astype(np.float32)
        fMRg = cl.array.to_device(qST2, fMR)
        fMRVg = cl.array.empty_like(vs30g)
        c_fMRV = cl.elementwise.ElementwiseKernel(clST2,
        "float *fMRVg, float *vs30g, float *fMRg, float V1, float V2, float DfV",
        """
            if ( V2 <= vs30g[i] ) {
                fMRVg[i]= fMRg[i];
            } else if (V1 <= vs30g[i] < V2) {
                fMRVg[i] = fMRg[i] - DfV * ( log(V2/vs30g[i])/log(V2/V1) );
            } else if (vs30g[i] < V1) {
                fMRVg[i] = fMRg[i] - DfV;
            }
        """,
        "c_FMRV"
        )
        c_fMRV(fMRVg, vs30g, fMRg, V1, V2, DfV)

        fMRV = fMRVg.get()
        fMRV = np.array(fMRV)
        tM = np.array(tM)
        sigmaT = np.sqrt(fMRV**2 + tM**2)
        return sigmaT, fMRV, tM

    def calcIM(self, period, moment, rjb, vs30, faultType, region):
        pgR = self.getPGAr(moment,rjb,faultType,region)
        IM = self.eventFunction(period, moment, faultType)+\
        self.pathFunction(period, moment, rjb, region) +\
        self.linearAmp(period,vs30) + self.nonlinearAmp(period, vs30, pgR)+\
        self.basinDepth(period, vs30, region)
        return np.exp(IM)

    def calc_NGA(self, period, moment, rjb, vs30, faultType, region):
        SA = self.calcIM(period, moment, rjb, vs30, faultType, region)
        st, s, t = self.calcSigmaTau(period, moment, rjb, vs30)
        return SA, s, t, st
