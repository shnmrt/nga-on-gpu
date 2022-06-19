from __future__ import absolute_import
from __future__ import print_function
# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
import pyopencl as cl
import pyopencl.array

class ASK14():

    #ASK14 NGA Model class

    def __init__(self):
        self.n = 1.5
        self.a7 = 0
        self.coefASK=np.array([[ 0.000e+00,  6.750e+00,  6.600e+02, -1.470e+00,  2.400e+00,
         4.500e+00,  5.870e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.154e+00, -1.500e-02,  1.735e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -7.200e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -1.500e-03,
         2.500e-03, -3.400e-03, -1.503e-01,  2.650e-01,  3.370e-01,
         1.880e-01,  0.000e+00,  8.800e-02, -1.960e-01,  4.400e-02,
         7.540e-01,  5.200e-01,  4.700e-01,  3.600e-01,  7.410e-01,
         5.010e-01,  5.400e-01,  6.300e-01],
       [-1.000e+00,  6.750e+00,  3.300e+02, -2.020e+00,  2.400e+03,
         4.500e+00,  5.975e+00, -9.190e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.366e+00, -9.400e-02,  2.360e+00,  0.000e+00,
        -1.000e-01,  2.500e-01,  2.200e-01,  3.000e-01, -5.000e-04,
         2.800e-01,  1.500e-01,  9.000e-02,  7.000e-02, -1.000e-04,
         5.000e-04, -3.700e-03, -1.462e-01,  3.770e-01,  2.120e-01,
         1.570e-01,  0.000e+00,  9.500e-02, -3.800e-02,  6.500e-02,
         6.620e-01,  5.100e-01,  3.800e-01,  3.800e-01,  6.600e-01,
         5.100e-01,  5.800e-01,  5.300e-01],
       [ 1.000e-02,  6.750e+00,  6.600e+02, -1.470e+00,  2.400e+00,
         4.500e+00,  5.870e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.154e+00, -1.500e-02,  1.735e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -7.200e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -1.500e-03,
         2.500e-03, -3.400e-03, -1.503e-01,  2.650e-01,  3.370e-01,
         1.880e-01,  0.000e+00,  8.800e-02, -1.960e-01,  4.400e-02,
         7.540e-01,  5.200e-01,  4.700e-01,  3.600e-01,  7.410e-01,
         5.010e-01,  5.400e-01,  6.300e-01],
       [ 2.000e-02,  6.750e+00,  6.800e+02, -1.460e+00,  2.400e+00,
         4.500e+00,  5.980e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.146e+00, -1.500e-02,  1.718e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -7.300e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -1.500e-03,
         2.400e-03, -3.300e-03, -1.479e-01,  2.550e-01,  3.280e-01,
         1.840e-01,  0.000e+00,  8.800e-02, -1.940e-01,  6.100e-02,
         7.600e-01,  5.200e-01,  4.700e-01,  3.600e-01,  7.470e-01,
         5.010e-01,  5.400e-01,  6.300e-01],
       [ 3.000e-02,  6.750e+00,  7.700e+02, -1.390e+00,  2.400e+00,
         4.500e+00,  6.020e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.157e+00, -1.500e-02,  1.615e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -7.500e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -1.600e-03,
         2.300e-03, -3.400e-03, -1.447e-01,  2.490e-01,  3.200e-01,
         1.800e-01,  0.000e+00,  9.300e-02, -1.750e-01,  1.620e-01,
         7.810e-01,  5.200e-01,  4.700e-01,  3.600e-01,  7.690e-01,
         5.010e-01,  5.500e-01,  6.300e-01],
       [ 5.000e-02,  6.750e+00,  9.150e+02, -1.220e+00,  2.400e+00,
         4.500e+00,  7.070e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.085e+00, -1.500e-02,  1.358e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -8.000e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -2.000e-03,
         2.700e-03, -3.300e-03, -1.326e-01,  2.020e-01,  2.890e-01,
         1.670e-01,  0.000e+00,  1.330e-01, -9.000e-02,  4.510e-01,
         8.100e-01,  5.300e-01,  4.700e-01,  3.600e-01,  7.980e-01,
         5.120e-01,  5.600e-01,  6.500e-01],
       [ 7.500e-02,  6.750e+00,  9.600e+02, -1.150e+00,  2.400e+00,
         4.500e+00,  9.730e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.029e+00, -1.500e-02,  1.258e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -8.900e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -2.700e-03,
         3.200e-03, -2.900e-03, -1.353e-01,  1.260e-01,  2.750e-01,
         1.730e-01,  0.000e+00,  1.860e-01,  9.000e-02,  5.060e-01,
         8.100e-01,  5.400e-01,  4.700e-01,  3.600e-01,  7.980e-01,
         5.220e-01,  5.700e-01,  6.900e-01],
       [ 1.000e-01,  6.750e+00,  9.100e+02, -1.230e+00,  2.400e+00,
         4.500e+00,  1.169e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.041e+00, -1.500e-02,  1.310e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -9.500e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -3.300e-03,
         3.600e-03, -2.500e-03, -1.128e-01,  2.200e-02,  2.560e-01,
         1.890e-01,  0.000e+00,  1.600e-01,  6.000e-03,  3.350e-01,
         8.100e-01,  5.500e-01,  4.700e-01,  3.600e-01,  7.950e-01,
         5.270e-01,  5.700e-01,  7.000e-01],
       [ 1.500e-01,  6.750e+00,  7.400e+02, -1.590e+00,  2.400e+00,
         4.500e+00,  1.442e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.121e+00, -2.200e-02,  1.660e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -9.500e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -5.000e-02, -3.500e-03,
         3.300e-03, -2.500e-03,  3.830e-02, -1.360e-01,  1.620e-01,
         1.080e-01,  0.000e+00,  6.800e-02, -1.560e-01, -8.400e-02,
         8.010e-01,  5.600e-01,  4.700e-01,  3.600e-01,  7.730e-01,
         5.190e-01,  5.800e-01,  7.000e-01],
       [ 2.000e-01,  6.750e+00,  5.900e+02, -2.010e+00,  2.400e+00,
         4.500e+00,  1.637e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.224e+00, -3.000e-02,  2.220e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -3.000e-01,  1.100e+00, -8.600e-03,
         1.000e-01,  5.000e-02,  0.000e+00, -3.000e-02, -3.300e-03,
         2.700e-03, -3.100e-03,  7.750e-02, -7.800e-02,  2.240e-01,
         1.150e-01,  0.000e+00,  4.800e-02, -2.740e-01, -1.780e-01,
         7.890e-01,  5.650e-01,  4.700e-01,  3.600e-01,  7.530e-01,
         5.140e-01,  5.900e-01,  7.000e-01],
       [ 2.500e-01,  6.750e+00,  4.950e+02, -2.410e+00,  2.400e+00,
         4.500e+00,  1.701e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.312e+00, -3.800e-02,  2.770e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -2.400e-01,  1.100e+00, -7.400e-03,
         1.000e-01,  5.000e-02,  0.000e+00,  0.000e+00, -2.900e-03,
         2.400e-03, -3.600e-03,  7.410e-02,  3.700e-02,  2.480e-01,
         1.220e-01,  0.000e+00,  5.500e-02, -2.480e-01, -1.870e-01,
         7.700e-01,  5.700e-01,  4.700e-01,  3.600e-01,  7.290e-01,
         5.130e-01,  6.100e-01,  7.000e-01],
       [ 3.000e-01,  6.750e+00,  4.300e+02, -2.760e+00,  2.400e+00,
         4.500e+00,  1.712e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.338e+00, -4.500e-02,  3.250e+00,  0.000e+00,
        -1.000e-01,  6.000e-01, -1.900e-01,  1.030e+00, -6.400e-03,
         1.000e-01,  5.000e-02,  3.000e-02,  3.000e-02, -2.700e-03,
         2.000e-03, -3.900e-03,  2.548e-01, -9.100e-02,  2.030e-01,
         9.600e-02,  0.000e+00,  7.300e-02, -2.030e-01, -1.590e-01,
         7.400e-01,  5.800e-01,  4.700e-01,  3.600e-01,  6.930e-01,
         5.190e-01,  6.300e-01,  7.000e-01],
       [ 4.000e-01,  6.750e+00,  3.600e+02, -3.280e+00,  2.400e+00,
         4.500e+00,  1.662e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.469e+00, -5.500e-02,  3.990e+00,  0.000e+00,
        -1.000e-01,  5.800e-01, -1.100e-01,  9.200e-01, -4.300e-03,
         1.000e-01,  7.000e-02,  6.000e-02,  6.000e-02, -2.300e-03,
         1.000e-03, -4.800e-03,  2.136e-01,  1.290e-01,  2.320e-01,
         1.230e-01,  0.000e+00,  1.430e-01, -1.540e-01, -2.300e-02,
         6.990e-01,  5.900e-01,  4.700e-01,  3.600e-01,  6.440e-01,
         5.240e-01,  6.600e-01,  7.000e-01],
       [ 5.000e-01,  6.750e+00,  3.400e+02, -3.600e+00,  2.400e+00,
         4.500e+00,  1.571e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.559e+00, -6.500e-02,  4.450e+00,  0.000e+00,
        -1.000e-01,  5.600e-01, -4.000e-02,  8.400e-01, -3.200e-03,
         1.000e-01,  1.000e-01,  1.000e-01,  9.000e-02, -2.000e-03,
         8.000e-04, -5.000e-03,  1.542e-01,  3.100e-01,  2.520e-01,
         1.340e-01,  0.000e+00,  1.600e-01, -1.590e-01, -2.900e-02,
         6.760e-01,  6.000e-01,  4.700e-01,  3.600e-01,  6.160e-01,
         5.320e-01,  6.900e-01,  7.000e-01],
       [ 7.500e-01,  6.750e+00,  3.300e+02, -3.800e+00,  2.400e+00,
         4.500e+00,  1.299e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.682e+00, -9.500e-02,  4.750e+00,  0.000e+00,
        -1.000e-01,  5.300e-01,  7.000e-02,  6.800e-01, -2.500e-03,
         1.400e-01,  1.400e-01,  1.400e-01,  1.300e-01, -1.000e-03,
         7.000e-04, -4.100e-03,  7.870e-02,  5.050e-01,  2.080e-01,
         1.290e-01,  0.000e+00,  1.580e-01, -1.410e-01,  6.100e-02,
         6.310e-01,  6.150e-01,  4.700e-01,  3.600e-01,  5.660e-01,
         5.480e-01,  7.300e-01,  6.900e-01],
       [ 1.000e+00,  6.750e+00,  3.300e+02, -3.500e+00,  2.400e+00,
         4.500e+00,  1.043e+00, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.763e+00, -1.100e-01,  4.300e+00,  0.000e+00,
        -1.000e-01,  5.000e-01,  1.500e-01,  5.700e-01, -2.500e-03,
         1.700e-01,  1.700e-01,  1.700e-01,  1.400e-01, -5.000e-04,
         7.000e-04, -3.200e-03,  4.760e-02,  3.580e-01,  2.080e-01,
         1.520e-01,  0.000e+00,  1.450e-01, -1.440e-01,  6.200e-02,
         6.090e-01,  6.300e-01,  4.700e-01,  3.600e-01,  5.410e-01,
         5.650e-01,  7.700e-01,  6.800e-01],
       [ 1.500e+00,  6.750e+00,  3.300e+02, -2.400e+00,  2.400e+00,
         4.500e+00,  6.650e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.836e+00, -1.240e-01,  2.600e+00,  0.000e+00,
        -1.000e-01,  4.200e-01,  2.700e-01,  4.200e-01, -2.200e-03,
         2.200e-01,  2.100e-01,  2.000e-01,  1.600e-01, -4.000e-04,
         6.000e-04, -2.000e-03, -1.630e-02,  1.310e-01,  1.080e-01,
         1.180e-01,  0.000e+00,  1.310e-01, -1.260e-01,  3.700e-02,
         5.780e-01,  6.400e-01,  4.700e-01,  3.600e-01,  5.060e-01,
         5.760e-01,  8.000e-01,  6.600e-01],
       [ 2.000e+00,  6.750e+00,  3.300e+02, -1.000e+00,  2.400e+00,
         4.500e+00,  3.290e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.897e+00, -1.380e-01,  5.500e-01,  0.000e+00,
        -1.000e-01,  3.500e-01,  3.500e-01,  3.100e-01, -1.900e-03,
         2.600e-01,  2.500e-01,  2.200e-01,  1.600e-01, -2.000e-04,
         3.000e-04, -1.700e-03, -1.203e-01,  1.230e-01,  6.800e-02,
         1.190e-01,  0.000e+00,  8.300e-02, -7.500e-02, -1.430e-01,
         5.550e-01,  6.500e-01,  4.700e-01,  3.600e-01,  4.800e-01,
         5.870e-01,  8.000e-01,  6.200e-01],
       [ 3.000e+00,  6.820e+00,  3.300e+02,  0.000e+00,  2.400e+00,
         4.500e+00, -6.000e-02, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.906e+00, -1.720e-01, -9.500e-01,  0.000e+00,
        -1.000e-01,  2.000e-01,  4.600e-01,  1.600e-01, -1.500e-03,
         3.400e-01,  3.000e-01,  2.300e-01,  1.600e-01,  0.000e+00,
         0.000e+00, -2.000e-03, -2.719e-01,  1.090e-01, -2.300e-02,
         9.300e-02,  0.000e+00,  7.000e-02, -2.100e-02, -2.800e-02,
         5.480e-01,  6.400e-01,  4.700e-01,  3.600e-01,  4.720e-01,
         5.760e-01,  8.000e-01,  5.500e-01],
       [ 4.000e+00,  6.920e+00,  3.300e+02,  0.000e+00,  2.400e+00,
         4.500e+00, -2.990e-01, -7.900e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.889e+00, -1.970e-01, -9.500e-01,  0.000e+00,
        -1.000e-01,  0.000e+00,  5.400e-01,  5.000e-02, -1.000e-03,
         4.100e-01,  3.200e-01,  2.300e-01,  1.400e-01,  0.000e+00,
         0.000e+00, -2.000e-03, -2.958e-01,  1.350e-01,  2.800e-02,
         8.400e-02,  0.000e+00,  1.010e-01,  7.200e-02, -9.700e-02,
         5.270e-01,  6.300e-01,  4.700e-01,  3.600e-01,  4.470e-01,
         5.650e-01,  7.600e-01,  5.200e-01],
       [ 5.000e+00,  7.000e+00,  3.300e+02,  0.000e+00,  2.400e+00,
         4.500e+00, -5.620e-01, -7.650e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.898e+00, -2.180e-01, -9.300e-01,  0.000e+00,
        -1.000e-01,  0.000e+00,  6.100e-01, -4.000e-02, -1.000e-03,
         5.100e-01,  3.200e-01,  2.200e-01,  1.300e-01,  0.000e+00,
         0.000e+00, -2.000e-03, -2.718e-01,  1.890e-01,  3.100e-02,
         5.800e-02,  0.000e+00,  9.500e-02,  2.050e-01,  1.500e-02,
         5.050e-01,  6.300e-01,  4.700e-01,  3.600e-01,  4.250e-01,
         5.680e-01,  7.200e-01,  5.000e-01],
       [ 6.000e+00,  7.060e+00,  3.300e+02,  0.000e+00,  2.400e+00,
         4.500e+00, -8.750e-01, -7.110e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.896e+00, -2.350e-01, -9.100e-01,  0.000e+00,
        -2.000e-01,  0.000e+00,  6.500e-01, -1.100e-01, -1.000e-03,
         5.500e-01,  3.200e-01,  2.000e-01,  1.000e-01,  0.000e+00,
         0.000e+00, -2.000e-03, -2.517e-01,  2.150e-01,  2.400e-02,
         6.500e-02,  0.000e+00,  1.330e-01,  2.850e-01,  1.040e-01,
         4.770e-01,  6.300e-01,  4.700e-01,  3.600e-01,  3.950e-01,
         5.710e-01,  7.000e-01,  5.000e-01],
       [ 7.500e+00,  7.150e+00,  3.300e+02,  0.000e+00,  2.400e+00,
         4.500e+00, -1.303e+00, -6.340e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.870e+00, -2.550e-01, -8.700e-01,  0.000e+00,
        -2.000e-01,  0.000e+00,  7.200e-01, -1.900e-01, -1.000e-03,
         4.900e-01,  2.800e-01,  1.700e-01,  9.000e-02,  0.000e+00,
         0.000e+00, -2.000e-03, -1.400e-01,  1.500e-01, -7.000e-02,
         0.000e+00,  0.000e+00,  1.510e-01,  3.290e-01,  2.990e-01,
         4.570e-01,  6.300e-01,  4.700e-01,  3.600e-01,  3.780e-01,
         5.750e-01,  6.700e-01,  5.000e-01],
       [ 1.000e+01,  7.250e+00,  3.300e+02,  0.000e+00,  2.400e+00,
         4.500e+00, -1.928e+00, -5.290e-01,  2.750e-01, -1.000e-01,
        -4.100e-01,  2.843e+00, -2.850e-01, -8.000e-01,  0.000e+00,
        -2.000e-01,  0.000e+00,  8.000e-01, -3.000e-01, -1.000e-03,
         4.200e-01,  2.200e-01,  1.400e-01,  8.000e-02,  0.000e+00,
         0.000e+00, -2.000e-03, -2.160e-02,  9.200e-02, -1.590e-01,
        -5.000e-02,  0.000e+00,  1.240e-01,  3.010e-01,  2.430e-01,
         4.290e-01,  6.300e-01,  4.700e-01,  3.600e-01,  3.590e-01,
         5.850e-01,  6.400e-01,  5.000e-01]])


    def getCoefs(self, period, varNames):
        #varNames is a list

        varIndex = ['Period','M1','Vlin','b','c','c4','a1','a2','a3','a4','a5',
        'a6','a8','a10','a11','a12','a13','a14','a15','a17','a43','a44','a45',
        'a46','a25','a28','a29','a31','a36','a37','a38','a39','a40','a41','a42',
        's1e','s2e','s3','s4','s1m','s2m','s5j','s6j']

        varDict = {}
        for v in varNames:
            varDict[v] = self.coefASK[self.coefASK[:,0]==period][0][varIndex.index(v)]
        return varDict

    def baseModel(self, period, moment, rrup):
        # getting the Period dependend and independent variables
        varsBM = self.getCoefs(period,['c4','a1','a2','a3','a4','a5','a6','a8','a17','M1'])
        c4, a1  = varsBM['c4'], varsBM['a1']
        a2, a3  = varsBM['a2'], varsBM['a3']
        a4, a5  = varsBM['a4'], varsBM['a5']
        a6, a8  = varsBM['a6'], varsBM['a8']
        a17, M1 = varsBM['a17'], varsBM['M1']
        a7 = self.a7

        # ASK14. Eq.4
        if moment >5:
            c4m = c4
        elif 4 < moment <= 5:
            c4m = c4-(c4-1)*(5-moment)
        else:
            c4m = 1

        rrup = np.array(rrup).astype(np.float32) # converting rrup to numpy array as float32

        ctx = cl.create_some_context() # context for GPU
        queue = cl.CommandQueue(ctx)   # Queue for GPU

        rrup_g = pyopencl.array.to_device(queue, rrup) # rrup array for GPU
        res_g = pyopencl.array.empty_like(rrup_g) # result array
        # GPU implementation
        # ASK14 Eq. 2 abd Eq 3
        # Eq. 3 is in the Eq. 2 form
        baseFunc = cl.elementwise.ElementwiseKernel(ctx,
        """float c4m, float a1,  float a2,  float a3,  float a4,  float a5,
         float a6,  float a7,  float a8,  float a17 , float M1,  float *rrup_g,
          float *res_g,  float moment """,
        """
            if(moment > M1) {
                res_g[i] = a1+a5*(moment-M1)+a8*pow(8.5-moment,2)+(a2+a3*(moment-M1))*log(sqrt(pow(rrup_g[i],2)+pow(c4m,2)))+a17*rrup_g[i];
            } else if(5 <= moment <= M1) {
                res_g[i] = a1+a4*(moment-M1)+a8*pow(8.5-moment,2)+(a2+a3*(moment-M1))*log(sqrt(pow(rrup_g[i],2)+pow(c4m,2)))+a17*rrup_g[i];
            } else if (moment < 5) {
                res_g[i] = a1+a4*(5-M1)+a8*pow(8.5-5,2)+a6*(moment-5)+a7*pow(moment-5,2)+(a2+a3*(5-M1))*log(sqrt(pow(rrup_g[i],2)+pow(c4m,2)))+a17*rrup_g[i];
            }
        """,
        """baseFunc"""
        )
        baseFunc(c4m,a1,a2,a3,a4,a5,a6,a7,a8,a17,M1,rrup_g,res_g,moment)
        return(res_g.get())

    def sofModel(self, period, moment, faultType):
        varSOF = self.getCoefs(period, ['a11','a12'])
        a11, a12 = varSOF['a11'], varSOF['a12']

        if faultType == 'TH':
            Frv,Fnm=1,0
        elif faultType == 'NR':
            Frv,Fnm=0,1
        else:
            Frv,Fnm=0,0

        if moment > 5:
            f7 = a11
            f8 = a12
        elif 4 <= moment <=5:
            f7 = a11*(moment-4)
            f8 = a12*(moment-4)
        else:
            f7=0
            f8=0
        return Frv*f7 + Fnm*f8

    def getVs30star(self, period, vs30):
        # ASK14 Eq. 9
        if period <= 0.5:
            v1 = 1500
        elif 0.5 < period < 3:
            v1 = np.exp(-0.35 * np.log(period/0.5)+np.log(1500))
        elif period >= 3:
            v1 = 800

        ctx1 = cl.create_some_context()
        queue1 = cl.CommandQueue(ctx1)

        vs30 = np.array(vs30).astype(np.float32)
        vs30_g = cl.array.to_device(queue1, vs30)
        vs30star_g = cl.array.empty_like(vs30_g)

        result = cl.elementwise.ElementwiseKernel(ctx1,
        "  float *vs30_g,  float *vs30star_g, float v1",
        " vs30star_g[i] = min(vs30_g[i], v1) ",
        "result")
        result(vs30_g, vs30star_g, v1)

        return(vs30star_g.get())

    def siteResponseModel(self, period, vs30, SA1180):
        varSRM = self.getCoefs(period, ['a10','b','c','Vlin'])
        a10, b = varSRM['a10'], varSRM['b']
        zn, c   = self.n,   varSRM['c']
        Vlin   = varSRM['Vlin']
        vs30s = self.getVs30star(period, vs30)

        vs30 = np.array(vs30).astype(np.float32)
        vs30s = np.array(vs30s).astype(np.float32)
        SA1180 = np.array(SA1180).astype(np.float32)

        ctxSR = cl.create_some_context()
        queueSR = cl.CommandQueue(ctxSR)

        vs30g = cl.array.to_device(queueSR, vs30)
        vs30sg = cl.array.to_device(queueSR, vs30s)
        SA1180g = cl.array.to_device(queueSR, SA1180)
        resSA = cl.array.empty_like(vs30g)
        SRM=cl.elementwise.ElementwiseKernel(ctxSR,
        """float *vs30g, float *vs30sg, float *SA1180g, float *resSA,
        float a10, float b, float zn, float c, float Vlin""",
        """
            if (vs30g[i] >= Vlin) {
                resSA[i] = log(vs30sg[i]/Vlin) *  (a10 + (b * zn)) ;
            } else if (vs30g[i] < Vlin) {
                resSA[i] = (a10 * log(vs30sg[i]/Vlin)) -( b * log(SA1180g[i]+c)) + b * log(SA1180g[i] + c * pow(vs30sg[i]/Vlin, zn));
            }

        """,
        "SRM"
        )
        SRM(vs30g, vs30sg, SA1180g, resSA, a10, b, zn, c, Vlin)
        return resSA.get()

    #ASK Eq.10
    def hangWallModel(self, period, moment, rjb, rrup, rx, width, dipAngle, ztor): # ry0 excluded

        # ASK Eq. 11
        if dipAngle > 30:
            t1 = (90-dipAngle)/45
        else:
            t1 = 60/45
        # ASK Eq. 12
        if moment >=6.5:
            t2 = 1+0.2*(moment-6.5)
        elif 5.5 < moment < 6.5:
            t2 = 1+0.2*(moment-6.5)-(1-0.2)*(moment-6.5)**2
        else:
            t2 = 0
        #ASK Eq.14
        if ztor <= 10:
            t4 = 1-ztor**2/100
        else:
            t4 = 0
        # constanst for Eq. 13
        r1 = width * np.cos(np.radians(dipAngle))
        r2 = 3*r1
        h1, h2, h3 = 0.25, 1.5, -0.75
        dipRD = np.radians(dipAngle) # dip angle in Radians

        ctx3 = cl.create_some_context()
        queue3 = cl.CommandQueue(ctx3)


        rx = np.array(np.abs(rx)).astype(np.float32)
        rx_g = pyopencl.array.to_device(queue3, rx)
        t3_1 = pyopencl.array.empty_like(rx_g)

        calcT3_1 = cl.elementwise.ElementwiseKernel(ctx3,
        " float *rx_g,  float *t3_1, float r1, float r2, float h1, float h2, float h3",
        """
            if(rx_g[i]<r1) {
                t3_1[i] = h1+h2*(rx_g[i]/r1)+h3*pow(rx_g[i]/r1,2);
            } else if (r1 <= rx_g[i] <= r2) {
                t3_1[i] = 1-((rx_g[i]-r1)/(r2-r1));
            } else {
                t3_1[i] = 0;
            }
        """,
        "caclT3_1"
        )
        calcT3_1(rx_g, t3_1, r1, r2, h1, h2, h3)
        t3 = t3_1.get()
        # ASK Eq. 15b
        ctx4 = cl.create_some_context()
        queue4 = cl.CommandQueue(ctx4)

        rjb = np.array(rjb).astype(np.float32)
        rjb_g = pyopencl.array.to_device(queue4, rjb)
        t5_1 = pyopencl.array.empty_like(rjb_g)

        calcT5 = cl.elementwise.ElementwiseKernel(ctx4,
        " float *rjb_g,  float *t5_1",
        """
            if (rjb_g[i] == 0) {
                t5_1[i] = 1;
            } else if (rjb_g[i] < 30) {
                t5_1[i] = 1-(rjb_g[i]/30);
            } else {
                t5_1[i] = 0;
            }
        """,
        "calcT5"
        )
        calcT5(rjb_g,t5_1)
        t5 = t5_1.get()
        # ASK14 Eq. 10
        # a13*t1*t2*t3*t4*t5
        # a13, t1, t2 and t4 is float
        # t3 and t5 are array
        a13 = self.getCoefs(period,['a13'])['a13']

        ctx5 = cl.create_some_context()
        queue5 = cl.CommandQueue(ctx5)
        t3_g = pyopencl.array.to_device(queue5,t3)
        t5_g = pyopencl.array.to_device(queue5,t5)
        Reshw = pyopencl.array.empty_like(t3_g)

        calcHW = cl.elementwise.ElementwiseKernel(ctx5,
        "float a13, float t1, float t2, float t4,  float *t3_g,  float *t5_g,  float *Reshw",
        "Reshw[i]=a13*t1*t2*t3_g[i]*t4*t5_g[i]",
        "calcHW"
        )
        calcHW(a13,t1,t2,t4,t3_g,t5_g,Reshw)
        return(Reshw.get())

    def depth2rupture(self,period, ztor):
        a15 = self.getCoefs(period,['a15'])['a15']
        if ztor < 20:
            return a15 * ztor/20
        else:
            return a15


    def calcZ10(self, vs30, country):
        if country == 'JP':
            eqs = 1
        else:
            eqs = 0
        ctxZ = cl.create_some_context()
        queueZ = cl.CommandQueue(ctxZ)
        vs30 = np.array(vs30).astype(np.float32)
        vs30_g = cl.array.to_device(queueZ, vs30)
        z1_g = cl.array.empty_like(vs30_g)
        calcZ1=cl.elementwise.ElementwiseKernel(ctxZ,
        " float *vs30_g, float * z1_g, float eqs",
        """
            if (eqs == 1) {
                z1_g[i]= exp((-5.23/2) * log((pow(vs30_g[i],2)+169744)/2019344));
            } else {
                z1_g[i] = exp((-7.15/4) * log((pow(vs30_g[i],4)+106302733681)/3527322893681));
            }
        """,
        "calcZ1")
        calcZ1(vs30_g,z1_g,eqs)
        return z1_g.get()


    # ASK 14 Eq. 18 and 19
    def calcZ10mean(self, vs30, region):
        if region == 'JP':
            rg = 1
        else:
            rg = 0

        ctxZ1 = cl.create_some_context()
        queueZ1 = cl.CommandQueue(ctxZ1)
        vs30 = np.array(vs30).astype(np.float32)
        vs30_g = cl.array.to_device(queueZ1, vs30)
        z1r_g = cl.array.empty_like(vs30_g)
        calcZ1mean = cl.elementwise.ElementwiseKernel(ctxZ1,
        " float *vs30_g,  float *z1r_g, float rg",
        """
            if (rg == 1) {
                z1r_g[i] =  exp((-5.23/2)*log((pow(vs30_g[i],2)+169744)/2019344))/1000;
            } else {
                z1r_g[i] =  exp((-7.67/4)*log((pow(vs30_g[i],4)+138458410000)/3559478570000))/1000;
            }
        """,
        "calcZ1mean"
        )
        calcZ1mean(vs30_g, z1r_g, rg)
        return z1r_g.get()

    # ASK Eq. 17, 18 and 19
    def soilDepthModel(self, period, vs30, region): # calcZ1Ref returns 0  something is odd
        # ASK 14 Eq 17.
        soilCoefs = self.getCoefs(period, ['a43','a44','a45','a46'] )
        a43, a44 = soilCoefs['a43'], soilCoefs['a44']
        a45, a46 = soilCoefs['a45'], soilCoefs['a46']

        vs30_bins=np.array([150,250,400,700,1000])
        z1_scaling = np.array([a43,a44,a45,a46,a46])
        funk = interpolate.interp1d(vs30_bins, z1_scaling,fill_value='extrapolate')

        multip = funk(vs30)

        ctxSD = cl.create_some_context()
        queueSD = cl.CommandQueue(ctxSD)

        z1 = self.calcZ10(vs30,region)
        z1ref = self.calcZ10mean(vs30, region)
        z1 = np.array(z1).astype(np.float32)
        z1ref = np.array(z1ref).astype(np.float32)
        multip = np.array(multip).astype(np.float32)

        multip_g = cl.array.to_device(queueSD, multip)
        z1_g = cl.array.to_device(queueSD, z1)
        z1ref_g = cl.array.to_device(queueSD, z1ref)
        SDres = cl.array.empty_like(multip_g)

        calcSDM = cl.elementwise.ElementwiseKernel(ctxSD,
        " float *SDres,  float *multip_g,  float *z1_g,  float *z1ref_g",
        "SDres[i] = multip_g[i] * log((z1_g[i]+0.01)/(z1ref_g[i]+0.01)) ",
        "calcSDM"
        )
        calcSDM(SDres, multip_g, z1_g, z1ref_g)

        return SDres.get()

    def afterShock(self, period, CRjb, shockType):
        if shockType == 'A':
            a14 = self.getCoefs(period, ['a14'])['a14']
            ctx9 = cl.create_some_context()
            queue9 = cl.CommandQueue(ctx9)
            CRjb = np.array(CRjb).astype(np.float32)
            CRjb_g = cl.array.to_device(queue9, CRjb)
            ASres = cl.array.empty_like(CRjb_g)
            calcAS = cl.elementwise.ElementwiseKernel(ctx9,
            "float a14,  float *CRjb_g,  float *ASres",
            """
                if (CRjb_g[i] <= 5) {
                    ASres[i] = a14
                } else if (5 < CRjb[i] < 15) {
                    ASres[i] = a14 * (1- (CRjb_g[i]-5)/10)
                } else {
                    ASres[i] = 0
                }

            """,
            "calcAS"
            )
            calcAS(a14, CRjb_g, ASres)
            return(ASres.get())
        else:
            return 0



    def calcSA1180(self, period, moment, rjb, rrup, rx, CRjb, width, dipAngle, ztor, faultType, shockType, region):
        sa1180rock = 0
        vs30rock = 1180

        sa1180 =self.baseModel(period, moment, rrup) +  self.sofModel(period,moment,faultType)+\
        self.hangWallModel(period,moment, rjb, rrup, rx, width, dipAngle, ztor)+\
        self.depth2rupture(period,ztor)+\
        self.afterShock(period,CRjb,shockType) +\
        self.siteResponseModel(period,vs30rock, sa1180rock)+\
        self.regionalCorrection(period, vs30rock, rrup, region)+\
        self.soilDepthModel(period,vs30rock,region)

        return np.exp(sa1180)

    # ASK 14 Eq. 21
    def regionalCorrection(self, period, vs30, rrup, country):
        """
            Country: for Taiwan : 'TW'
                     for Japan  : 'JP'
                     for China  : 'CN'
        """
        """Regional(vs30,Rrup) = Ftw * (f12(vs30)+a25*Rrup) + Fcn(a28*Rrup) + Fjp(f13(vs30)+a29*Rrup) """

        if country == 'TW':
            twCoefs = self.getCoefs(period,['a31','a25','Vlin'])
            a31,a25,Vlin = twCoefs['a31'], twCoefs['a25'], twCoefs['Vlin']
            vs30s = self.getVs30star(period, vs30)
            ctxTW = cl.create_some_context()
            queueTW = cl.CommandQueue(ctxTW)
            vs30s = np.array(vs30s).astype(np.float32)
            rrup = np.array(rrup).astype(np.float32)
            vs30s_g = cl.array.to_device(queueTW, vs30s)
            rrup_g = cl.array.to_device(queueTW, rrup)
            resTW = cl.array.empty_like(vs30s_g)
            TW = cl.elementwise.ElementwiseKernel(ctxTW,
            """  float *vs30s_g,  float * rrup_g,  float *resTW,
            float a31, float a25, float Vlin""",
            " resTW[i] = a31 * log(vs30s_g[i]/Vlin) + a25 * rrup_g[i] ",
            "TW")
            TW(vs30s_g, rrup_g, resTW, a31, a25, Vlin)
            return resTW.get()
        elif country == 'CN':
            a28 = self.getCoefs(period,['a28'])['a28']
            return a28*rrup
        elif country == 'JP':
            jpCoefs = self.getCoefs(period,['a29', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42'])
            a29, a36 = jpCoefs['a29'], jpCoefs['a36']
            a37, a38 = jpCoefs['a37'], jpCoefs['a38']
            a39, a40 = jpCoefs['a39'], jpCoefs['a40']
            a41, a42 = jpCoefs['a41'], jpCoefs['a42']


            vs30 = np.array(vs30).astype(np.float32)
            rrup = np.array(vs30).astype(np.float32)

            ctxJP = cl.create_some_context()
            queueJP = cl.CommandQueue(ctxJP)
            vs30_g = cl.array.to_device(queueJP, vs30)
            rrup_g = cl.array.to_device(queueJP, rrup)
            resJP = cl.array.empty_like(vs30_g)
            JP = cl.elementwise.ElementwiseKernel(ctxJP,
            """ float *vs30_g, float *rrup_g,  float *resJP,
            float a29,float a36,float a37,float a38,float a39,float a40,float a41,float a42""",
            """
                if (vs30_g[i] < 200) {
                    resJP[i] = a36 + a29 * rrup_g[i];
                } else if (200 <= vs30_g[i] < 300) {
                    resJP[i] = a37 + a29 * rrup_g[i];
                } else if ( 300 <= vs30_g[i] < 400) {
                    resJP[i] = a38 + a29 * rrup_g[i];
                } else if ( 400 <= vs30_g[i] < 500) {
                    resJP[i] = a39 + a29 * rrup_g[i];
                } else if ( 500 <= vs30_g[i] < 700) {
                    resJP[i] = a40 + a29 * rrup_g[i];
                } else if ( 700 <= vs30_g[i] < 10000) {
                    resJP[i] = a41 + a29 * rrup_g[i];
                } else {
                    resJP[i] = a42 + a29 * rrup_g[i];
                }
            """,
            "JP")
            JP(vs30_g,rrup_g,resJP,a29,a36,a37,a38,a39,a40,a41,a42)
            return resJP.get()
        else:
            return 0

    # ASK 14 Eq. 30
    def calc_alpha(self, period, vs30, SA1180):
        alpCoefs= self.getCoefs(period, ['Vlin','b','c'])
        Vlin, b = alpCoefs['Vlin'], alpCoefs['b']
        c, zn = alpCoefs['c'], self.n
        vs30 = np.array(vs30).astype(np.float32)
        SA1180 = np.array(SA1180).astype(np.float32)

        ctxA = cl.create_some_context()
        queueA = cl.CommandQueue(ctxA)

        vs30_g = cl.array.to_device(queueA, vs30)
        SA1180_g = cl.array.to_device(queueA, SA1180)
        resA = cl.array.empty_like(vs30_g)

        cAlpha = cl.elementwise.ElementwiseKernel(ctxA,
        " float *vs30_g,  float *SA1180_g,  float *resA, float Vlin, float b, float c, float zn",
        """
            if (vs30_g[i] >= Vlin) {
                resA[i] = 0;
            } else {
                resA[i] = -b * SA1180_g[i]/(SA1180_g[i]+c) + b* SA1180_g[i] / (SA1180_g[i]+c*pow(vs30_g[i]/Vlin,zn));
            }
        """,
        "cAlpha")
        cAlpha(vs30_g, SA1180_g, resA, Vlin, b, c, zn)
        return resA.get()

    def calcSigmaTau(self, period, moment, rrup, alpha, vsCond, country):
        if country != 'JP':
            if vsCond == 'E':
                s1 = self.getCoefs(period, ['s1e'])['s1e']
                s2 = self.getCoefs(period, ['s2e'])['s2e']
            else:
                s1 = self.getCoefs(period, ['s1m'])['s1m']
                s2 = self.getCoefs(period, ['s2m'])['s2m']
            # ASK 15 Eq. 24
            if moment < 4:
                phiAL = s1
            elif 4 <= moment <= 6:
                phiAL = s1 + (s2-s1)/2 * (moment-4)
            else:
                phiAL = s2
        else: # ASK 15 Eq. 26
            s5 = self.getCoefs(period, ['s5j'])['s5j']
            s6 = self.getCoefs(period, ['s6j'])['s6j']
            rrup = np.array(rrup).astype(np.float32)

            ctxP = cl.create_some_context()
            queueP = cl.CommandQueue(ctxP)
            rrup_g = cl.array.to_device(queueP,rrup)
            phiAL_g = cl.array.empty_like(rrup_g)
            phiA = cl.elementwise.ElementwiseKernel(ctxP,
            " float *rrup_g,  float *phiAL_g, float s5, float s6",
            """
                if (rrup_g[i]<30) {
                    phiAL_g[i] = s5;
                } else if (30 <= rrup[i] <= 80) {
                    phiAL_g[i] = s5 + (s6-s5)/50 * (rrup_g[i]-30);
                } else {
                    phiAL_g[i] = s6;
                }
            """,
            "phiA")
            phiA(rrup_g, phiAL_g, s5, s6)
            phiAL = phiAL_g.get()

        s3 = self.getCoefs(period, ['s3'])['s3']
        s4 = self.getCoefs(period, ['s4'])['s4']
        # ASK 14 Eq. 25
        if moment < 5:
            tauAL = s3
        elif 5 <= moment <= 7:
            tauAL = s3 + (s4-s3)/2 * (moment-5)
        else:
            tauAL = s4

        phiAmp = 0.5
        phiB = np.sqrt(np.power(phiAL,2)-np.power(phiAmp,2))
        sigma = np.sqrt(np.power(np.array(phiB) * np.array(1+np.array(alpha)), 2) + np.power(phiAmp,2) )

        tauB = tauAL
        tau = np.array(tauB) * np.array(1+np.array(alpha))

        sigmaT = np.sqrt(np.power(sigma,2) + np.power(tau,2))

        return sigma,tau,sigmaT

    def calc_NGA(self, period, moment, vs30, rjb, rrup, rx, CRjb, width, dipAngle, ztor, vsCond, faultType, shockType, region, country):
        SA1180 = self.calcSA1180(period, moment, rjb, rrup, rx, CRjb, width, dipAngle, ztor, faultType, shockType, region)
        alpha = self.calc_alpha(period, vs30, SA1180)
        lnSa = self.baseModel(period, moment, rrup) + self.sofModel(period,moment,faultType) +\
        self.hangWallModel(period, moment, rjb, rrup, rx, width, dipAngle, ztor)+\
        self.depth2rupture(period, ztor) + self.soilDepthModel(period, vs30, region)+\
        self.afterShock(period, CRjb, shockType) + self.regionalCorrection(period, vs30, rrup, country)+\
        self.siteResponseModel(period, vs30, SA1180)

        s,t,st = self.calcSigmaTau(period, moment, rrup, alpha, vsCond, country)
        return np.exp(lnSa),s,t,st

def testME():
    r=[i for i in range(0,10)]
    rj=[i+1 for i in range(0,5)]
    rx=[i+2 for i in range(0,5)]
    a = ASK14().calc_NGA(0,7,150,r,r,r,0,30,90,1,'E','SS','M','CA','CA')[0]
    b = ASK14().calc_NGA(0,7,300,r,r,r,0,30,90,1,'E','SS','M','CA','CA')[0]
    c = ASK14().calc_NGA(0,7,500,r,r,r,0,30,90,1,'E','SS','M','CA','CA')[0]
    d = ASK14().calc_NGA(0,7,700,r,r,r,0,30,90,1,'E','SS','M','CA','CA')[0]
    e = ASK14().calc_NGA(0,7,1000,r,r,r,0,30,90,1,'E','SS','M','CA','CA')[0]
    f = ASK14().calc_NGA(0,7,1180,r,r,r,0,30,90,1,'E','SS','M','CA','CA')[0]
    import matplotlib.pyplot as plt
    plt.plot(r,a, 'r--', label='vs30 180')
    plt.plot(r,b, 'b--', label='vs30 300')
    plt.plot(r,c, 'y--', label='vs30 500')
    plt.plot(r,d, 'g--', label='vs30 700')
    plt.plot(r,e, 'c--', label='vs30 1000')
    plt.plot(r,f, 'm--', label='vs30 1180')

    plt.legend(loc='upper right')
    plt.title('Mw=7')
    #plt.savefig('WOsite.png')
    plt.show()
