from __future__ import absolute_import
from __future__ import print_function
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import pyopencl.array

class AS08():
    def __init__(self):
        self.coefAS = np.array([[ 1.0000e-02,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  8.6510e+02,
        -1.1860e+00,  8.1100e-01, -9.6790e-01, -3.7200e-02,  9.4450e-01,
         0.0000e+00, -6.0000e-02,  1.0800e+00, -3.5000e-01,  9.0000e-01,
        -6.7000e-03,  5.9000e-01,  4.7000e-01,  5.7600e-01,  4.5300e-01,
         4.2000e-01,  3.0000e-01,  1.0000e+00],
       [ 2.0000e-02,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  8.6510e+02,
        -1.2190e+00,  8.5500e-01, -9.7740e-01, -3.7200e-02,  9.8340e-01,
         0.0000e+00, -6.0000e-02,  1.0800e+00, -3.5000e-01,  9.0000e-01,
        -6.7000e-03,  5.9000e-01,  4.7000e-01,  5.7600e-01,  4.5300e-01,
         4.2000e-01,  3.0000e-01,  1.0000e+00],
       [ 3.0000e-02,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  9.0780e+02,
        -1.2730e+00,  9.6200e-01, -1.0024e+00, -3.7200e-02,  1.0471e+00,
         0.0000e+00, -6.0000e-02,  1.1331e+00, -3.5000e-01,  9.0000e-01,
        -6.7000e-03,  6.0500e-01,  4.7800e-01,  5.9100e-01,  4.6100e-01,
         4.6200e-01,  3.0500e-01,  9.9100e-01],
       [ 4.0000e-02,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  9.9450e+02,
        -1.3080e+00,  1.0370e+00, -1.0289e+00, -3.1500e-02,  1.0884e+00,
         0.0000e+00, -6.0000e-02,  1.1708e+00, -3.5000e-01,  9.0000e-01,
        -6.7000e-03,  6.1500e-01,  4.8300e-01,  6.0200e-01,  4.6600e-01,
         4.9200e-01,  3.0900e-01,  9.8200e-01],
       [ 5.0000e-02,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  1.0535e+03,
        -1.3460e+00,  1.1330e+00, -1.0508e+00, -2.7100e-02,  1.1333e+00,
         0.0000e+00, -6.0000e-02,  1.2000e+00, -3.5000e-01,  9.0000e-01,
        -7.6000e-03,  6.2300e-01,  4.8800e-01,  6.1000e-01,  4.7100e-01,
         5.1500e-01,  3.1200e-01,  9.7300e-01],
       [ 7.5000e-02,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  1.0857e+03,
        -1.4710e+00,  1.3750e+00, -1.0810e+00, -1.9100e-02,  1.2808e+00,
         0.0000e+00, -6.0000e-02,  1.2000e+00, -3.5000e-01,  9.0000e-01,
        -9.3000e-03,  6.3000e-01,  4.9500e-01,  6.1700e-01,  4.7900e-01,
         5.5000e-01,  3.1700e-01,  9.5200e-01],
       [ 1.0000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  1.0325e+03,
        -1.6240e+00,  1.5630e+00, -1.0833e+00, -1.6600e-02,  1.4613e+00,
         0.0000e+00, -6.0000e-02,  1.2000e+00, -3.5000e-01,  9.0000e-01,
        -9.3000e-03,  6.3000e-01,  5.0100e-01,  6.1700e-01,  4.8500e-01,
         5.5000e-01,  3.2100e-01,  9.2900e-01],
       [ 1.5000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  8.7760e+02,
        -1.9310e+00,  1.7160e+00, -1.0357e+00, -2.5400e-02,  1.8071e+00,
         1.8100e-02, -6.0000e-02,  1.1683e+00, -3.5000e-01,  9.0000e-01,
        -9.3000e-03,  6.3000e-01,  5.0900e-01,  6.1600e-01,  4.9100e-01,
         5.5000e-01,  3.2600e-01,  8.9600e-01],
       [ 2.0000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  7.4820e+02,
        -2.1880e+00,  1.6870e+00, -9.7000e-01, -3.9600e-02,  2.0773e+00,
         3.0900e-02, -6.0000e-02,  1.1274e+00, -3.5000e-01,  9.0000e-01,
        -8.3000e-03,  6.3000e-01,  5.1400e-01,  6.1400e-01,  4.9500e-01,
         5.2000e-01,  3.2900e-01,  8.7400e-01],
       [ 2.5000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  6.5430e+02,
        -2.3810e+00,  1.6460e+00, -9.2020e-01, -5.3900e-02,  2.2794e+00,
         4.0900e-02, -6.0000e-02,  1.0956e+00, -3.5000e-01,  9.0000e-01,
        -6.9000e-03,  6.3000e-01,  5.1800e-01,  6.1200e-01,  4.9700e-01,
         4.9700e-01,  3.3200e-01,  8.5600e-01],
       [ 3.0000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  5.8710e+02,
        -2.5180e+00,  1.6010e+00, -8.9740e-01, -6.5600e-02,  2.4201e+00,
         4.9100e-02, -6.0000e-02,  1.0697e+00, -3.5000e-01,  9.0000e-01,
        -5.7000e-03,  6.3000e-01,  5.2200e-01,  6.1100e-01,  4.9900e-01,
         4.7900e-01,  3.3500e-01,  8.4100e-01],
       [ 4.0000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  5.0300e+02,
        -2.6570e+00,  1.5110e+00, -8.6770e-01, -8.0700e-02,  2.5510e+00,
         6.1900e-02, -6.0000e-02,  1.0288e+00, -3.5000e-01,  8.4230e-01,
        -3.9000e-03,  6.3000e-01,  5.2700e-01,  6.0800e-01,  5.0100e-01,
         4.4900e-01,  3.3800e-01,  8.1800e-01],
       [ 5.0000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.5660e+02,
        -2.6690e+00,  1.3970e+00, -8.4750e-01, -9.2400e-02,  2.5395e+00,
         7.1900e-02, -6.0000e-02,  9.9710e-01, -3.1910e-01,  7.4580e-01,
        -2.5000e-03,  6.3000e-01,  5.3200e-01,  6.0600e-01,  5.0400e-01,
         4.2600e-01,  3.4100e-01,  7.8300e-01],
       [ 7.5000e-01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.1050e+02,
        -2.4010e+00,  1.1370e+00, -8.2060e-01, -1.1370e-01,  2.1493e+00,
         8.0000e-02, -6.0000e-02,  9.3950e-01, -2.6290e-01,  5.7040e-01,
         0.0000e+00,  6.3000e-01,  5.3900e-01,  6.0200e-01,  5.0600e-01,
         3.8500e-01,  3.4600e-01,  6.8000e-01],
       [ 1.0000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
        -1.9550e+00,  9.1500e-01, -8.0880e-01, -1.2890e-01,  1.5705e+00,
         8.0000e-02, -6.0000e-02,  8.9850e-01, -2.2300e-01,  4.4600e-01,
         0.0000e+00,  6.3000e-01,  5.4500e-01,  5.9400e-01,  5.0300e-01,
         3.5000e-01,  3.5000e-01,  6.0700e-01],
       [ 1.5000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
        -1.0250e+00,  5.1000e-01, -7.9950e-01, -1.5340e-01,  3.9910e-01,
         8.0000e-02, -6.0000e-02,  8.4090e-01, -1.6680e-01,  2.7070e-01,
         0.0000e+00,  6.1500e-01,  5.5200e-01,  5.6600e-01,  4.9700e-01,
         3.5000e-01,  3.5000e-01,  5.0400e-01],
       [ 2.0000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
        -2.9900e-01,  1.9200e-01, -7.9600e-01, -1.7080e-01, -6.0720e-01,
         8.0000e-02, -6.0000e-02,  8.0000e-01, -1.2700e-01,  1.4630e-01,
         0.0000e+00,  6.0400e-01,  5.5800e-01,  5.4400e-01,  4.9100e-01,
         3.5000e-01,  3.5000e-01,  4.3100e-01],
       [ 3.0000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
         0.0000e+00, -2.8000e-01, -7.9600e-01, -1.9540e-01, -9.6000e-01,
         8.0000e-02, -6.0000e-02,  4.7930e-01, -7.0800e-02, -2.9100e-02,
         0.0000e+00,  5.8900e-01,  5.6500e-01,  5.2700e-01,  5.0000e-01,
         3.5000e-01,  3.5000e-01,  3.2800e-01],
       [ 4.0000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
         0.0000e+00, -6.3900e-01, -7.9600e-01, -2.1280e-01, -9.6000e-01,
         8.0000e-02, -6.0000e-02,  2.5180e-01, -3.0900e-02, -1.5350e-01,
         0.0000e+00,  5.7800e-01,  5.7000e-01,  5.1500e-01,  5.0500e-01,
         3.5000e-01,  3.5000e-01,  2.5500e-01],
       [ 5.0000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
         0.0000e+00, -9.3600e-01, -7.9600e-01, -2.2630e-01, -9.2080e-01,
         8.0000e-02, -6.0000e-02,  7.5400e-02,  0.0000e+00, -2.5000e-01,
         0.0000e+00,  5.7000e-01,  5.8700e-01,  5.1000e-01,  5.2900e-01,
         3.5000e-01,  3.5000e-01,  2.0000e-01],
       [ 7.5000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
         0.0000e+00, -1.5270e+00, -7.9600e-01, -2.5090e-01, -7.7000e-01,
         8.0000e-02, -6.0000e-02,  0.0000e+00,  0.0000e+00, -2.5000e-01,
         0.0000e+00,  6.1100e-01,  6.1800e-01,  5.7200e-01,  5.7900e-01,
         3.5000e-01,  3.5000e-01,  2.0000e-01],
       [ 1.0000e+01,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
         0.0000e+00, -1.9930e+00, -7.9600e-01, -2.6830e-01, -6.6300e-01,
         8.0000e-02, -6.0000e-02,  0.0000e+00,  0.0000e+00, -2.5000e-01,
         0.0000e+00,  6.4000e-01,  6.4000e-01,  6.1200e-01,  6.1200e-01,
         3.5000e-01,  3.5000e-01,  2.0000e-01],
       [ 0.0000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  8.6510e+02,
        -1.1860e+00,  8.0400e-01, -9.6790e-01, -3.7200e-02,  9.4450e-01,
         0.0000e+00, -6.0000e-02,  1.0800e+00, -3.5000e-01,  9.0000e-01,
        -6.7000e-03,  5.9000e-01,  4.7000e-01,  5.7600e-01,  4.5300e-01,
         4.7000e-01,  3.0000e-01,  1.0000e+00],
       [-1.0000e+00,  6.7500e+00,  4.5000e+00,  2.6500e-01, -2.3100e-01,
        -3.9800e-01,  1.1800e+00,  1.8800e+00,  5.0000e+01,  4.0000e+02,
        -1.9550e+00,  5.7578e+00, -9.0460e-01, -1.2000e-01,  1.5390e+00,
         8.0000e-02, -6.0000e-02,  7.0000e-01, -3.9000e-01,  6.3000e-01,
         0.0000e+00,  5.9000e-01,  4.7000e-01,  5.7600e-01,  4.5300e-01,
         4.2000e-01,  3.0000e-01,  7.4000e-01]])

    def __call__(self, period, moment, Vs30, Rrup, Rjb, Rx, azimuth, Ztor, dip_angle, width, faultType, VsCondition, ShockType):
        # , moment, rjb, vs30, Ftype, rrup, rx, dip, ztor, z10, width,
        #azimuth, Fhw, afterShock, vsFlag):
        self.period = period
        self.moment = moment
        self.ztor = Ztor
        self.dip = dip_angle
        self.azimuth = azimuth
        self.width = width
        self.faultType = faultType
        self.vsFlag = VsCondition
        self.shocktype = ShockType


        self.shear_wave = np.array(Vs30).astype(np.float32)
        self.rrup = np.array(Rrup).astype(np.float32)
        self.rjb = np.array(Rjb).astype(np.float32)
        self.rx = np.array(Rx).astype(np.float32)
        self.azimuth = np.array(self.azimuth).astype(np.float32)



        ##################################################
        ### Definition Heteregenous System Computation ###
        ##### Implementation Used Defiend Variables ######
        ##################################################
        self.CTX = cl.create_some_context()
        self.QUE = cl.CommandQueue(self.CTX)
        self.shear_waveG = cl.array.to_device(self.QUE, self.shear_wave)
        self.rrupG = cl.array.to_device(self.QUE, self.rrup)
        self.rjbG = cl.array.to_device(self.QUE, self.rjb)
        self.rxG = cl.array.to_device(self.QUE, self.rx)
        self.azimuthG = cl.array.to_device(self.QUE, self.azimuth)

    def getCoefs(self, varNames, periodTerm=None):
        if periodTerm is None:
            periodTerm = self.period
        varIndex = ['Period', 'c1', 'c4', 'a3', 'a4', 'a5', 'n',
        'c', 'c2', 'VLIN', 'b', 'a1', 'a2', 'a8', 'a10', 'a12',
        'a13', 'a14', 'a15', 'a16', 'a18',
        'se1', 'se2', 'sm1', 'sm2', 's3', 's4', 'rh/t']

        varDict = {}
        for v in varNames:
            varDict[v] = self.coefAS[self.coefAS[:,0]==periodTerm][0][varIndex.index(v)]
        return varDict

    def baseModel(self, periodTerm=None): #AS08 Eqs. 2 and 3
        coefBM = self.getCoefs(['a1','a2','a3','a4','a5','a8','c1','c4'], periodTerm)
        a1,a2,a3,a4 = coefBM['a1'], coefBM['a2'], coefBM['a3'], coefBM['a4']
        a5,a8,c1,c4 = coefBM['a5'], coefBM['a8'], coefBM['c1'], coefBM['c4']

        bMod = cl.elementwise.ElementwiseKernel(self.CTX,
        "float m, float *r, float *res, float a1, float a2, float a3, float a4, float a5, float a8, float c1, float c4",
        """
            if (m <= c1) {
                res[i] = a1 + a4 * (m-c1) + a8 * pow(8.5-m,2) + (a2+a3*(m-c1)) * log(sqrt(pow(r[i],2)+pow(c4,2)));
            } else {
                res[i] = a1 + a5 * (m-c1) + a8 * pow(8.5-m,2) + (a2+a3*(m-c1)) * log(sqrt(pow(r[i],2)+pow(c4,2)));
            }
        """,
        "BM")

        baseModel = cl.array.empty_like(self.rrupG)
        bMod(self.moment, self.rrupG,baseModel,a1,a2,a3,a4,a5,a8,c1,c4)
        return baseModel.get()

    def faultStyle(self, periodTerm=None):

        if self.faultType == 'TH':
            Frv, Fnm = 1,0
        elif self.faultType == 'NR':
            Frv, Fnm = 0,1
        elif self.faultType == 'SS':
            Frv, Fnm = 0,0

        if self.shocktype == 'M':
            Fas = 0
        elif self.shocktype == 'A':
            Fas = 1

        return self.getCoefs(['a12'], periodTerm)['a12'] * Frv + self.getCoefs(['a13'], periodTerm)['a13'] * Fnm + self.getCoefs(['a15'], periodTerm)['a15'] * Fas

    def hangingModel(self, periodTerm=None): # AS08 Eqs. 7, 8, 9, 10, 11, 12
        # term rjb
        t1r= cl.array.empty_like(self.rjbG)
        t1 = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *rj, float *t1",
        """
            if (rj[i]<30) {
                t1[i] = 1 - rj[i]/30;
            } else {
                t1[i] = 0;
            }
        """,
        "t1")
        t1(self.rjbG, t1r)

        # term Rx W, S
        wcosSig = self.width * np.cos(np.radians(self.dip))
        t2r=cl.array.empty_like(self.rxG)
        t2 = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *rx, float wcos, float *t2, float dd",
        """
            if ((dd == 90) || (rx[i] > wcos)) {
                t2[i] = 1;
            } else {
                t2[i] = 0.5 + rx[i] / (2*wcos);
            }
        """,
        "t2")
        t2(self.rxG,wcosSig,t2r, self.dip)

        # term Rx Ztor
        t3 = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *t3, float *rx, float tor",
        """
            if (rx[i]>=tor) {
                t3[i]=1;
            } else {
                t3[i]=rx[i]/tor;
            }
        """,
        "t3")
        t3r = cl.array.empty_like(self.rxG)
        t3(t3r, self.rxG, self.ztor)
        # term moment
        if self.moment <= 6:
            t4=0
        elif 6 < self.moment < 7:
            t4=self.moment-6
        else:
            t4 = 1

        # term sigma from errata
        if self.dip >=30:
            t5 = 1 - (self.dip-30)/60
        else:
            t5=1

        # definition of hanging wall side by using Rx and azimuth terms
        fHW = cl.array.empty_like(self.azimuthG)
        hangForce = cl.elementwise.ElementwiseKernel(self.CTX, 
        " float *fhw, float *azi, float *rx",
        """
            if ((azi[i]< 0) || (rx[i]<0)) {
                fhw[i] = 0;
            } else {
                fhw[i] = 1;
            }
        """,
        "hangForce")
        hangForce(fHW, self.azimuthG, self.rxG)
        
        #print(fHW.get(),t1r.get(),t2r.get(),t3r.get(),t4,t5)
        return fHW.get()*self.getCoefs(['a14'],periodTerm)['a14']*t1r.get()*t2r.get()*t3r.get()*t4*t5

    def depthRuptureModel(self, periodTerm=None): # AS08 Eq. 13
        if self.ztor < 10:
            return self.getCoefs(['a16'], periodTerm)['a16'] * self.ztor / 10
        else:
            return self.getCoefs(['a16'], periodTerm)['a16']

    def largeDistanceModel(self,periodTerm=None): # AS08 Eqs. 14 and 15
        if self.moment < 5.5:
            t6=1
        elif 5.5 <= self.moment <= 6.5:
            t6=0.5*(6.5-self.moment)+0.5
        else:
            t6=0.5

        lDM = cl.elementwise.ElementwiseKernel(self.CTX,
        " float *dm, float *rr, float t6, float a18",
        """
            if (rr[i]<100) {
                dm[i]=0;
            } else {
                dm[i] =a18 * (rr[i]-100)* t6;
            }
        """,
        "lDM")
        rLDM = cl.array.empty_like(self.rrupG)
        lDM(rLDM, self.rrupG, t6, self.getCoefs(['a18'],periodTerm)['a18'])
        return rLDM.get()

    def soilDepthModel(self, vsTerm=None, periodTerm=None): # AS08 Eqs.16,17,18,19, and 20.
        if vsTerm == None:
            vsTerm = self.shear_wave
        if periodTerm == None:
            periodTerm = self.period

        vsTerm2 = np.array(vsTerm).astype(np.float32)
        z1 = np.exp(28.5-0.4775*np.log(vsTerm2**8+378.7**8)) # NGA-W1
        #print('z1 :', z1 )

        vsTerm2 = cl.array.to_device(self.QUE, vsTerm2)

        #############
        # AS08 Eq. 20
        #############


        if periodTerm < 2:
            a22 = 0
        else:
            a22 = 0.0625*(periodTerm-2)

        #############
        # AS08 Eq. 19
        #############
        e2 = cl.array.empty_like(vsTerm2)
        e2H = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *rE, float *vs, float prd",
        """
            if ((prd < 0.35) || (vs[i]>1000)) {
                rE[i]=0;
            } else if ((0.35 <= prd <=2) || (prd = -1)) {
                rE[i]=-0.25*log(vs[i]/1000)*log(prd/0.35);
            } else if (prd > 2) {
                rE[i]=-0.25*log(vs[i]/1000)*1.742969;
            }
        """,
        "e2H")

        e2H(e2, vsTerm2, periodTerm)



        # Calculation of the Z1hat and Z1
        z1H = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *zh, float *vs",
        """
            if (vs[i]<180) {
                zh[i] = exp(6.745);
            } else if (180<=vs[i]<=500) {
                zh[i] = exp(6.745 - 1.35*log(vs[i]/180));
            } else {
                zh[i] = exp(5.394 - 4.48*log(vs[i]/500));
            }
        """,
        "z1H")
        z1hat = cl.array.empty_like(vsTerm2)
        z1H(z1hat,vsTerm2)
        #print('z1hat :', z1hat.get() )





        #############
        # AS08 Eq. 18
        #############
        v1, vs30ss = self.get_v1vs30star(vsTerm2, periodTerm)
        a21 = cl.array.empty_like(vsTerm2)
        a21H = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *ra21, float *vs, float *vss, float *e2, float *z1, float *z1h, float a10, float bc, float nc, float v1, float c2",
        """

            if (vs[i] >= 1000) {
                ra21[i] = 0;
            } else if ( (a10+bc*nc)*log(vss[i]/v1) + e2[i]*log((z1[i]+c2)/(z1h[i]+c2)) < 0) {
                ra21[i] = -1*(a10+bc*nc)*log(vss[i]/v1);
            } else {
                ra21[i]= e2[i];
            }
        """,
        "a21H")
        v1m = min(v1,1000)
        a10,b,n,c2= self.getCoefs(['a10'],periodTerm)['a10'], self.getCoefs(['b'],periodTerm)['b'],self.getCoefs(['n'],periodTerm)['n'], self.getCoefs(['c2'],periodTerm)['c2']
        e2s = cl.array.to_device(self.QUE,e2.get())
        z1hs = cl.array.to_device(self.QUE,z1hat.get())
        z1 =  cl.array.to_device(self.QUE,z1) ### z1 assinged zhat
        vs30ss = cl.array.to_device(self.QUE, vs30ss)

        a21H(a21, vsTerm2, vs30ss, e2s, z1, z1hs, a10, b, n, v1m, c2)

        # AS08 Eq. 17

        sDM = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *sdm, float *z1, float *zh, float *a21, float a22, float c2",
        """
            if (z1[i] >= 200) {
                sdm[i]= a21[i] * log((z1[i]+c2)/(zh[i]+c2)) + a22*log(z1[i]/200);
            } else {
                sdm[i]= a21[i] * log((z1[i]+c2)/(zh[i]+c2));
            }
        """,
        "sDM")
        rSDM = cl.array.empty_like(z1)
        c2 = self.getCoefs(['c2'],periodTerm)['c2']
        a21 = cl.array.to_device(self.QUE, a21.get())
        sDM(rSDM, z1, z1hs,a21,a22,c2) # z1
        return rSDM.get()

    def get_v1vs30star(self, vsTerm=None, periodTerm=None): # AS08 Eqs. 5 and 6

        if periodTerm is None:
            periodTerm = self.period

        if type(vsTerm) is not cl.array.Array:
            if vsTerm == None:
                vsTerm = self.shear_wave
            vsTerm1 = np.array(vsTerm).astype(np.float32)
            vsTerm1 = cl.array.to_device(self.QUE, vsTerm1)
        else:
            vsTerm1 = vsTerm



        if periodTerm == -1:
            v1 = 862
        elif 0 <= periodTerm <=0.5:
            v1 = 1500
        elif 0.5 < periodTerm <=1:
            v1 =np.exp(8.0-0.795*np.log(periodTerm/0.21))
        elif 1 < periodTerm < 2:
            v1 = np.exp(6.76-0.297*np.log(periodTerm))
        elif periodTerm >=2:
            v1 = 700

        vs30Star = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *a, float *b, float c",
        "b[i]=min(a[i],c)",
        "vs30star")
        vs30s = cl.array.empty_like(vsTerm1)
        vs30Star(vsTerm1, vs30s, v1)
        return v1, vs30s.get()

    def siteResponse(self, pga1100, vsTerm=None, periodTerm=None): # AS08 Eq. 4
        if vsTerm is None:
            vsTerm = self.shear_wave
        vsTerm = np.array(vsTerm).astype(np.float32)
        vsTerm = cl.array.to_device(self.QUE, vsTerm)
        coefSR = self.getCoefs(['VLIN','c','b','n','a10'], periodTerm)
        pga1100 = np.array(pga1100).astype(np.float32)
        pga1100 = cl.array.to_device(self.QUE, pga1100)

        v1, vss = self.get_v1vs30star(vsTerm, periodTerm)
        vss = cl.array.to_device(self.QUE, vss)

        sResp = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *res1, float *pg, float *vs, float *vs2, float bm, float cm, float nm, float a10, float vlin",
        """
            if (vs[i] < vlin) {
                res1[i] = a10*log(vs2[i]/vlin) -1 * bm*log(pg[i]+cm) + bm*log(pg[i]+cm*pow(vs2[i]/vlin,nm));
            } else {
                res1[i] = (a10+bm*nm)*log(vs2[i]/vlin);
            }
        """,
        "sResp")
        Resp = cl.array.empty_like(vsTerm)
        sResp(Resp, pga1100, vsTerm, vss, coefSR['b'],coefSR['c'],coefSR['n'],coefSR['a10'],coefSR['VLIN'])

        return Resp.get()

    def get_PGA1100(self):
        PGA1100 = self.baseModel(0)+self.faultStyle(0)+\
        self.hangingModel(0)+self.depthRuptureModel(0)+\
        self.largeDistanceModel(0)+self.soilDepthModel(1100,0)+\
        self.siteResponse(0,1100,0)
        return np.exp(PGA1100)

    def logline(self,x1,x2,y1,y2,x):
        k = (y2-y1)/(x2-x1)
        C = y1-k*x1
        y = k*x+C
        return y

    def get_SA(self): # AS08 Eqs. 21 and 22
        pga=self.get_PGA1100()
        #############
        # AS08 Eq. 21
        #############
        Td = min(10**(-1.25 + 0.3 * self.moment),10)
        
        #############
        # AS08 Eq. 22
        #############
        if self.period <= Td:
            LnSa = self.baseModel()+self.faultStyle()+\
            self.hangingModel()+self.depthRuptureModel()+\
            self.largeDistanceModel()+self.siteResponse(pga)+\
            self.soilDepthModel()
        else:
            periods = np.array([ -1, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0] )

            Td1 = max((periods<Td).nonzero()[0])
            Td2 = min((periods<Td).nonzero()[0])

            LnSa1=self.baseModel(Td1)+self.faultStyle(Td1)+\
            self.hangingModel()+self.depthRuptureModel(Td1)+\
            self.largeDistanceModel(Td1)+self.siteResponse(pga,1100,Td1)+\
            self.soilDepthModel(1100,Td1)

            LnSa2=self.baseModel(Td2)+self.faultStyle(Td2)+\
            self.hangingModel()+self.depthRuptureModel(Td2)+\
            self.largeDistanceModel(Td2)+self.siteResponse(pga,1100,Td2)+\
            self.soilDepthModel(1100,Td2)

            LnSaRockTd = self.logline(np.log(Td1), np.log(Td2), LnSa1, LnSa2, np.log(Td))
            LnSaRockT = LnSaRockTd + np.log((Td/self.period)**2)
            siteSoil = self.siteResponse(pga) + self.soilDepthModel()
            siteRock = self.siteResponse(pga,1100) + self.soilDepthModel(1100)
            soilAmp = siteSoil-siteRock

            LnSa = LnSaRockT + soilAmp
        return np.exp(LnSa)

    def get_alpha(self): # AS08 Eq. 26 from errata
        coefAL = self.getCoefs(['VLIN','b','c','n'])
        
        pga = self.get_PGA1100()
        pga1 = cl.array.to_device(self.QUE, np.array(pga).astype(np.float32))
        
        #alpha[i] = -b_cons*pg[i]/(pg[i]+c_cons) + b_cons*pg[i]/(pg[i]+c_cons*pow(vs[i]/vl,n_cons));


        alpG = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *alpha, float *vs, float *pg, float b_cons, float c_cons, float n_cons, float vl",
        """
            if (vs[i] < vl) {
                alpha[i] = -b_cons*pg[i]/(pg[i]+c_cons) + b_cons*pg[i]/(pg[i]+c_cons*pow(vs[i]/vl,n_cons));
            } else {
                alpha[i] = 0;
            }
        """,
        "alpG"
        )
        alpha = cl.array.empty_like(self.shear_waveG)
        
        alpG(alpha, self.shear_waveG, pga1, coefAL['b'], coefAL['c'], coefAL['n'], coefAL['VLIN'])
        
        print(alpha.get())
        return alpha.get()
    
    def get_sigma0Tau0(self, periodTerm=None): # AS08 Eqs. 27 and 28
        if self.vsFlag == 'M':
            s1, s2 = self.getCoefs(['sm1'],periodTerm)['sm1'],self.getCoefs(['sm2'],periodTerm)['sm2']
        else:
            s1, s2 = self.getCoefs(['se1'],periodTerm)['se1'],self.getCoefs(['se2'],periodTerm)['se2']
        s3, s4 = self.getCoefs(['s3'],periodTerm)['s3'],self.getCoefs(['s4'],periodTerm)['s4']

        if self.moment < 5:
            sigma0 = s1
            tau0 = s3
        elif 5 <= self.moment <= 7:
            sigma0 = s1 + ((s2-s1)/2) * (self.moment-5)
            tau0 = s3 + ((s4-s3)/2) * (self.moment-5)
        else:
            sigma0 = s2
            tau0 = s4
        return sigma0, tau0
    
    def get_sigmaTau(self): # AS08 Eqs. 24 and 25
        sigmaAmp = 0.3
        
        sigma0, tau0 = self.get_sigma0Tau0() # at given Period
        sigmaBT = np.sqrt(sigma0**2 - sigmaAmp**2)
        tauBT = tau0

        # FOR PGA
        sigma0, tau0 = self.get_sigma0Tau0(0)
        sigmaBpga = np.sqrt(sigma0**2 - sigmaAmp**2)
        tauBpga = tau0

        alpha = self.get_alpha()
        #print(alpha,sigmaBT, tauBT, sigmaBpga, tauBpga)
        # AS08 incorrect check the PEER report errata
        rho = self.getCoefs(['rh/t'])['rh/t']
        sigma = np.sqrt(sigmaBT**2 + sigmaAmp**2 + (alpha*sigmaBpga)**2 + \
            2*alpha*sigmaBT*sigmaBpga*rho)
        
        tau = np.sqrt(tauBT**2 +(alpha*tauBpga)**2 + \
            2*alpha*tauBT*tauBpga*rho)
        sigmaTot = np.sqrt(sigma**2 + tau**2)
        return [sigmaTot, sigma, tau]