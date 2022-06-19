from __future__ import absolute_import
from __future__ import print_function
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import pyopencl.array

class CB08():
    def __init__(self):
        self.coefCB = np.array([[ 1.0000e-02, -1.7150e+00,  5.0000e-01, -5.3000e-01, -2.6200e-01,
        -2.1180e+00,  1.7000e-01,  5.6000e+00,  2.8000e-01, -1.2000e-01,
         4.9000e-01,  1.0580e+00,  4.0000e-02,  6.1000e-01,  8.6500e+02,
        -1.1860e+00,  1.8390e+00,  1.8800e+00,  1.1800e+00,  4.7800e-01,
         2.1900e-01,  3.0000e-01,  1.6600e-01,  1.0000e+00],
       [ 2.0000e-02, -1.6800e+00,  5.0000e-01, -5.3000e-01, -2.6200e-01,
        -2.1230e+00,  1.7000e-01,  5.6000e+00,  2.8000e-01, -1.2000e-01,
         4.9000e-01,  1.1020e+00,  4.0000e-02,  6.1000e-01,  8.6500e+02,
        -1.2190e+00,  1.8400e+00,  1.8800e+00,  1.1800e+00,  4.8000e-01,
         2.1900e-01,  3.0000e-01,  1.6600e-01,  9.9900e-01],
       [ 3.0000e-02, -1.5520e+00,  5.0000e-01, -5.3000e-01, -2.6200e-01,
        -2.1450e+00,  1.7000e-01,  5.6000e+00,  2.8000e-01, -1.2000e-01,
         4.9000e-01,  1.1740e+00,  4.0000e-02,  6.1000e-01,  9.0800e+02,
        -1.2730e+00,  1.8410e+00,  1.8800e+00,  1.1800e+00,  4.8900e-01,
         2.3500e-01,  3.0000e-01,  1.6500e-01,  9.8900e-01],
       [ 5.0000e-02, -1.2090e+00,  5.0000e-01, -5.3000e-01, -2.6700e-01,
        -2.1990e+00,  1.7000e-01,  5.7400e+00,  2.8000e-01, -1.2000e-01,
         4.9000e-01,  1.2720e+00,  4.0000e-02,  6.1000e-01,  1.0540e+03,
        -1.3460e+00,  1.8430e+00,  1.8800e+00,  1.1800e+00,  5.1000e-01,
         2.5800e-01,  3.0000e-01,  1.6200e-01,  9.6300e-01],
       [ 7.5000e-02, -6.5700e-01,  5.0000e-01, -5.3000e-01, -3.0200e-01,
        -2.2770e+00,  1.7000e-01,  7.0900e+00,  2.8000e-01, -1.2000e-01,
         4.9000e-01,  1.4380e+00,  4.0000e-02,  6.1000e-01,  1.0860e+03,
        -1.4710e+00,  1.8450e+00,  1.8800e+00,  1.1800e+00,  5.2000e-01,
         2.9200e-01,  3.0000e-01,  1.5800e-01,  9.2200e-01],
       [ 1.0000e-01, -3.1400e-01,  5.0000e-01, -5.3000e-01, -3.2400e-01,
        -2.3180e+00,  1.7000e-01,  8.0500e+00,  2.8000e-01, -9.9000e-02,
         4.9000e-01,  1.6040e+00,  4.0000e-02,  6.1000e-01,  1.0320e+03,
        -1.6240e+00,  1.8470e+00,  1.8800e+00,  1.1800e+00,  5.3100e-01,
         2.8600e-01,  3.0000e-01,  1.7000e-01,  8.9800e-01],
       [ 1.5000e-01, -1.3300e-01,  5.0000e-01, -5.3000e-01, -3.3900e-01,
        -2.3090e+00,  1.7000e-01,  8.7900e+00,  2.8000e-01, -4.8000e-02,
         4.9000e-01,  1.9280e+00,  4.0000e-02,  6.1000e-01,  8.7800e+02,
        -1.9310e+00,  1.8520e+00,  1.8800e+00,  1.1800e+00,  5.3200e-01,
         2.8000e-01,  3.0000e-01,  1.8000e-01,  8.9000e-01],
       [ 2.0000e-01, -4.8600e-01,  5.0000e-01, -4.4600e-01, -3.9800e-01,
        -2.2200e+00,  1.7000e-01,  7.6000e+00,  2.8000e-01, -1.2000e-02,
         4.9000e-01,  2.1940e+00,  4.0000e-02,  6.1000e-01,  7.4800e+02,
        -2.1880e+00,  1.8560e+00,  1.8800e+00,  1.1800e+00,  5.3400e-01,
         2.4900e-01,  3.0000e-01,  1.8600e-01,  8.7100e-01],
       [ 2.5000e-01, -8.9000e-01,  5.0000e-01, -3.6200e-01, -4.5800e-01,
        -2.1460e+00,  1.7000e-01,  6.5800e+00,  2.8000e-01,  0.0000e+00,
         4.9000e-01,  2.3510e+00,  4.0000e-02,  7.0000e-01,  6.5400e+02,
        -2.3810e+00,  1.8610e+00,  1.8800e+00,  1.1800e+00,  5.3400e-01,
         2.4000e-01,  3.0000e-01,  1.9100e-01,  8.5200e-01],
       [ 3.0000e-01, -1.1710e+00,  5.0000e-01, -2.9400e-01, -5.1100e-01,
        -2.0950e+00,  1.7000e-01,  6.0400e+00,  2.8000e-01,  0.0000e+00,
         4.9000e-01,  2.4600e+00,  4.0000e-02,  7.5000e-01,  5.8700e+02,
        -2.5180e+00,  1.8650e+00,  1.8800e+00,  1.1800e+00,  5.4400e-01,
         2.1500e-01,  3.0000e-01,  1.9800e-01,  8.3100e-01],
       [ 4.0000e-01, -1.4660e+00,  5.0000e-01, -1.8600e-01, -5.9200e-01,
        -2.0660e+00,  1.7000e-01,  5.3000e+00,  2.8000e-01,  0.0000e+00,
         4.9000e-01,  2.5870e+00,  4.0000e-02,  8.5000e-01,  5.0300e+02,
        -2.6570e+00,  1.8740e+00,  1.8800e+00,  1.1800e+00,  5.4100e-01,
         2.1700e-01,  3.0000e-01,  2.0600e-01,  7.8500e-01],
       [ 5.0000e-01, -2.5690e+00,  6.5600e-01, -3.0400e-01, -5.3600e-01,
        -2.0410e+00,  1.7000e-01,  4.7300e+00,  2.8000e-01,  0.0000e+00,
         4.9000e-01,  2.5440e+00,  4.0000e-02,  8.8300e-01,  4.5700e+02,
        -2.6690e+00,  1.8830e+00,  1.8800e+00,  1.1800e+00,  5.5000e-01,
         2.1400e-01,  3.0000e-01,  2.0800e-01,  7.3500e-01],
       [ 7.5000e-01, -4.8440e+00,  9.7200e-01, -5.7800e-01, -4.0600e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  2.8000e-01,  0.0000e+00,
         4.9000e-01,  2.1330e+00,  7.7000e-02,  1.0000e+00,  4.1000e+02,
        -2.4010e+00,  1.9060e+00,  1.8800e+00,  1.1800e+00,  5.6800e-01,
         2.2700e-01,  3.0000e-01,  2.2100e-01,  6.2800e-01],
       [ 1.0000e+00, -6.4060e+00,  1.1960e+00, -7.7200e-01, -3.1400e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  2.5500e-01,  0.0000e+00,
         4.9000e-01,  1.5710e+00,  1.5000e-01,  1.0000e+00,  4.0000e+02,
        -1.9550e+00,  1.9290e+00,  1.8800e+00,  1.1800e+00,  5.6800e-01,
         2.5500e-01,  3.0000e-01,  2.2500e-01,  5.3400e-01],
       [ 1.5000e+00, -8.6920e+00,  1.5130e+00, -1.0460e+00, -1.8500e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  1.6100e-01,  0.0000e+00,
         4.9000e-01,  4.0600e-01,  2.5300e-01,  1.0000e+00,  4.0000e+02,
        -1.0250e+00,  1.9740e+00,  1.8800e+00,  1.1800e+00,  5.6400e-01,
         2.9600e-01,  3.0000e-01,  2.2200e-01,  4.1100e-01],
       [ 2.0000e+00, -9.7010e+00,  1.6000e+00, -9.7800e-01, -2.3600e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  9.4000e-02,  0.0000e+00,
         3.7100e-01, -4.5600e-01,  3.0000e-01,  1.0000e+00,  4.0000e+02,
        -2.9900e-01,  2.0190e+00,  1.8800e+00,  1.1800e+00,  5.7100e-01,
         2.9600e-01,  3.0000e-01,  2.2600e-01,  3.3100e-01],
       [ 3.0000e+00, -1.0556e+01,  1.6000e+00, -6.3800e-01, -4.9100e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  0.0000e+00,  0.0000e+00,
         1.5400e-01, -8.2000e-01,  3.0000e-01,  1.0000e+00,  4.0000e+02,
         0.0000e+00,  2.1100e+00,  1.8800e+00,  1.1800e+00,  5.5800e-01,
         3.2600e-01,  3.0000e-01,  2.2900e-01,  2.8900e-01],
       [ 4.0000e+00, -1.1212e+01,  1.6000e+00, -3.1600e-01, -7.7000e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00, -8.2000e-01,  3.0000e-01,  1.0000e+00,  4.0000e+02,
         0.0000e+00,  2.2000e+00,  1.8800e+00,  1.1800e+00,  5.7600e-01,
         2.9700e-01,  3.0000e-01,  2.3700e-01,  2.6100e-01],
       [ 5.0000e+00, -1.1684e+01,  1.6000e+00, -7.0000e-02, -9.8600e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00, -8.2000e-01,  3.0000e-01,  1.0000e+00,  4.0000e+02,
         0.0000e+00,  2.2910e+00,  1.8800e+00,  1.1800e+00,  6.0100e-01,
         3.5900e-01,  3.0000e-01,  2.3700e-01,  2.0000e-01],
       [ 7.5000e+00, -1.2505e+01,  1.6000e+00, -7.0000e-02, -6.5600e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00, -8.2000e-01,  3.0000e-01,  1.0000e+00,  4.0000e+02,
         0.0000e+00,  2.5170e+00,  1.8800e+00,  1.1800e+00,  6.2800e-01,
         4.2800e-01,  3.0000e-01,  2.7100e-01,  1.7400e-01],
       [ 1.0000e+01, -1.3087e+01,  1.6000e+00, -7.0000e-02, -4.2200e-01,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00, -8.2000e-01,  3.0000e-01,  1.0000e+00,  4.0000e+02,
         0.0000e+00,  2.7440e+00,  1.8800e+00,  1.1800e+00,  6.6700e-01,
         4.8500e-01,  3.0000e-01,  2.9000e-01,  1.7400e-01],
       [ 0.0000e+00, -1.7150e+00,  5.0000e-01, -5.3000e-01, -2.6200e-01,
        -2.1180e+00,  1.7000e-01,  5.6000e+00,  2.8000e-01, -1.2000e-01,
         4.9000e-01,  1.0580e+00,  4.0000e-02,  6.1000e-01,  8.6500e+02,
        -1.1860e+00,  1.8390e+00,  1.8800e+00,  1.1800e+00,  4.7800e-01,
         2.1900e-01,  3.0000e-01,  1.6600e-01,  1.0000e+00],
       [-1.0000e+00,  9.5400e-01,  6.9600e-01, -3.0900e-01, -1.9000e-02,
        -2.0160e+00,  1.7000e-01,  4.0000e+00,  2.4500e-01,  0.0000e+00,
         3.5800e-01,  1.6940e+00,  9.2000e-02,  1.0000e+00,  4.0000e+02,
        -1.9550e+00,  1.9290e+00,  1.8800e+00,  1.1800e+00,  4.8400e-01,
         2.0300e-01,  3.0000e-01,  1.9000e-01,  6.9100e-01],
       [-2.0000e+00, -5.2700e+00,  1.6000e+00, -7.0000e-02,  0.0000e+00,
        -2.0000e+00,  1.7000e-01,  4.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00, -8.2000e-01,  3.0000e-01,  1.0000e+00,  4.0000e+02,
         0.0000e+00,  2.7440e+00,  1.8800e+00,  1.1800e+00,  6.6700e-01,
         4.8500e-01,  3.0000e-01,  2.9000e-01,  1.7400e-01]])

    def __call__(self, period, moment, Vs30, Rrup, Rjb, Rx, azimuth, Ztor, dipAngle, FaultType):
        self.period = period
        self.moment = moment
        self.ztor = Ztor
        self.faultType = FaultType
        self.dip = dipAngle

        self.rrup = np.array(Rrup).astype(np.float32)
        self.rjb = np.array(Rjb).astype(np.float32)
        self.rx = np.array(Rx).astype(np.float32)
        self.azi = np.array(azimuth).astype(np.float32)
        self.vs30 = np.array(Vs30).astype(np.float32)


        #PyOpenCL arrays
        self.CTX = cl.create_some_context()
        self.QUE = cl.CommandQueue(self.CTX)

        self.rrupG = cl.array.to_device(self.QUE, self.rrup)
        self.rjbG = cl.array.to_device(self.QUE, self.rjb)
        self.rxG = cl.array.to_device(self.QUE, self.rx)
        self.aziG = cl.array.to_device(self.QUE, self.azi)
        self.vs30G = cl.array.to_device(self.QUE, self.vs30)


    def get_coefs(self, varNames, periodTerm=None):
        if periodTerm is None:
            periodTerm = self.period
        varIndex = ['Period', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'k1', 'k2', 'k3', 'c', 'n', 'slnY', 'tlnY', 'slnAF', 'sC', 'r']

        varDict = {}
        for v in varNames:
            varDict[v] = self.coefCB[self.coefCB[:,0]==periodTerm][0][varIndex.index(v)]
        return varDict

    # lnY = Fmag + Fdis + Fflt + Fhng + Fsite + Fsed

    def momentModel(self,periodTerm=None): # CB08 Eq. 2
        coefMM = self.get_coefs(['c0','c1','c2','c3'],periodTerm)

        if self.moment <= 5.5:
            return coefMM['c0'] + coefMM['c1']*self.moment
        elif (5.5 < self.moment <= 6.5):
            return coefMM['c0'] + coefMM['c1']*self.moment + coefMM['c2']*(self.moment-5.5)
        else:
            return coefMM['c0'] + coefMM['c1']*self.moment + coefMM['c2']*(self.moment-5.5) + coefMM['c3']*(self.moment-6.5)

    def distanceModel(self,periodTerm=None): # CB08 Eq. 3
        coefDM = self.get_coefs(['c4','c5','c6'],periodTerm)

        return (coefDM['c4']+coefDM['c5']*self.moment)*np.log(np.sqrt(self.rrup**2 + coefDM['c6']**2))

    def faultingStyle(self,periodTerm=None): # CB08 Eqs. 4 and 5
        coefFS = self.get_coefs(['c7','c8'],periodTerm)
        if self.ztor < 1:
            flt = self.ztor
        else:
            flt = 1
        if self.faultType == 'TH':
            Frv,Fnm = 1,0
        elif self.faultType == 'NR':
            Frv,Fnm = 0,1
        else:
            Frv,Fnm = 0,0
        return coefFS['c7']*Frv*flt + coefFS['c8']*Fnm

    def hangingWall(self, periodTerm=None):
        # Eq. 10
        if self.dip <= 70:
            termD = 1
        else:
            termD = (90-self.dip)/20
        # Eq. 9
        if self.ztor >= 20:
            termZ = 0
        else:
            termZ = (20-self.ztor)/20
        # Eq. 8
        if self.moment <= 6:
            termM = 0
        elif (6 < self.moment) and (self.moment < 6.5):
            termM = 2*(self.moment-6)
        else:
            termM = 1

        # Eq. 7

        termR = cl.array.empty_like(self.rjbG)
        getR = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *res, float *rj, float *rr, float z",
        """
            if (rj[i] == 0) {
                res[i]=1;
            } else if (z < 1) {
                res[i] = (max(rr[i],sqrt(pow(rj[i],2)+1))-rj[i]) / max(rr[i],sqrt(pow(rj[i],2)+1));
            } else {
                res[i] = (rr[i]-rj[i])/rr[i];
            }

        """,
        "getR")
        getR(termR, self.rjbG, self.rrupG,self.ztor)

        fHW = cl.array.empty_like(self.rxG)

        HWg=cl.elementwise.ElementwiseKernel(self.CTX,
        "float *hw, float *rx, float *azi",
        """
            if ((rx[i]<0) || (azi[i]<0)) {
                hw[i]=0;
            } else {
                hw[i]=1;
            }
        """,
        "HWg")
        HWg(fHW, self.rxG, self.aziG)


        return self.get_coefs(['c9'],periodTerm)['c9'] *termD*termZ*termM*termR.get()*fHW.get()


    def siteResponse(self,A1100=None,vsREF=None,periodTerm=None):
        if A1100 is None:
            A1100 = self.get_A1100()
        
        A1100 = np.array(A1100).astype(np.float32)
        A1100g = cl.array.to_device(self.QUE, A1100)

        if vsREF is None:
            vsREF1 = self.vs30
        else:
            vsREF1 = np.array(vsREF).astype(np.float32)
        vsREFg = cl.array.to_device(self.QUE, vsREF1)
            
        coefSR = self.get_coefs(['c10','k1','k2','c','n'],periodTerm)
        
        fSite = cl.array.empty_like(vsREFg)
        getSite = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *res, float *vs, float *a11, float c10, float k1, float k2, float cn, float nn",
        """
            if (vs[i] < k1) {
                res[i] = c10 * log(vs[i]/k1) + k2*(log(a11[i]+cn*pow(vs[i]/k1,nn)) - log(a11[i]+cn));
            } else if ((k1 <= vs[i]) && (vs[i] < 1100)) {
                res[i] = (c10+k2*nn)*log(vs[i]/k1);
            } else {
                res[i] = (c10+k2*nn)*log(1100/k1);
            }
        """,
        "getSite")
        getSite(fSite, vsREFg, A1100g, coefSR['c10'], coefSR['k1'], coefSR['k2'], coefSR['c'], coefSR['n'])
        return fSite.get()

    def basinResponse(self,vsREF=None,periodTerm=None):
        if vsREF is None:
            vsREF2 = self.vs30
        else:
            vsREF2 = np.array(vsREF).astype(np.float32)
        z1 = np.exp(28.5-0.4775*np.log(vsREF2**8 + 378.7**8))
        z25 = (519 + 3.595 * z1)/1000
        
        print('z1  is :',z1)
        print('z25 is :',z25)
        

        z25 = np.array(z25).astype(np.float32)
        z25G = cl.array.to_device(self.QUE, z25)

        coefBR = self.get_coefs(['c11','c12','k3'],periodTerm)
        fSED = cl.array.empty_like(z25G)
        getfSED = cl.elementwise.ElementwiseKernel(self.CTX,
        "float *sed, float *z25, float c11, float c12, float k3",
        """
            if (z25[i]<1) {
                sed[i] = c11 * (z25[i]-1);
            } else if ( (1 <= z25[i]) && (z25[i]<=3) ){
                sed[i] = 0;
            } else {
                sed[i] = c12*k3*exp(-0.75) * (1-exp(-0.25*(z25[i]-3)));
            }
        """,
        "getfSED")
        getfSED(fSED, z25G, coefBR['c11'], coefBR['c12'], coefBR['k3'])
        return fSED.get()

    def get_A1100(self):
        A1100 = \
        self.momentModel(0) +\
            self.distanceModel(0) +\
                self.faultingStyle(0)+\
                    self.faultingStyle(0)+\
                        self.hangingWall(0)+\
                            self.siteResponse(0,1100,0)
        return np.exp(A1100)
    
    def get_SA(self):
        SA = \
            self.momentModel() +\
                self.distanceModel() +\
                    self.faultingStyle()+\
                        self.hangingWall() +\
                            self.siteResponse() +\
                                self.basinResponse()
        SA = np.exp(SA)
        if self.period <= 0.25:
            SA1 = \
            self.momentModel(periodTerm=0) +\
                self.distanceModel(periodTerm=0) +\
                    self.hangingWall(periodTerm=0) +\
                        self.faultingStyle(periodTerm=0)+\
                            self.siteResponse(periodTerm=0) +\
                                self.basinResponse(periodTerm=0)
            SA1 = np.exp(SA1)

            if SA < SA1:
                SA = SA1
        return SA

    def calc_alpha(self, vsTerm=None, periodTerm=None):
        if vsTerm is None:
            vsTerm = self.vs30
        
        coefA = self.get_coefs(['k1','k2','c','n'])

        A1100 = self.get_A1100()
        A1100g = cl.array.to_device(self.QUE, np.array(A1100).astype(np.float32))
        vsTermg = cl.array.to_device(self.QUE, np.array(vsTerm).astype(np.float32))

        alpha = cl.array.empty_like(vsTermg)
        getAlpha=cl.array.elementwise.ElementwiseKernel(self.CTX,
        "float *alp, float *vs, float *a11, float k1, float k2, float cn, float nn",
        """
            if (vs[i]<k1) {
                alp[i] = k2 * a11[i] * (1/(a11[i]+cn*pow((vs[i]/k1),nn))-1/(a11[i]+cn));
            } else {
                alp[i]=0;
            }
        """,
        "getAlpha")
        getAlpha(alpha, vsTermg, A1100g, coefA['k1'], coefA['k2'], coefA['c'], coefA['n'])
        return alpha.get()

    def sigma_calc(self, vsTerm=None, periodTerm=None):
        coefSC = self.get_coefs(['slnY','r'])
        slnAF  = 0.3
        slnYb   = np.sqrt(coefSC['slnY']**2-slnAF**2)
        slnAb  = np.sqrt(self.get_coefs(['slnY'],0)['slnY']**2-slnAF**2)
        alpha  = self.calc_alpha()
        sigma  = np.sqrt(slnYb**2 + slnAF**2 + alpha**2 * slnAb**2 + 2* alpha * coefSC['r']*slnYb*slnAb)
        return sigma
    
    def get_SD(self):
        tau = self.get_coefs(['tlnY'])['tlnY']
        sc = self.get_coefs(['sC'])['sC']
        sigma = self.sigma_calc()
        sigmaT = np.sqrt(sigma**2 + tau**2)
        sigmaArb = np.sqrt(sigmaT**2 + sc**2)
        return sigma, tau, sigmaT, sigmaArb