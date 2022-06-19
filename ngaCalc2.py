

def convFault(fpoints,inepsg,outepsg):
    from pyproj import transform, Proj
    inepsg = Proj('epsg:'+str(inepsg))
    outepsg = Proj('epsg:'+str(outepsg))
    xs,ys = transform(inepsg,outepsg, fpoints[:,0], fpoints[:,1])
    return xs,ys


def calcSiteParams(siteP, faultP):
    import numpy as np
    from numba import jit


    #siteP and faultP are numpy arrays in form x,y,z triplets, 
    #function returns Rrup,Rjb Rx in kms and alpha in degree
    #both Rx and alpha is return positive for hanging wall side and negative for the
    #footwall side. Rx in kms and alpha is in degrees.
    
    # Variables for the calculation of Rx and Alpha
    fTops = faultP[faultP[:,2] == np.max(faultP[:,2])] # Top edge of the Fault
    abc = np.polyfit(fTops[:,0],fTops[:,1],1) # constants for calculation in cartesian system
    a1,b1,c1 = abc[0], -1, abc[1] # in form of Ax+By+C=0 line equation in cartesian system
    funk = np.poly1d(abc) # polynominal form of the line equation in form P(x)=Ax+C <=> By=Ax+C for top fault trace
    x1,x2 = np.min(siteP[:,0])-10000, np.max(siteP[:,0])+10000 # selecting two points on the line and outside of the Site points
    y1,y2 = funk([x1,x2]) # purpose of this selection is determining the site point in which side of the fault (hang or foot) 


    @jit(nopython=True)
    def getVals(siteP,faultP):
        rx, alpha, side = [],[],[]
        rrup,rjb=[],[]

        p1,p2,p3=0,0,0

        for x,y,z in siteP:
            #Calculation of Rrup and Rjb
            rrups = np.sqrt(np.power(faultP[:,0]-x,2)+np.power(faultP[:,1]-y,2)+np.power(faultP[:,2]-z,2))
            rrup.append(np.min(rrups)/1000)
            rjbs = np.sqrt(np.power(faultP[:,0]-x,2)+np.power(faultP[:,1]-y,2))
            rjb.append(np.min(rjbs)/1000)

            # Calculation of Rx and alpha
            a2,b2,c2 = -1/a1,-1,x*(1/a1)+y # constant definition for second perpendicular line to polyfit
            A = np.array([[a1,b1],[a2,b2]]) # Solving the two linear
            b = np.array([-c1,-c2]) # equation with two
            Ab = np.linalg.solve(A,b) # unknown parameters
            xl,yl=Ab[0],Ab[1] # as x and 
            rxline = np.sqrt(float(np.power(x-xl,2)+np.power(y-yl,2))) # distance between two point (Rx)
            rxFtops = np.sqrt((np.power(x-fTops[:,0],2)+np.power(y-fTops[:,1],2))) # distances to top edge of fault
            rxFtop = np.min(rxFtops) # shortest distance to top edge of the fault 
            index1 = np.where(rxFtops==rxFtops)[0][0] # index of the point which is other edge of the shortest distance
            if index1 != 0:
                index2 = index1 - 1
                p1,p2,p3=np.array([x,y]),fTops[index1][0:2],fTops[index2][0:2] # P1 is the site P2 is the closest point P3 is the next point to the closest
            
                p21 = p1-p2
                p23 = p3-p2
                coAng = np.dot(p21,p23)/ (np.linalg.norm(p21)*np.linalg.norm(p23))
                coAng = np.arccos(coAng)
                coAng = np.degrees(coAng)
            else:
                index2 = index1+1
                P1,P2,P3=np.array([x,y]), fTops[index1][0:2],fTops[index2][0:2] # P1 is the site P2 is the closest point P3 is the next point to the closest
                p21 = P1-P2
                p23 = P3-P2
                coAng = np.dot(p21,p23)/ (np.linalg.norm(p21)*np.linalg.norm(p23))
                coAng = np.arccos(coAng)
                coAng = np.degrees(coAng)
                coAng = np.abs(180-coAng)
      
            sign = (x-x1)*(y2-y1)-(y-y1)*(x2-x1) # which side of the line?
            if sign >= 0:
                sign = 1 # for hanging wall
            else: # for footwall
                sign = -1
        
            rxline = rxline/1000 * sign # /1000 for m to km
            coAng = coAng * sign
            rx.append(rxline)
            alpha.append(coAng)
            side.append(sign)
        return np.array(rjb), np.array(rrup), np.array(rx), np.array(alpha), np.array(side)
    f1,f2,f3,f4,f5 = getVals(siteP,faultP)
    return f1,f2,f3,f4,f5
    # returns Rjb Rrup Rx Alpha Side