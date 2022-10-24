#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.optimize as opt
from pathlib import Path
libpath=r"../Lib/site-packages/"
sys.path.insert(0,libpath)
print(sys.path)
import okada4py as ok92
class objectivefunction:
    'Objective function containing all the requiered informations for the optimisation'
    def __init__(self):
        pass
 
    def loadgps(self,fname):
        'function loading the gps data from an input file fname'

        #self.x,self.y,self.ue,self.un,self.sige,self.sign = np.loadtxt(fname,comments="#",unpack=True,dtype='f,f,f,f,f,f')
        #self.x,self.y = self.x.flatten(),self.y.flatten()       
 
        data = np.loadtxt(fname,comments="#")
        self.x = data[:,0].flatten(); self.y = data[:,1].flatten()
        self.ue = data[:,2].flatten(); self.un = data[:,3].flatten() 
 
        # set depth to zero 
        self.z = np.zeros(self.x.shape)
        # initialise data vector
        self.d = np.hstack([self.ue, self.un]).flatten()
    
    def loadfault(self,fname):
        'function loading the fault parameter from an input file fname'

        x1,x2,x3,length,width,strike,dip = np.loadtxt(fname,comments="#",unpack=True)
        self.x1,self.x2,self.x3,self.length,self.width,self.strike,self.dip=np.array([x1]),np.array([x2]),np.array([x3]),np.array([length]),np.array([width]),np.array([strike]),np.array([dip])
         
    def gm(self,m):
        'forward dislocation model in an elastic half-space from Okada 1992'

        # forward model
        mu = 30.0e9
        nu = 0.25
        self.slip, self.width, self.depth = m
        self.slip, self.width, self.depth = np.array([self.slip]), np.array([self.width]), np.array([self.depth])
        
        #####test####
        if test:
            self.x = np.arange(-20.0, 20.0, 5)
            self.y = np.arange(-20.0, 20.0, 5)
            self.x, self.y = np.meshgrid(self.x, self.y)
            self.x = self.x.flatten()
            self.y = self.y.flatten()
            self.z = np.zeros(self.x.shape) 
        #######
       
        u, d, s, flag, flag2 = ok92.okada92(self.x, self.y, self.z, self.x1, self.x2, self.x3, self.length, self.width, self.dip, self.strike, self.slip, np.array([0.0]), np.array([0.0]), mu, nu)
        u = u.reshape((self.x.shape[0], 3))
        self.me = u[:,0]; self.mn = u[:,1]

        #if test:
        #    # save forward model
        #    gpsdata = np.vstack([self.x,self.y,self.me,self.mn]).T
        #    np.savetxt('gpsdata.txt', gpsdata, header='East(UTM)   North(UTM) Ue(m)    Un(m)',fmt=('%.5f','%.5f','%.5f','%.5f'))

        return np.hstack([self.me, self.mn]).flatten()
 
    def residual(self,m):
        'Misfit function'
        g=np.asarray(self.gm(m))
        return np.sum((self.d-g)**2)

    def callback(self,m):
        'function printing the parameters'

        self.slip,self.width, self.depth = m
        self.slip,self.width, self.depth = np.array([self.slip]), np.array([self.width]), np.array([self.depth])
        print('x1:{} x2:{} x3:{} length:{} width:{} strike:{} dip:{} slip:{}'.format(self.x1[0],self.x2[0],self.x3[0],self.length[0],self.width[0],self.strike[0],self.dip[0],self.slip[0]))

    def plot(self):
        'function plot'

        if test == False:
            plt.scatter(self.x,self.y, marker='^',c='black' ,label='GPS network')
            plt.quiver(self.x,self.y,self.ue,self.un,color='b',label='GPS displacements',scale=.5)
        plt.quiver(self.x,self.y,self.me,self.mn,color='r',label='Model',scale=.5)
        ######
        # print fault trace
        ######
        fx,fy = [self.x1-sin(np.deg2rad(self.strike))*(self.length/2), self.x1+sin(np.deg2rad(self.strike))*(self.length/2) ], [self.x2-cos(np.deg2rad(self.strike))*(self.length/2),self.x2+cos(np.deg2rad(self.strike))*(self.length/2)]
        plt.plot(fx,fy,linewidth=3,color='black')
        plt.legend(loc='best')
    
    def getN(self):
        'function returning the number of gps stations'
        return self.x.shape

#-------------------------------------
# 1. Open data set and fault parameters
#-------------------------------------

wdir = './' 
inv = objectivefunction()
faultname=wdir + '/faults/' + 'pkfdi_seg.flt'
gpsname = wdir + '/gps/' + 'cgps_table.txt'
#gpsname = wdir + '/gps/' + 'gpsdata.txt'
inv.loadgps(gpsname)
inv.loadfault(faultname)
print('Number of GPS stations:',inv.getN())
test = False

#-------------------------------------
# 2. Initialise invert parameters
#-------------------------------------

slipi = -0.5 # slip
wi = 10.0 # width
di = 10.0 # depth

print('Initial parameter:')

# set model vector
minit = np.array([slipi, wi, di]) 

# set boundaries for model vector during optimisation
sigmam = np.array([1.,10., 10.]) # uncertainties for all parameters
mmax=minit+sigmam; mmin=minit-sigmam
bnd=np.column_stack((mmin,mmax))

# test forward model
inv.gm(minit)
inv.callback(minit)

#-------------------------------------
# 3. Optimisation
#-------------------------------------

mf = opt.fmin_slsqp(inv.residual,minit,iter=10000,bounds=bnd,full_output=False,acc=1e-09)

#-------------------------------------
# 4. Plot results
#-------------------------------------

print("Best-fitting model: ")
inv.callback(mf)
inv.plot()
plt.show()
