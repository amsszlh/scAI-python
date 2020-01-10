# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:26:52 2019

@author: Lihua Zhang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:02:27 2019

@author: Lihua Zhang
"""

# scAI main function
# import packages
import numpy as np
import numpy.matlib
from numpy import linalg as LA
from numpy import matrix


def run_scAI(X1,X2,K,S,Alpha,Lambda,Gamma,Stop_rule,Seed,W1,W2,H,Z,R):
# parameters for scAI
    if S is None:
        S = 0.25
        
    if Alpha is None:
        Alpha = 1
        
    if Lambda is None:
        Lambda = 10000
        
    if Gamma is None:
        Gamma = 1
        
    if Stop_rule is None:
        Stop_rule = 1
        
    if Seed is None or Seed==0:
        Seed = 1
        
    print("Initializing...")
    d1 = X1.shape
    d2 = X2.shape
    p = d1[0]
    n = d1[1]
    q = d2[0]
    if type(W1) == type(None):
        np.random.seed(Seed)
        W1 = numpy.matlib.rand(p,K)
        
    if type(W2) == type(None):
        np.random.seed(Seed)
        W2 = numpy.matlib.rand(q,K)
    
    if type(H) == type(None):
        np.random.seed(Seed)
        H = numpy.matlib.rand(K,n)
    
    if type(Z) == type(None):
        np.random.seed(Seed)
        Z = numpy.matlib.rand(n,n)
    
    if type(R) == type(None):
        np.random.seed(Seed)
        R = np.random.binomial(1,S,size=(n,n))
        
    # main function for scAI
    Maxiter = 500
    XtX2 = np.dot(np.transpose(X2),X2)
    obj_old = 1
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    Z = np.asarray(Z)
    R = np.asarray(R)
    print('Updating...')
    for iter in range(1,Maxiter+1):
        print(iter)
        # normalize H
        H = matrix(H)
        lib = np.sum(H,axis = 1)
        H = H/np.tile(lib,(1,n))
        H = np.asarray(H)
        # update W1
        HHt = np.dot(H,np.transpose(H))
        X1Ht = np.dot(X1,np.transpose(H))
        W1HHt = np.dot(W1,HHt)
        W1 = W1*X1Ht/(W1HHt+np.spacing(1))
        # update W2
        ZR = Z*R
        ZRHt = np.dot(ZR,np.transpose(H))
        X2ZRHt = np.dot(X2,ZRHt)
        W2HHt = np.dot(W2,HHt)
        W2 = W2*X2ZRHt/(W2HHt+np.spacing(1))
        # update H
        W1tX1 = np.dot(np.transpose(W1),X1)
        W2tX2 = np.dot(np.transpose(W2),X2)
        W2tX2ZR = np.dot(W2tX2,ZR)
        HZZt = np.dot(H,(Z+np.transpose(Z)))
        W1tW1 = np.dot(np.transpose(W1),W1)
        W2tW2 = np.dot(np.transpose(W2),W2)
        H = H*(Alpha*W1tX1+W2tX2ZR+Lambda*HZZt)/(np.dot(Alpha*W1tW1+W2tW2+2*Lambda*HHt+Gamma*np.ones([K,K]),H)+np.spacing(1))
        # update Z
        HtH = np.dot(np.transpose(H),H)
        X2tW2H = np.dot(np.transpose(W2tX2),H)
        RX2tW2H = R*X2tW2H
        XtX2ZR = np.dot(XtX2,ZR)
        XtX2ZRR = XtX2ZR*R
        Z = Z*(RX2tW2H+Lambda*HtH)/(XtX2ZRR+Lambda*Z+np.spacing(1))
        if Stop_rule == 2:
            obj = Alpha*pow(LA.norm(X1-np.dot(W1,H),ord = 'fro'),2)+pow(LA.norm(np.dot(X2,ZR)-np.dot(W2,H),ord = 'fro'),2)+Lambda*pow(LA.norm(Z-np.dot(np.transpose(H),H),ord = 'fro'),2)+Gamma*pow(LA.norm(np.dot(np.ones([1,K]),H),ord = 'fro'),2)
            if (obj_old-obj)/obj_old < 1e-6 and iter > 1:
                break
        iter = iter+1
            
    print ("\n")
    print ("## Running scAI with seed %d ##" % Seed)
    print ("\n")
    
    return W1, W2, H, Z, R
        
    
    
    