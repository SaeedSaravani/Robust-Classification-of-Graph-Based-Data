#-- Import Modules ------------------------------------------------------------
import numpy as np
import math


#------------------------------------------------------------------------------
###############################################################################

SIGMA = 0.15

###############################################################################

#-- Functions -----------------------------------------------------------------
#-- Run RobustGC Algortihm ----------------------------------------------------
"""
    Inputs:
        W: Weight Matrix (Symmetric)
        Y: labels (+1/-1 for Labeled Data and 0 for Un-Labeled Data)
        reg: Regularization Parameter (0<reg<1)
    Outputs:
        Predicted Labels for semi-supervised samples
"""
def RobustGC(W , Y , reg):
    number_of_nodes = W.shape[0]
    
    D = np.zeros((number_of_nodes, number_of_nodes) , dtype=float)
    for i in range(number_of_nodes):
        D[i][i] = np.sum(W[i,:])
    
    one = np.ones((number_of_nodes , number_of_nodes) , dtype=float)
    D_s = np.sqrt(D)
    D_s_inv = np.divide(one , D_s, out=np.zeros_like(one), where=D_s!=0)
    
    S = np.dot((np.dot(D_s_inv , W)),D_s_inv)   
    
    I = np.identity(n = number_of_nodes , dtype = float)    
    
    L = I - S    
    
    v = np.zeros((number_of_nodes), dtype = float)
    for i in range(number_of_nodes):
        v[i] = np.sqrt(D[i][i])
   
    
    
    #print(D)
    eigen_values , eigen_vectors = np.linalg.eig(L)
    eigen_values = np.sort(eigen_values)
    lambda_1 = eigen_values[1]    
    #print(eigen_values)
    gamma = reg * lambda_1
    #print("G" , reg , lambda_1 , gamma)
    
    t1 = L/gamma    
    t1 = t1 - I
    t1 = np.linalg.inv(t1)
    #t1 = np.linalg.pinv(t1)
    
    
    v_norm = np.divide(v , np.linalg.norm(v))
    
    t2 = np.dot(v_norm.T , Y)
    t2 = np.dot(v_norm , t2)
    t2 = Y - t2
        
    f = np.dot(t1 , t2)
    predicted_y = np.sign(f)    
    
    return f , predicted_y , gamma

#------------------------------------------------------------------------------

#-- PF_RobustGC : RobustGC with reg = 0.9 -------------------------------------
def PF_RobustGC(W , Y):
    return RobustGC(W, Y, reg=0.9)
#------------------------------------------------------------------------------


#-- Run Extended RobustGC Algortihm ----------------------------------------------------
"""
    Inputs:
        W: Weight Matrix (Symmetric)
        X: Feature vectors
        Y: labels (+1/-1 for Labeled Data and 0 for Un-Labeled Data)
        reg: Regularization Parameter (0<reg<1)
        queries: list of input feature vectors for prediction label
    Outputs:
        Predicted Labels for semi-supervised samples
"""


def RobustGC_Out_of_Sample(W , X , Y , reg , queries , pf = False):
    
    if pf:
        f , predicted_y , gamma = PF_RobustGC(W.copy() , Y.copy())
        
    else:
        f , predicted_y , gamma = RobustGC(W.copy() , Y.copy() , reg)
    
    number_of_nodes = W.shape[0]
    
    f_queries = []
    label_queries = []
    
    d_i_values = []
    
    for i in range(number_of_nodes):
        d_i = 0
        for j in range(number_of_nodes):
            k2 = Gaussian_Kernel(x= X[i,:], y=X[j,:], sigma=SIGMA)
            d_i += k2
        d_i_values.append(d_i)
    
    for q in queries:
        
        d_x = 0
        for i in range(number_of_nodes):
            k = Gaussian_Kernel(x = q, y = X[i,:], sigma=SIGMA)
            #print("SSS" , SIGMA , k)            
            d_x += k
    
        
        sum_sf = 0
        for i in range(number_of_nodes):
            k = Gaussian_Kernel(x = q, y = X[i,:], sigma=SIGMA)
            
            d_i = d_i_values[i]
            """
            d_i = 0
            for j in range(number_of_nodes):
                k2 = Gaussian_Kernel(x= X[i,:], y=X[j,:])
                d_i += k2
            """
            #print(d_x , d_i)
            s = k / math.sqrt(d_x * d_i)
            
            f_i = f[i]    
            sum_sf += s * f_i
        
        
        t1 = Gaussian_Kernel(x= q, y= q, sigma=SIGMA)
        t1 = t1 / d_x
        t1 = 1- gamma - t1
        t1 = 1/ t1
    
        f_x = t1 * sum_sf
        f_queries.append(f_x)
        label_queries.append(np.sign(f_x))
    
    return f , predicted_y , f_queries , label_queries 

#------------------------------------------------------------------------------

#-- Gaussian Kernel for two vector x , y -------------------------------------    
def Gaussian_Kernel(x,y, sigma=SIGMA):
    
    
    z = x - y
    z_norm = np.linalg.norm(z)
    z_norm_s = z_norm * z_norm
    
    
    
    return  math.exp(-1 * (z_norm_s / (2*sigma * sigma))) 
    
    
     
#------------------------------------------------------------------------------
###############################################################################
