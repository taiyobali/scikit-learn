import numpy as np
import copy,math

def J_wb(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(w,x[i]) + b
        cost += (f_wb_i - y[i])**2
    return cost/(2*m)


def gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        err = (np.dot(w,x[i]) + b) - y[i]

        for j in range(n):
            dj_dw[j] += err * x[i,j]

        dj_db += err
    
    return dj_dw/m , dj_db/m 


def gradient_dest(x,y,alpha,iteration):
    w = 0
    b = 0
    J_hist = []
    for i in range(iteration):
        dj_dw_i,dj_db_i = gradient(x,y,w,b)

        w = w - (alpha * dj_dw_i)
        b = b - alpha * dj_db_i

        if i<100000:
            J_hist.append(J_wb(x,y,w,b))
    
    return w,b,J_hist
