import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def SNR_2d (x, y, diagnal_estimate = False):
    n,p = x.shape
    x_bar = x.mean(axis = 0)
    x = x - x_bar

    yi = y[:, 0]
    li = max(yi)+1
    yj = y[:, 1]
    lj = max(yj)+1

    nij = np.array([[sum((yi == i) * (yj == j)) for j in range(lj)] for i in range(li)])
    ni_ = np.sqrt(nij.sum(axis = 1))
    nj_ = np.sqrt(nij.sum(axis = 0))
    nij_ = np.sqrt(nij).reshape(-1,1)

    xi = np.array([x[yi == i,].mean(axis = 0) for i in range(li)])
    xj = np.array([x[yj == j,].mean(axis = 0) for j in range(lj)])
    xij = np.array([[x[(yi == i) * (yj == j), :].mean(axis = 0) for j in range(lj)] for i in range(li)])

    a = np.multiply(xi.T, ni_)
    A = a.dot(a.T)
    b = np.multiply(xj.T, nj_)
    B = b.dot(b.T)
    xij_ = xij.reshape(-1,p)
    c = np.multiply(xij_, nij_)
    C = c.T.dot(c) #C is for all types

    D = np.zeros((p,p))
    if diagnal_estimate:
        wcd = np.zeros((1, p))
    for i in range(li):
        for j in range(lj):  
            d = np.subtract(x[(yi == i) * (yj == j), :], xij[i][j])
            if diagnal_estimate:
                wcd = wcd + np.sum(d**2, axis = 0)
            else:  
                D = D + d.T.dot(d)
    if diagnal_estimate:
        wcd_ = wcd[0]
        assert (sum(wcd_ == 0) == 0)
        D = np.diag(wcd_)

    return C/D, A/D, B/D

def EV_2d (x, y, diagnal_estimate = False):
    n,p = x.shape
    x_bar = x.mean(axis = 0)
    x = x - x_bar

    yi = y[:, 0]
    li = max(yi)+1
    yj = y[:, 1]
    lj = max(yj)+1

    nij = np.array([[sum((yi == i) * (yj == j)) for j in range(lj)] for i in range(li)])
    ni_ = np.sqrt(nij.sum(axis = 1))
    nj_ = np.sqrt(nij.sum(axis = 0))
    nij_ = np.sqrt(nij).reshape(-1,1)

    xi = np.array([x[yi == i,].mean(axis = 0) for i in range(li)])
    xj = np.array([x[yj == j,].mean(axis = 0) for j in range(lj)])
    xij = np.array([[x[(yi == i) * (yj == j), :].mean(axis = 0) for j in range(lj)] for i in range(li)])

    a = np.multiply(xi.T, ni_)
    A = a.dot(a.T)
    b = np.multiply(xj.T, nj_)
    B = b.dot(b.T)

    D = np.zeros((p,p))
    if diagnal_estimate:
        wcd = np.zeros((1, p))
    for i in range(li):
        for j in range(lj):  
            d = x[(yi == i) * (yj == j), :]
            if diagnal_estimate:
                wcd = wcd + np.sum(d**2, axis = 0)
            else:  
                D = D + d.T.dot(d)
    if diagnal_estimate:
        wcd_ = wcd[0]
        assert (sum(wcd_ == 0) == 0)
        D = np.diag(wcd_)

    return A/D, B/D

def MI_2d(x, y):
    n, p = x.shape
    MI = []
    Hs = []
    for k in range(p):
        xx = x[:,k]
        d = pd.cut(xx, 10).codes
        df = pd.DataFrame(y, columns = ['i', 'j'])
        df['u'] = d
        df['iu'] = df['i']*10+df['u']
        df['ju'] = df['j']*10+df['u']
        pu = (df.groupby(by=['u']).count()/n).iloc[:,0]
        Hu = sum(-pu*np.log2(pu))
        pi = (df.groupby(by=['i']).count()/n).iloc[:,0]
        Hi = sum(-pi*np.log2(pi))
        piu = (df.groupby(by=['iu']).count()/n).iloc[:,0]
        Hiu = sum(-piu*np.log2(piu))
        Miu = Hi+Hu-Hiu
        pj = (df.groupby(by=['j']).count()/n).iloc[:,0]
        Hj = sum(-pj*np.log2(pj))
        pju = (df.groupby(by=['ju']).count()/n).iloc[:,0]
        Hju = sum(-pju*np.log2(pju))
        Mju = Hj+Hu-Hju
        MI.append([Miu, Mju])
    Hs.append(Hi)
    Hs.append(Hj)
    return np.array(MI), np.array(Hs)

def MIG (MI, Hs): # MI is of axes x features (2 here)
    n, p = MI.shape
    m_sum = 0
    for k in range(p):
        m = np.sort(MI[:,k])
        m_diff = m[-1]-m[-2]
        m_diff_normalized = m_diff/Hs[k]
        m_sum += m_diff_normalized
    result = m_sum/n
    print (result)

def Modularity(MI, N):
    n,p = MI.shape
    ds = []
    for k in range(n):
        m = np.sort(MI[k,:])
        m_rest = m[:-1]
        theta = m[-1]
        m_rest_sum = np.sum(m_rest**2)
        delta = m_rest_sum/theta**2/(N-1)
        ds.append(1-delta)
    D = np.mean(ds)
    return ds, D

