#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np


def lcs(X, Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n + 1) for i in range(m + 1)] 
  
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1])  
    return L[m][n] 

        
def lcs_etoile(s1,s2) :
    if len(s1) == 0:
        return 0
    else :
        l2 = s2.copy()
        s = 0
        for w1 in s1 :
            if len(l2) > 0:
               l = [lcs(w1, w2) for w2 in l2]
               if max(l) > 0 :
                  del l2[l.index(max(l))]
                  if max(l) / len(w1) >= 0.6 :
                    s += max(l) / len(w1)
        return s / float(len(s1))
    
       
def cosine(p1,p2) :
    dot_product = sum(p*q for p,q in zip(p1, p2))
    magnitude = sqrt(sum([val**2 for val in p1])) * sqrt(sum([val**2 for val in p2]))
    if not magnitude :
        return 0.0
    return dot_product/magnitude


def similarite(frame, alpha=0.9, similarity_threshold = 0.07) :
        similarities = np.zeros(shape=(frame.values.shape[0],
                                       frame.values.shape[0]))
        degrees = np.zeros(shape=(frame.values.shape[0]))
        for ind1, s1 in frame.iterrows() :
            for ind2, s2 in frame.iterrows() :
                sim = alpha*cosine(s1.values[:len(frame.columns)-2], s2.values[
                      :len(frame.columns)-2]) + (1-alpha)*lcs_etoile(s1.values[
                           -2], s2.values[-2])
                if sim > similarity_threshold :
                   similarities[ind1][ind2] = 1
                   degrees[ind2] += 1
                else :
                    similarities[ind1][ind2] = 0.0
        return  similarities / degrees.reshape(frame.values.shape[0],1), degrees
    
    

def powerMethod(similarity_matrix,
                degrees,
                stopping_criterion=0.00005,
                max_loops=3000):
    
    p_initial = np.ones(shape=len(degrees))/len(degrees)
    i = 0
    # loop until no change between successive matrix iterations
    while True:
        i += 1
        p_update = np.matmul(similarity_matrix.T, p_initial)
        delta = np.linalg.norm(p_update - p_initial)
        if delta < stopping_criterion or i >= max_loops:
            break
        else:
            p_initial = p_update
    p_update = p_update/np.max(p_update)
    return p_update





    