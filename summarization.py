#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from algorithme import similarite, powerMethod
from preprocessing import tfIdf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Sammurization :
        
    def __init__(self, corpus, sum_size = 5) :
        self.corpus = corpus
        self.sum_size   = sum_size
        self.scores     = ''
        self.similarite = '' 
        
    def sumurize(self) :
        tf_idf = tfIdf(self.corpus)
        self.similarite, degrees = similarite(tf_idf)
        self.scores  = powerMethod(self.similarite, degrees)
        sent_index   = np.argsort(self.scores )[::-1][:self.sum_size]
        sent_index.sort()
        sent_list    = list(tf_idf.loc[sent_index]['sentences'].values)
        summu        = ' '.join(sent_list)
        return summu
        
    
    def graph(self) :
        edges = []
        for ids, v in enumerate(self.similarite) :
            for i, s in enumerate(v) :
                if i != ids and s > 0.0 :
                   edges.append(("s"+str(ids), "s"+str(i)))
        G = nx.Graph()
        G.add_edges_from(edges)
        options = {'node_size':300, 'node_color':'red', 'with_labels':True}
        nx.draw_circular(G, **options)
        plt.show()
