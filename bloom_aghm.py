#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:05:42 2022

@author: aghm
"""
import numpy as np
import math
import random 
import mmh3
from bitarray import bitarray
from bitarray.util import ba2int
#items_count : int
#Number of items expected to be stored in bloom filter
#fp_prob : float
#False Positive probability in decimal
class BloomFilter(object):
    def __init__(self, items_count,p):
		# False possible probability in decimal
        self.p = p
        # Size of bit array to use
        self.size = self.get_size(items_count, p)
        # number of hash functions to use
        self.hash_count = self.get_hash_count(p)
        # Bit array of given size
        self.bit_array = bitarray(self.size)
		# initialize all bits as 0
        self.bit_array.setall(0)

       ##########################
        self.bitarr = bitarray(self.size)
		# initialize all bits as 0
        self.bitarr.setall(0) 
    def add(self, item):
		#Add an item in the filter
        digests = []
        for i in range(self.hash_count):
			# create digest for given item.
			# i work as seed to mmh3.hash() function
			# With different seed, digest created is different
            digest = mmh3.hash(item, i) % self.size
            digests.append(digest)
			# set the bit True in bit_array
            self.bit_array[digest] = True
            
    def check(self, item):
		#Check for existence of an item in filter
        self.bitarr.setall(0)
        digests = []
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            digests.append(digest)
                #return False
            self.bitarr[digest] = True
    
    def perturb(self,fbp):
        L = [self.bitarr[i:i+1] for i in range(len(self.bitarr))]
        digests = []
        for i in L:
           if(ba2int(i)==0):
               y=False
           if(ba2int(i)==1):
               y=True
           p_sample = np.random.random_sample()#returns value between 0 and 1  
           if p_sample <= fbp:
               v = np.random.randint(0,2)
               if(v==0):
                   y=False
               if(v==1):
                   y=True
           digests.append(y)
        self.bitarr= bitarray(digests)   
      
           
    @classmethod
    def get_size(self, n, p):
        #n : int, number of items expected to be stored in filter
	#p : float, False Positive probability in decimal
        m = n*(math.log(1/p))/(math.log(2)*math.log(2))
        return int(m)
    
    @classmethod
    def get_hash_count(self, p):

        #k=math.log(1/p)/math.log(2)
        k=4
        return int(k)

