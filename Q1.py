# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:05:31 2019

@author: Lok Hsiao Yen
"""

import numpy as np
def SumExcept(nums):
    out = []
    n = len(nums)
    if n >= 2:
        for i in range(n):
            out.append(np.concatenate((nums[:i],nums[i+1:])).sum().astype(int))           
    else:
        print('length of nums must be greater than 1')     
    return out


nums = [1,2,3,4]
SumExcept(nums)








