"""
Created on 14 November 2025
@author:RSJ

In this test we generate some random data and check the `register` method
of `reference_image.ReferenceImage`.
"""
import regatta
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob

ss,sd,sf = 120,15,25

sample = np.random.rand(ss,sd,sf)# + 1j*np.random.rand(100,10,20)

rs,rd,rf = 100,10,20
ref_data = sample[:rs,:rd,:rf]


# create a ReferenceImage object based on the sample data
refim = regatta.reference_image.ReferenceImage(ref_data)

# Case 0: register 3D ref_data to itself
res = refim.register(ref_data)
try:
    assert res['d0']==0 and res['d1']==0 and res['d2']==0
    print('Test case 0 passed. Registering a volume to itself.')
except AssertionError as ae:
    sys.exit('Test case 0 failed: %s'%res)


# Case 1: register a shifted 3D target to the reference image
shifts = [2,3,4]
target = sample[shifts[0]:rs+shifts[0],shifts[1]:rd+shifts[1],shifts[2]:rf+shifts[2]]

res = refim.register(target)
try:
    assert res['d0']==-shifts[0] and res['d1']==-shifts[1] and res['d2']==-shifts[2]
    print('Test case 1 passed. Registering a rigid 3D translation in 3D.')
except AssertionError as ae:
    sys.exit('Test case 1 failed: %s'%res)
    
# Case 2: register a shifted 2D target to the reference image
target_slow_coordinate = 50
target = sample[target_slow_coordinate,shifts[1]:rd+shifts[1],shifts[2]:rf+shifts[2]]

res = refim.register(target)
try:
    assert res['d0']==target_slow_coordinate and res['d1']==-shifts[1] and res['d2']==-shifts[2]
    print('Test case 2 passed. Registering a 2D image to a 3D reference using broadcasting.')
except AssertionError as ae:
    sys.exit('Test case 2 failed: %s'%res)

# Case 3: register a shifted 2D target to a 2D reference image
target_slow_coordinate = 50
ref_data_slice = ref_data[target_slow_coordinate,:,:]
refim_2 = regatta.reference_image.ReferenceImage(ref_data_slice)
target = sample[target_slow_coordinate,shifts[1]:rd+shifts[1],shifts[2]:rf+shifts[2]]

res = refim_2.register(target)
try:
    assert res['d0'] is None and res['d1']==-shifts[1] and res['d2']==-shifts[2]
    print('Test case 3 passed. Registering a 2D image to a 2D reference.')
except AssertionError as ae:
    sys.exit('Test case 3 failed: %s'%res)
