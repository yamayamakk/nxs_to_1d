"""
Created on Fri Apr 24 11:42:54 2020

@author: ShogoK2
"""


import fabio
import sys

#import h5py
import struct, os
import numpy as np
import pyFAI
from pyFAI import detectors
from pyFAI.geometryRefinement import GeometryRefinement
from PIL import Image, ImageOps
import math
from decimal import Decimal, ROUND_HALF_UP
import h5py
import glob

######
#flat field補正はspmファイル
#263->s1_m0, 114->s1_m1, 115->s1_m2, 264->s2_m0, 265->s2_m1, 266->s2_m2
#####

file_path_all=glob.glob("*.nxs")
for file_path in file_path_all:
    """ Lambda nxs """
    fh5 = h5py.File(file_path, "r", driver="stdio")
    datastream = fh5["/entry/instrument/detector/data"]
    image_num, BRY, BRX = datastream.shape
    for i in range(image_num):
                if i==0:
                    raw_data0 = np.array(datastream[0, :, :])
                else:
                    raw_data0 = raw_data0 + np.array(datastream[i, :, :])
    ######異常値判定######
    average=np.average(raw_data0)
    condition1 = raw_data0 >average*3
    condition2 = raw_data0 <average/3
    raw_data0[condition1] = -1
    raw_data0[condition2] = -1
    average2=np.average(raw_data0[raw_data0>0])

    raw_data_ave = average2/raw_data0
    raw_data_ave[condition1] = 0
    raw_data_ave[condition2] = 0
    #condition2 = raw_data_ave >3

    #raw_data_ave[condition2] = -1


    im = Image.fromarray(np.float64(raw_data_ave))
    im.save(file_path+"_FF.tif", format="tiff")
