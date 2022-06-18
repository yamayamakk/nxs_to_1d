
from re import L
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
import matplotlib.pyplot as plt


class Lambda_to_1d():
    def __init__(self):
        # Essential for changing
        this_path = os.path.dirname(os.path.abspath(__file__))
        self.ffpath   = this_path + "/ff_image/"
        self.maskpath = this_path +  "/mask/"        

        
        
        # Will be changed by function
        self.ponipath = "/Users/hirokiyamada/Desktop/pytest/lambda/1D-temp/data_1/poni/220602_SiO2_60s-scan48_1_m2.poni"
        self.file_path = "/Users/hirokiyamada/Desktop/pytest/lambda/1D-temp/data_1/220602_SiO2_60s-scan48_1_s1_m2.nxs"
        
        
        self.datapath="./data_1/"
        
        
        fflist = glob.glob(self.ffpath+"*.tiff")
        masklist = glob.glob(self.maskpath+"*.mask")
        
        ######## FF reading ######## 
        #263->s1_m0, 114->s1_m1, 115->s1_m2, 264->s2_m0, 265->s2_m1, 266->s2_m2
        # self.det_order = ["sn263_100sec_spm", "sn114_100sec_spm", "sn115_100sec_spm", "sn264_100sec_spm", "sn265_100sec_spm", "sn266_100sec_spm"]
        # self.ffimage = []
    
        # for i in range(len(self.det_order)):  
        #     tes = [order for order in fflist if self.det_order[i] in order]
        #     width=516
        #     height=1554
        #     fd = open(tes[0], "rb")
        #     f = np.fromfile(fd, dtype=np.float64, count=height*width)
        #     img = f.reshape((height,width))
        #     img[img >2.5]=0
        #     img[img < 0.4]=0
    
    
        #     fd.close()
        #     self.ffimage.append(img)
        # print("FF reading finished ")

        ######## Mask reading ######## 
   
        self.module_order =["m0", "m1", "m2", "m3", "m4", "m5"]
        self.maskimage = []
        self.ffimage = []
        self.x_energy = 60
        self.m_mode = "csm"


        for i in range(len(self.module_order)):  
            temp_mask = [order for order in masklist if self.module_order[i] in order]
            im = np.array(Image.open(temp_mask[0]))
            self.maskimage.append(im)
        
            temp_ff = [order for order in fflist if self.module_order[i] in order]
            im2 = np.array(Image.open(temp_ff[0]))
            self.ffimage.append(im2)
  
        print("Mask reading finished ")    
        print("initialization finished ")        



    def Change_FF(self):
        new_ff_path = glob.glob("%s%skeV_%s_*/"%(self.ffpath, self.x_energy, self.m_mode))
        self.ffpath = new_ff_path[0]
        fflist = glob.glob(self.ffpath+"*.tiff")

        print(self.ffpath)
        
        # reset the ff image        
        self.ffimage = []
        for i in range(len(self.module_order)):  
        
            temp_ff = [order for order in fflist if self.module_order[i] in order]
            im2 = np.array(Image.open(temp_ff[0]))
            self.ffimage.append(im2)
  
        print("FF re-reading finished ")    

       
       
        
        return True




    def Poni_detection(self):
        temp_path = self.file_path.rsplit("/",1)

        fname_split = temp_path[-1].split("_")        
        self.num_scan = fname_split[-3]


        if fname_split[-2] == "s2":
            self.num_module = 3 + int(fname_split[-1][1])
            #print(num_module)
        else:
            self.num_module = 0 + int(fname_split[-1][1])
            #print(num_module)
        
        poni_check = "%s_m%s.poni" %(self.num_scan, self.num_module)
        
        ch = glob.glob(temp_path[0] + "/poni/*%s_m%s.poni"%(self.num_scan, self.num_module))
        self.ponipath = ch[0]
        
        return True

    def Read_NXS_Accumulation(self):
        
        fh5 = h5py.File(self.file_path, "r", driver="stdio")
        f_nxs = fh5["/entry/instrument/detector/data"]
        
        fname = self.file_path.split("/")
        fname_split = fname[-1].split("_")
        
        # Judging the module number
        self.num_module = 0 # detector number (0-5)
        if fname_split[-2] == "s2":
            self.num_module = 3 + int(fname_split[-1][1])
            #print(num_module)
        else:
            self.num_module = 0 + int(fname_split[-1][1])
            #print(num_module)
               
        
        
        image_num, _, _ = f_nxs.shape
        for i in range(image_num):
            if i==0:
                raw_data = np.array(f_nxs[0, :, :])
            else:
                raw_data = raw_data + np.array(f_nxs[i, :, :])


        raw_data_ff = raw_data * self.ffimage[self.num_module]
        #print(raw_data)
        #print(raw_data_ff)
        
        self.raw_data_ff = np.flipud(np.rot90(raw_data_ff,-1)) ###90°回転        

        #pil_img_gray = Image.fromarray(self.raw_data_ff)
        #pil_img_gray.save("tes1.tiff")
        
        return self.raw_data_ff





    def Integrate_1d_azimuth(self, num_points=None, azimuth_range=None, mask=None, polarization_factor=None, method="csr", unit="2th_deg"):
        ai = pyFAI.load(self.ponipath)
        image_data = self.raw_data_ff

 
        num_points=1553#####固定値に変更
        tth, intensity, error = ai.integrate1d(image_data,
                                                    num_points,
                                                    correctSolidAngle=True,
                                                    azimuth_range=azimuth_range,
                                                    mask=self.maskimage[self.num_module],
                                                    polarization_factor=0.99,
                                                    method=method,
                                                    unit=unit,
                                                    error_model="poisson")
            ####error_modelを入れる場合は、error_model="poisson"を入れればよい　　>>q, I, sigma = ai.integrate1d(data, npt, unit="q_nm^-1", error_model="poisson")
        #index = np.where((intensity > 0) & (~np.isnan(intensity)))
        #tth = tth[index]
        #intensity = intensity[index]
        #error=error[index]
        
        return tth, intensity, error

        
    
        


tes = Lambda_to_1d()
#datapath = "/Users/hirokiyamada/Desktop/pytest/lambda/1D-temp/data_1/"
#nxs_list = glob.glob(datapath + "*.nxs")

nxs_list = glob.glob("./*.nxs")
print(nxs_list)
#
# For changing FF images
tes.x_energy = 30
tes.m_mode = "csm"
tes.Change_FF()
#
#




for nxs_name in nxs_list:
        # file名の絶対パスを投入すれば自動でponi, FF, mask 補正してくれる　絶対パスの中に/poni/xxx.poniが必要
        tes.file_path = nxs_name
        tes.Read_NXS_Accumulation()
        tes.Poni_detection()
        
        name_poni = tes.ponipath.split("/")
        name_nxs  = nxs_name.split("/")
                
        print(name_nxs[-1], name_poni[-1])
       
        twotheta, intensity, _ =  tes.Integrate_1d_azimuth()
        #plt.plot(twotheta, intensity)
        #plt.show()
        write_name = name_poni[-1][:-5] + ".dat"
        np.savetxt(write_name,np.c_[twotheta,intensity],fmt="%0.6f")
