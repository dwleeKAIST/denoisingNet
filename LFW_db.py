from os import listdir

from PIL import Image
import numpy as np
import tensorflow as tf
import random
import pywt
from skimage.measure import compare_ssim as ssim
import ipdb

PIXEL_MAX     = 255.0
_20_div_Log10 = 8.6859

class LFW_RGB:
    def __init__(self, l_dir, noise_stdev=0.5):
        self.l_dir    = l_dir
        self.stdev    = noise_stdev
        self.jpgList  = []
        for a_dir in listdir(self.l_dir):
            for a_file in listdir(self.l_dir+a_dir):
                self.jpgList.append(a_dir+'/'+a_file)
        self.total_N  = int(len(self.jpgList))
        self.sz       = np.array(Image.open(self.l_dir+self.jpgList[0],'r')).shape
        
        random.seed(0)
        rand_id       = [i for i in range(self.total_N)]
        random.shuffle(rand_id)
        tmp_id        = int(self.total_N*0.9)    
        self.train_id = rand_id[0:tmp_id-1]
        self.test_id  = rand_id[tmp_id:]
        
    def getBatch(self, isTrain, batchStart, batchEnd, Aug=1):
        if isTrain:
            #training set
            ids = self.train_id
        else:
            #test set
            ids = self.test_id
            
        batchEnd   = min(batchEnd, len(ids))
        cur_idset = ids[batchStart:batchEnd]
        batch_num = batchEnd-batchStart
        labels    = np.empty([batch_num, self.sz[0], self.sz[1], self.sz[2]])
        
        for i in range(batch_num):
            fname = self.l_dir+self.jpgList[cur_idset[i]]
            labels_i = np.asarray(Image.open(fname,'r'))
            if Aug==1:
                labels_i = self.getAug(labels_i.reshape(1,self.sz[0], self.sz[1], self.sz[2]))
            labels[i,:,:,:] = labels_i
        
        data = labels + np.random.normal(0, self.stdev, labels.shape)
        data[data<0.0]  =0.0
        data[data>255.0]=255.0        
        
        return data,labels
    
    def getAug(self, img, idx1 = int(np.random.randint(2, size=1)), idx2 = int(np.random.randint(4, size=1))):
        return np.rot90(np.flip(img, idx1+2), idx2+1, axes=(1,2))
        
    
    def getPatch(self, img1, img2, patchSize):
        sz = img1.shape
        y = int(np.random.randint(sz[1]-patchSize[0]))
        x = int(np.random.randint(sz[2]-patchSize[1]))
        return img1[:,y:y+patchSize[0],x:x+patchSize[1],:], img2[:,y:y+patchSize[0],x:x+patchSize[1],:]
    
    def shuffleTrain_id(self):
        random.shuffle(self.train_id)
    
    def getTotalN(self):
        return self.total_N
    
    def getTrainN(self):
        return len(self.train_id)
    
    def getTestN(self):
        return len(self.test_id)
    
    def getDimForNet(self):
        return self.sz

# LFW_RGB 클래스를 상속 받아서 Wavelet 관련 기능 추가.
class LFW_RGB_wv(LFW_RGB):

    def __init__(self, l_dir, noise_stdev=0.5, wv_type='haar'):
        LFW_RGB.__init__(self, l_dir, noise_stdev)
        self.wv_type = wv_type
        self.sz_wv   = self.img2wv(Image.open(self.l_dir+self.jpgList[0],'r'), wv_dims=(0,1), concat_dim=2).shape
        
        #family = 'bior' #'db' #sym coif haar
        #wnames = pywt.wavelist(family)
        #print(wnames)
    
    def img2wv(self, img, wv_dims=(1,2), concat_dim=3):
        LL, (LH, HL, HH)    = pywt.dwt2(img, self.wv_type, axes=wv_dims)
        return np.concatenate((LL,LH,HL,HH),axis=concat_dim)
            
    def wv2img(self, wv, wv_dims=(1,2), split_axis=3 ):
        LL, LH, HL, HH    = np.split(wv,axis=split_axis,indices_or_sections=4)
        return pywt.idwt2( (LL, (LH, HL, HH)), self.wv_type, axes=wv_dims)

    
    def getBatch(self, isTrain, batchStart, batchEnd, Aug=1):        
        data_img, labels_img = LFW_RGB.getBatch(self, isTrain, batchStart, batchEnd, Aug=1)        
        return self.img2wv(data_img), self.img2wv(labels_img)
    
    def getABatch(self, isTrain, batchStart, Aug=0):        
        data_img, labels_img = LFW_RGB.getBatch(self, isTrain, batchStart, batchStart+1, Aug=Aug)        
        return self.img2wv(self.doPad(data_img)), self.img2wv(self.doPad(labels_img))
    
    def getPBatch(self, isTrain, batchStart, batchEnd, patchSize, Aug=1):       
        data_img, labels_img = LFW_RGB.getBatch(self, isTrain, batchStart, batchEnd, Aug=Aug)        
        return LFW_RGB.getPatch(self, self.img2wv(data_img), self.img2wv(labels_img), patchSize=patchSize)
    
    def doPad(self, img, top=3, bottom=3,left=3, right=3):
        img = np.pad(img, pad_width=((0,0),(top,bottom),(left,right),(0,0)), mode='edge')        
        return img
    
    def doInvPad(self, img, top=3, bottom=3,left=3, right=3, axis=1):
        sz = img.shape
        if len(sz)==4:
            return img[:, top:-bottom, left:-right, :]
        else:
            return img[top:-bottom, left:-right, :]
    
    def getDimForNet(self):
        return self.sz_wv
        
    def np2img_save(self, inp_img, rec_img, lbl_img, log_dir, save_str='test1'):
        
        tmp_inp = self.wv2img(inp_img, wv_dims=(1,2), split_axis=3 )
        tmp_img = self.wv2img(rec_img, wv_dims=(1,2), split_axis=3 )
        tmp_lbl = self.wv2img(lbl_img, wv_dims=(1,2), split_axis=3 )
        tmp_inp = self.doInvPad(tmp_inp)
        tmp_img = self.doInvPad(tmp_img)
        tmp_lbl = self.doInvPad(tmp_lbl)
        
        tmp_inp[tmp_inp<0.0]   = 0.0
        tmp_inp[tmp_inp>255.0] = 255.0
        tmp_img[tmp_img<0.0]   = 0.0
        tmp_img[tmp_img>255.0] = 255.0
        tmp_lbl[tmp_lbl<0.0]   = 0.0
        tmp_lbl[tmp_lbl>255.0] = 255.0
        error_  = np.absolute(tmp_img-tmp_lbl)
        error_[error_>255.0]=255.0
        
        print(save_str)
        PSNR_in = np.log(PIXEL_MAX/np.sqrt(np.mean((tmp_inp-tmp_lbl)**2)))*_20_div_Log10
        PSNR    = np.log(PIXEL_MAX/np.sqrt(np.mean((tmp_img-tmp_lbl)**2)))*_20_div_Log10
        print(" --PSNR :  %.4f --> %.4f" % (PSNR_in, PSNR))
        
        SSIMin  = ssim(np.uint8(tmp_inp[0,:,:,:]),np.uint8(tmp_lbl[0,:,:,:]), multichannel=True)
        SSIMv   = ssim(np.uint8(tmp_img[0,:,:,:]),np.uint8(tmp_lbl[0,:,:,:]), multichannel=True)
        print(" --SSIM :  %.4f --> %.4f" % (SSIMin, SSIMv))
        
        im = Image.fromarray(np.uint8(tmp_inp[0,:,:,:]))
        im.save(log_dir+'/'+save_str+'-inp.jpg')
        im = Image.fromarray(np.uint8(tmp_img[0,:,:,:]))
        im.save(log_dir+'/'+save_str+'-rec.jpg')
        im = Image.fromarray(np.uint8(tmp_lbl[0,:,:,:]))
        im.save(log_dir+'/'+save_str+'-lbl.jpg')
        im = Image.fromarray(np.uint8(error_[0,:,:,:]))
        im.save(log_dir+'/'+save_str+'-err.jpg')

