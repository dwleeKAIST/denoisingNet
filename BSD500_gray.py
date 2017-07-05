from os import listdir
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import pywt
import ipdb

PIXEL_MAX     = 255.0
_20_div_Log10 = 8.6859


#########################
## [ batch, imgY, imgX, channel ]
class imgSet:
    def __init__(self, l_dir_train, l_dir_test, noise_stdev=0.5):
        self.l_dir_train    = l_dir_train
        self.l_dir_test     = l_dir_test
        self.stdev          = noise_stdev
        
        self.jpgList_train  = []
        for a_file in listdir(self.l_dir_train):
            self.jpgList_train.append(a_file)
               
        self.jpgList_test  = []        
        for a_file in listdir(self.l_dir_test):
            self.jpgList_test.append(a_file)        
            
        self.train_N  = int(len(self.jpgList_train))
        self.test_N   = int(len(self.jpgList_test))
        self.total_N  = self.train_N + self.test_N
        
        self.train_id = np.arange(self.train_N)
        self.test_id  = np.arange(self.test_N)
        
    def getABatch(self, isTrain, batchStart, aug=1):
        if isTrain:
            #training set
            ids = self.train_id
            jpgList = self.jpgList_train
            l_dir   = self.l_dir_train
        else:
            #test set
            ids = self.test_id
            jpgList = self.jpgList_test
            l_dir   = self.l_dir_test
            aug=0

        fname           = l_dir+jpgList[ids[batchStart]]
        labels        = np.asarray(Image.open(fname,'r'))
        if aug==1:
            labels        = self.getAug(labels)
        
        data = labels + np.random.normal(0, self.stdev, labels.shape)
        data[data<0.0]   = 0.0
        data[data>255.0] = 255.0        
        sz               = data.shape
        return data.reshape(1,sz[0],sz[1],1), labels.reshape(1,sz[0],sz[1],1)
    
    def getPatch(self, img1, img2, patchSize):
        sz = img1.shape
        y = int(np.random.randint(sz[1]-patchSize[0]))
        x = int(np.random.randint(sz[2]-patchSize[1]))
        return img1[:,y:y+patchSize[0],x:x+patchSize[1],:], img2[:,y:y+patchSize[0],x:x+patchSize[1],:]
    
    def getAug(self, img, idx1 = int(np.random.randint(2, size=1)), idx2 = int(np.random.randint(4, size=1))):
        return np.rot90(np.flip(img, idx1+2), idx2+1, axes=(1,2))
    

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

# imgSet 클래스를 상속 받아서 Wavelet 관련 기능 추가.
class imgSet_wv(imgSet):

    def __init__(self, l_dir_train, l_dir_test, noise_stdev=0.5, wv_type='haar'):
        imgSet.__init__(self, l_dir_train, l_dir_test,  noise_stdev)
        self.wv_type = wv_type
        self.sz_wv   = self.img2wv(Image.open(self.l_dir_train+self.jpgList_train[0],'r'), wv_dims=(0,1), concat_dim=2).shape
        
        #family = 'bior' #'db' #sym coif haar
        #wnames = pywt.wavelist(family)
        #print(wnames)
    
    def img2wv(self, img, wv_dims=(1,2), concat_dim=3):
        LL, (LH, HL, HH)    = pywt.dwt2(img, self.wv_type, axes = wv_dims)
        if len(LL.shape)==2:
            return np.stack((LL,LH,HL,HH), axis=concat_dim)
        else:
            return np.concatenate((LL,LH,HL,HH), axis=concat_dim)
            
    def wv2img(self, wv, wv_dims=(1,2), split_axis=3 ):
        LL, LH, HL, HH    = np.split(wv,axis=split_axis, indices_or_sections=4)
        return pywt.idwt2( (LL, (LH, HL, HH)), self.wv_type, axes=wv_dims)

    
    def getABatch(self, isTrain, batchStart):      
        data_img, labels_img = imgSet.getABatch(self, isTrain, batchStart)        
        return self.img2wv(self.doPad(self.checkPad(data_img))), self.img2wv(self.doPad(self.checkPad(labels_img)))
    
    ## get batch with patch extraction on wavelet domain
    def getPWBatch(self, isTrain, batchStart, batchEnd, patchSize, Aug=1):    
        
        if isTrain:
            #training set
            ids = self.train_id
            jpgList = self.jpgList_train
            l_dir   = self.l_dir_train
        else:
            #test set
            ids = self.test_id
            jpgList = self.jpgList_test
            l_dir   = self.l_dir_test
            aug=0
    
        batchEnd   = min(batchEnd, len(ids))
        cur_idset = ids[batchStart:batchEnd]
        batch_num = batchEnd-batchStart
        labels    = np.empty([batch_num, patchSize[0], patchSize[1], self.sz_wv[2]])
        data      = np.empty([batch_num, patchSize[0], patchSize[1], self.sz_wv[2]])
        
        ##################################################
        for i in range(batch_num):
            fname     = l_dir+jpgList[cur_idset[i]]
            tmp       = np.asarray(Image.open(fname,'r'))
            sz        = tmp.shape
            labels_i  = self.checkPad(tmp.reshape(1,sz[0],sz[1],1))
            if Aug==1:
                labels_i = imgSet.getAug(self, img=labels_i)
            data_i      = labels_i + np.random.normal(0, self.stdev, labels_i.shape)
            data_i[data_i<0.0]  =0.0
            data_i[data_i>255.0]=255.0
            
            labels_wv = self.img2wv(labels_i)
            data_wv   = self.img2wv(data_i)
            data_wv_p, labels_wv_p = imgSet.getPatch(self, data_wv, labels_wv,patchSize)
            labels[i,:,:,:] = labels_wv_p
            data[i,:,:,:]   = data_wv_p
            
        return data, labels
    
    def checkPad(self, img):
        sz = img.shape
        if sz[1]%2==1:
            img = np.pad(img, pad_width=((0,0),(0,1),(0,0),(0,0)), mode='edge')        
        
        if sz[2]%2==1:
            img = np.pad(img, pad_width=((0,0),(0,0),(0,1),(0,0)), mode='edge')
            
        return img
            
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
        
        PSNR = np.log(PIXEL_MAX/np.sqrt(np.mean((tmp_img-tmp_lbl)**2)))*_20_div_Log10
        print(" -- %.4f" % PSNR)
        tmp_inp[tmp_inp<0.0]   = 0.0
        tmp_inp[tmp_inp>255.0] = 255.0
        tmp_img[tmp_img<0.0]   = 0.0
        tmp_img[tmp_img>255.0] = 255.0
        tmp_lbl[tmp_lbl<0.0]   = 0.0
        tmp_lbl[tmp_lbl>255.0] = 255.0
        error_  = np.absolute(tmp_img-tmp_lbl)
        error_[error_>255.0]=255.0
        
        im = Image.fromarray(np.uint8(tmp_inp[0,:,:,0]),'L')
        im.save(log_dir+'/'+save_str+'-inp.jpg')
        im = Image.fromarray(np.uint8(tmp_img[0,:,:,0]),'L')
        im.save(log_dir+'/'+save_str+'-rec.jpg')
        im = Image.fromarray(np.uint8(tmp_lbl[0,:,:,0]),'L')
        im.save(log_dir+'/'+save_str+'-lbl.jpg')
        im = Image.fromarray(np.uint8(error_[0,:,:,0]),'L')
        im.save(log_dir+'/'+save_str+'-err.jpg')

