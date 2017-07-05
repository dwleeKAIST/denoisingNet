from os import listdir, path, makedirs

from PIL import Image as img
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as li
import time 
import ipdb

from BSD500_gray import imgSet_wv
from nets import denoising_net as model

db    = imgSet_wv(l_dir_train='F:/flagship/denoising/data/train_gray/', l_dir_test='F:/flagship/denoising/data/test_gray/', noise_stdev=15, wv_type='haar')
totalN = db.getTotalN()   

BATCH_SIZE = 32
patchSize  = [20, 20]
NUM_EPOCHS = 1000


IMG_SIZE_Y, IMG_SIZE_X, IMG_SIZE_CH = db.getDimForNet()
#print(db.getDimForNet())

train_size = db.getTrainN()
test_size  = db.getTestN()
dtype      = tf.float32

##
ckpt_dir  = './result/wave15sgm_patch_BSD_gray/ckpt_dir'
log_dir   = './result/wave15sgm_patch_BSD_gray/log_dir'

if not path.exists(log_dir):
    makedirs(log_dir)
if not path.exists(ckpt_dir):
    makedirs(ckpt_dir)

## data feed dict and network 
data_node   = tf.placeholder( dtype, shape = (None, None, None, IMG_SIZE_CH) )
label_node  = tf.placeholder( dtype, shape = (None, None, None, IMG_SIZE_CH) )

net_out       = model.net(data_node, residual_tag=1, IMG_SIZE_CH=4)

loss          = tf.reduce_mean(tf.squared_difference(net_out,label_node,"L2_loss"))
tf.summary.scalar("loss", loss)

PIXEL_MAX     = 255.0
_20_div_Log10 = 8.6859
psnr          = tf.log(PIXEL_MAX/tf.sqrt(loss))*_20_div_Log10
tf.summary.scalar("PSNR", psnr)

batch         = tf.Variable(0, dtype=dtype)
#lr            = tf.train.exponential_decay(0.01, batch*BATCH_SIZE, totalN, 0.95, staircase=True )
lr            = tf.train.exponential_decay(0.1, batch*BATCH_SIZE, totalN, 0.95, staircase=False )


optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(loss, colocate_gradients_with_ops=True)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(loss,                                        var_list=lib.params_with_name('cost'), colocate_gradients_with_ops=True)
merged_all    = tf.summary.merge_all()
saver         = tf.train.Saver()

def myNumExtractor(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return int(tail)

def np2img_save(array, save_jpg):
    ipdb.set_trace()
    tmp_img = db.wv2img(array, wv_dims=(0,1), split_axis=2 )
    tmp_img = db.doInvPad(tmp_img)
    tmp_img[tmp_img<0.0]=0.0
    tmp_img[tmp_img>255.0]=255.0
    im = img.fromarray(np.uint8(tmp_img[:,:,0]),'L')
    im.save(save_jpg)
    
def np2img_save_(array, label, save_jpg):
    tmp_img = db.wv2img(array)
    tmp_lbl = db.wv2img(label)
    tmp_img = db.doInvPad(tmp_img)
    tmp_lbl = db.doInvPad(tmp_lbl)
    print(" -- %.4f" % np.log(PIXEL_MAX/np.sqrt(np.mean((tmp_img-tmp_lbl)**2)))*_20_div_Log10)
    error_  = np.absolute(tmp_img-tmp_lbl)
    error_[error_>255.0]=255.0

    im = img.fromarray(np.uint8(error_[0,:,:,0]),'L')
    im.save(save_jpg)
    
##
with tf.Session() as sess:
    
    len_batch      = int(train_size/BATCH_SIZE)
    len_batch_test = int(test_size/BATCH_SIZE)
    
    # check whether it have beed trained or not
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt==None:
        print("START! initially!")
        tf.global_variables_initializer().run()
        epoch_start=0
    else:
        print("STRATING from saved model-"+latest_ckpt)
        saver.restore(sess,latest_ckpt)
        epoch_start = myNumExtractor(latest_ckpt)+1
    
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        
    for iEpoch in range(epoch_start, NUM_EPOCHS) :
        start_time  = time.time()
        #Loop over all batches
        
        db.shuffleTrain_id()
        loss_train = 0.0
        
        for iBatch in range(len_batch):
            
            offset       = (iBatch*BATCH_SIZE)
            #ipdb.set_trace()
            batch_data, batch_labels = db.getPWBatch(isTrain=1, batchStart=offset, batchEnd=offset+BATCH_SIZE, patchSize=patchSize, Aug=1)
            
            _, merged, l_train = sess.run([optimizer, merged_all, loss],feed_dict = {data_node : batch_data, label_node : batch_labels})
            if (iBatch%20==0):
                print("---- processing...EPOCH#%d %d-th (%d-%d): LOSS %.4f" % (iEpoch, iBatch, offset, offset+BATCH_SIZE, l_train))
            loss_train += l_train
            summary_writer.add_summary(merged, iEpoch*len_batch+iBatch)
            
            #end of for loop - for training
        
        print("---1 epoch of train DONE -- time : %.2f min" % (float(time.time()-start_time)/60.0) )
        print("EPOCH(%d - train)--Loss : %.4f" % (iEpoch, loss_train/len_batch))
            
        
        loss_test = 0.0
        psnr_test = 0.0
        for iBatch_test in range(len_batch_test):
            offset       = iBatch_test*BATCH_SIZE
            batch_data, batch_labels = db.getPWBatch(isTrain=0, batchStart=offset, batchEnd=offset+BATCH_SIZE, patchSize=patchSize, Aug=0)
            
            l_test, prediction_test, p_test = sess.run([loss, net_out,psnr],feed_dict={data_node : batch_data, label_node : batch_labels})
            loss_test += l_test
            psnr_test += p_test
        
        print("EPOCH(%d - test )--Loss : %.4f , PSNR : %.4f" % (iEpoch, loss_test/len_batch_test, psnr_test/len_batch_test))
        print("-TOTAL time for 1 epoch : %.2f min" % (float(time.time()-start_time)/60.0) )

        path_saved = saver.save(sess, path.join(ckpt_dir, "model.ckpt"), global_step=iEpoch)
        print("The model saved in file :"+path_saved)
        
        ## saving jpg
        batch_data, batch_labels = db.getBatch(isTrain=0, batchStart=7, batchEnd=10)
        prediction_test = sess.run(net_out, feed_dict={data_node : batch_data, label_node : batch_labels})
        
        ## image save
        np2img_save(prediction_test[0,:,:,:], log_dir+'/test1-result.jpeg')
        np2img_save(batch_data[0,:,:,:], log_dir+'/test1-input.jpeg')
        np2img_save(batch_labels[0,:,:,:], log_dir+'/test1-label.jpeg')
        np2img_save_(prediction_test[0,:,:,:], batch_labels[0,:,:,:],  log_dir+'/test1-error.jpeg')
        
        np2img_save(prediction_test[1,:,:,:], log_dir+'/test1-result.jpeg')
        np2img_save(batch_data[1,:,:,:], log_dir+'/test1-input.jpeg')
        np2img_save(batch_labels[1,:,:,:], log_dir+'/test1-label.jpeg')
        np2img_save_(prediction_test[1,:,:,:], batch_labels[0,:,:,:],  log_dir+'/test1-error.jpeg')
        
        np2img_save(prediction_test[2,:,:,:], log_dir+'/test1-result.jpeg')
        np2img_save(batch_data[2,:,:,:], log_dir+'/test1-input.jpeg')
        np2img_save(batch_labels[2,:,:,:], log_dir+'/test1-label.jpeg')
        np2img_save_(prediction_test[2,:,:,:], batch_labels[0,:,:,:],  log_dir+'/test1-error.jpeg')
        