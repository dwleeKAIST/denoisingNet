from os import path, makedirs

import numpy as np
import tensorflow as tf
import time 


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
        print("STARTING from saved model-"+latest_ckpt)
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
        batch_data, batch_labels = db.getABatch(isTrain=0, batchStart=7)
        prediction_test = sess.run(net_out, feed_dict={data_node : batch_data, label_node : batch_labels})
        db.np2img_save(batch_data, prediction_test, batch_labels, log_dir, save_str='test1')
        
        batch_data, batch_labels = db.getABatch(isTrain=0, batchStart=8)
        prediction_test = sess.run(net_out, feed_dict={data_node : batch_data, label_node : batch_labels})
        db.np2img_save(batch_data, prediction_test, batch_labels, log_dir, save_str='test2')
        
        batch_data, batch_labels = db.getABatch(isTrain=0, batchStart=9)
        prediction_test = sess.run(net_out, feed_dict={data_node : batch_data, label_node : batch_labels})
        db.np2img_save(batch_data, prediction_test, batch_labels, log_dir, save_str='test3')
        