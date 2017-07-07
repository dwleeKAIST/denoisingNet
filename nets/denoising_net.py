import tensorflow as tf
import tensorflow.contrib.layers as li

##

def w_var(shape, name):
    return tf.get_variable("W"+name, shape=shape,initializer= li.xavier_initializer())
def b_var(shape):
    return tf.Variable( tf.constant(0.0, shape=shape) )
def conv2d(x, shape, name):
    return tf.nn.bias_add( tf.nn.conv2d(x, w_var(shape, name), strides=[1,1,1,1], padding='SAME'), b_var([shape[3]]))
def BN(x):
    return li.batch_norm( x, decay=0.999, center= True, scale=False, epsilon=0.001, activation_fn=None, param_initializers=None)


N_FEATURES  = 320

def net(data, residual_tag, IMG_SIZE_CH = 12):
    # this is the model -Beyond deep residual learnining for image restoration - W.Bae arXiv:1611.06345v4, Jun 2017
    # edited by DW.Lee
    
    
    out_green1  = tf.nn.relu(     conv2d(data,       [3,3,IMG_SIZE_CH, N_FEATURES],"1") )
    out_purple1 = tf.nn.relu( BN( conv2d(out_green1, [3,3, N_FEATURES, N_FEATURES],"2") ) )
    
    # module Unit #1
    out_11p    = tf.nn.relu( BN( conv2d(out_purple1,[3,3, N_FEATURES, N_FEATURES],"11") ) )
    out_12p    = tf.nn.relu( BN( conv2d(out_11p,    [3,3, N_FEATURES, N_FEATURES],"12") ) )
    out_13y    =             BN( conv2d(out_12p,    [3,3, N_FEATURES, N_FEATURES],"13") )
    out_14b    = tf.nn.relu( tf.add(out_purple1, out_13y, "skip1")) # skipped connection
    
    # module Unit #2
    out_21p    = tf.nn.relu( BN( conv2d(out_14b,   [3,3, N_FEATURES, N_FEATURES],"21") ) )
    out_22p    = tf.nn.relu( BN( conv2d(out_21p,   [3,3, N_FEATURES, N_FEATURES],"22") ) )
    out_23y    =             BN( conv2d(out_22p,   [3,3, N_FEATURES, N_FEATURES],"23") )
    out_24b    = tf.nn.relu( tf.add(out_14b, out_23y, "skip2")   ) # skipped connection
    
    # module Unit #3
    out_31p    = tf.nn.relu( BN( conv2d(out_24b,   [3,3, N_FEATURES, N_FEATURES],"31") ) )
    out_32p    = tf.nn.relu( BN( conv2d(out_31p,   [3,3, N_FEATURES, N_FEATURES],"32") ) )
    out_33y    =             BN( conv2d(out_32p,   [3,3, N_FEATURES, N_FEATURES],"33") )
    out_34b    = tf.nn.relu( tf.add(out_24b, out_33y, "skip3")   ) # skipped connection
    
    # module Unit #4
    out_41p    = tf.nn.relu( BN( conv2d(out_34b,   [3,3, N_FEATURES, N_FEATURES],"41") ) )
    out_42p    = tf.nn.relu( BN( conv2d(out_41p,   [3,3, N_FEATURES, N_FEATURES],"42") ) )
    out_43y    =             BN( conv2d(out_42p,   [3,3, N_FEATURES, N_FEATURES],"43") )
    out_44b    = tf.nn.relu( tf.add(out_34b, out_43y, "skip4")   ) # skipped connection
    
    # module Unit #5
    out_51p    = tf.nn.relu( BN( conv2d(out_44b,   [3,3, N_FEATURES, N_FEATURES],"51") ) )
    out_52p    = tf.nn.relu( BN( conv2d(out_51p,   [3,3, N_FEATURES, N_FEATURES],"52") ) )
    out_53y    =             BN( conv2d(out_52p,   [3,3, N_FEATURES, N_FEATURES],"53") )
    out_54b    = tf.nn.relu( tf.add(out_44b, out_53y, "skip5")   ) # skipped connection
    
    #
    out_6b    = tf.nn.relu( BN(  conv2d( out_54b,   [3,3, N_FEATURES, N_FEATURES],"6") ) )
    out_7p    = tf.nn.relu( BN(  conv2d( out_6b,   [3,3, N_FEATURES, N_FEATURES],"7") ) )
    
    # Last layer
    if residual_tag:
        # do something
        out_      = conv2d(out_7p, [3,3,N_FEATURES, IMG_SIZE_CH],"8")
        out       = tf.add(out_, data, "skip0")
    else:
        out       = conv2d(out_7p, [3,3,N_FEATURES, IMG_SIZE_CH],"8")
   
    return out