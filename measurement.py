import tensorflow as tf

def block_patch(batch, patch_size = 0):
    shape = batch.get_shape().as_list()
    
    patch = tf.zeros([patch_size, shape[-1]], dtype=tf.float32)
    
    res = []
    paddings = []
    for idx in range(0,shape[0]):
        rand_num = tf.random_uniform([1], minval=0, maxval=shape[1]-patch_size, dtype=tf.int32)
        w_ = rand_num[0]
        # print w_
        padding = [[w_, shape[1]-w_- patch_size ], [0,0]]
        padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
        
        paddings.append(padded)

        mask_clipping = (1 - padded) * 100.
        # 100 where we want to clip audio and 0 everywhere else
        clipped_audio = tf.clip_by_value(batch[idx] + mask_clipping, 
            clip_value_min = -1, 
            clip_value_max = 1)
        
        res.append(clipped_audio)
        

    paddings = tf.stack(paddings)
    res = tf.stack(res)

    res = tf.reshape(res,shape)
    paddings = tf.reshape(paddings, shape)

    return res, paddings


def drop_patches(batch, patch_size = 1000, drop_prob = 0.5):
    shape = batch.get_shape().as_list()

    res = []
    paddings = []
    
    mask_flags = tf.random_uniform(shape=[ shape[0], int(shape[1]/patch_size)])
    mask_flags = tf.cast(mask_flags >= drop_prob, dtype = tf.float32)
    # print "mask flags", mask_flags

    for idx in range(0,shape[0]):
        mask_tensor = []
        for pn in range(0, int(shape[1]/patch_size)):
            pad = tf.ones([patch_size,1], dtype = tf.float32)
            pad = pad * mask_flags[idx][pn]
            mask_tensor.append(pad)
        
        mask_tensor = tf.concat(mask_tensor, axis = 0)
        # print "mask tensor", mask_tensor
        mask_tensor_shape = int(shape[1]/patch_size) * patch_size
        pad_size = [[0, shape[1]-mask_tensor_shape], [0,0]]
        mask_tensor = tf.pad(mask_tensor, pad_size, "CONSTANT", constant_values=1)
        # print "mask tensor final", mask_tensor
        paddings.append(mask_tensor)

        mask_clipping = (1 - mask_tensor) * 100.
        # 100 where we want to clip audio and 0 everywhere else
        clipped_audio = tf.clip_by_value(batch[idx] + mask_clipping, 
            clip_value_min = -1, 
            clip_value_max = 1)

        res.append(clipped_audio)
        

    paddings = tf.stack(paddings)
    res = tf.stack(res)

    res = tf.reshape(res,shape)
    paddings = tf.reshape(paddings, shape)

    return res, paddings




def drop_2patch(batch, patch_size = 0):
    shape = batch.get_shape().as_list()
    
    patch = tf.zeros([patch_size, shape[-1]], dtype=tf.float32)
    
    res = []
    paddings = []
    for idx in range(0,shape[0]):
        rand_num = tf.random_uniform([2], minval=0, maxval=(int(shape[1]/2)-patch_size), dtype=tf.int32)
        #rand_num2 = tf.random_uniform([1], minval=int(shape[1]/2), maxval=shape[1]-patch_size, dtype=tf.int32)
        w_ = rand_num[0]
        w2_ = rand_num[1]
        # print w_
        padding1 = [[w_, int(shape[1]/2)-w_- patch_size ], [0,0]]
        padding2 = [[w2_, int(shape[1]/2)-w2_- patch_size ], [0,0]]
       
        padded1 = tf.pad(patch, padding1, "CONSTANT", constant_values=1)
        padded2 = tf.pad(patch, padding2, "CONSTANT", constant_values=1)
        padded = tf.concat([padded1,padded2], axis = 0)
        
        paddings.append(padded)
        res.append(tf.multiply(batch[idx], padded))
        

    paddings = tf.stack(paddings)
    res = tf.stack(res)

    res = tf.reshape(res,shape)
    paddings = tf.reshape(paddings, shape)

    return res, paddings

def drop_audio(batch, drop_prob = 0.6):
    noise_shape = batch.get_shape().as_list()
    mask = tf.random_uniform(shape=noise_shape)
    mask = tf.cast(mask >= drop_prob, dtype = tf.float32)
    theta_val = tf.ones(shape=noise_shape)
    theta_val = theta_val * mask

    return (theta_val * batch), theta_val


def add_gaussian_noise(batch, stddev = 0.2):
    noise = (1/5.0)*tf.random_normal(shape=tf.shape(batch), mean=0.0, stddev=stddev, dtype=tf.float32)
    return batch + noise, None

def convolve_kernel(batch, kernel, std = 0.2, no_noise = False):
    kernel = tf.cast(kernel, dtype = tf.float32)
    kernel = tf.expand_dims( tf.expand_dims(kernel, axis = -1), axis = -1)
    noise =  (1/100.0)*tf.random_normal(shape=tf.shape(batch), mean =0.0, stddev=std,dtype=tf.float32)
    if no_noise:
        noise = 0
    convsignal = tf.nn.conv1d(batch, kernel, stride = 1, padding = "SAME")
    return convsignal + noise, convsignal

def patch_noise(batch, patch_size = 6000,std = 0.2, snr = 5.0):
    shape = batch.get_shape().as_list()
    
    patch = tf.zeros([patch_size, shape[-1]], dtype=tf.float32)
    noise =  (1/snr)*tf.random_normal(shape=tf.shape(batch), mean =0.0, stddev=std,dtype=tf.float32)

    
    res = []
    paddings = []
    for idx in range(0,shape[0]):
        rand_num = tf.random_uniform([1], minval=0, maxval=shape[1]-patch_size, dtype=tf.int32)
        w_ = rand_num[0]
        # print w_
        padding = [[w_, shape[1]-w_- patch_size ], [0,0]]
        padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
        padded = 1-padded
        
        paddings.append(padded)
        res.append(tf.multiply(noise[idx], padded) + batch[idx])
        

    paddings = tf.stack(paddings)
    res = tf.stack(res)

    res = tf.reshape(res,shape)
    paddings = tf.reshape(paddings, shape)

    return res, paddings