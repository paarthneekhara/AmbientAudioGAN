import tensorflow as tf
from AudioModel.util import override_model_attrs
import EncDecModel
from AudioModel.model import Model, Modes

ckpt = 16384
a = tf.placeholder('float32', (32, 131072, 1, 1))
b = tf.placeholder('float32', (32, 131072, 1, 1))
model = EncDecModel.WaveAE(mode = Modes.INFER)
oberrides =  "objective=l1,batchnorm=False,train_batch_size=32,alpha=100.0,enc_length=16,stride=4,kernel_len=25,subseq_len=131072"
model, summary = override_model_attrs(model, oberrides)

model.build_inference(a, b)

# with tf.variable_scope('AE'):
#   with tf.variable_scope('E'):
#     enc = waveAE.WaveEncoderFactor256(batchnorm=False)
#     E_x = enc(a, training=False)


# e_flat = tf.reshape(E_x, [32, -1])
# rd_tensor = tf.layers.dense(e_flat, 5)
# model_dir_path = "/data2/paarth/TrainDir/WaveAE/WaveAEsc09_l1batchnormFalse/eval_sc09_valid"
# e_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='AE/E')
# print (e_vars)
# saver = tf.train.Saver(var_list=e_vars)
# sess = tf.InteractiveSession()

# saver.restore(sess, '{}/best_valid_l2-{}'.format(model_dir_path, ckpt))
