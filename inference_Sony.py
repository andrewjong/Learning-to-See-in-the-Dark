# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37

import os, scipy, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import argparse

# disable TF debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


parser = argparse.ArgumentParser(description="Run inference to brighten the Sony raw photos.")
parser.add_argument("input", help="Path to Sony raw input image, .ARW file")
parser.add_argument(
    "-o", "--out_dir", 
    help="Output directory to store converted images (default: ./result_Sony)", 
    default=os.path.join(".", "result_Sony")
)
parser.add_argument(
    "-r", "--ratio", 
    help="Ratio to brighten images by. Set this value to \
    DESIRED_EXPOSURE/TAKEN_EXPOSURE. Note, values are capped at 300 (default: 100).", 
    type=float, default=100
)
parser.add_argument(
    "-m", "--model_dir", 
    help="Directory of the TensorFlow model to use (default: ./checkpoint)", 
    default=os.path.join(".", "checkpoint")
)

args = parser.parse_args()

checkpoint_dir = os.path.join(args.model_dir, 'Sony')
result_dir = args.out_dir


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
out_image = network(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print(('Loaded model from ' + ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

print("Input path:", args.input)
print("Brighten ratio set to:", args.ratio)
in_path = args.input
in_fname = os.path.basename(in_path)
fname_no_extension = os.path.splitext(in_fname)[0]
ratio = min(args.ratio, 300)

print("Reading raw photo...")
raw = rawpy.imread(in_path)
input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
input_full = np.minimum(input_full, 1.0)

print("Converting...")
output = sess.run(out_image, feed_dict={in_image: input_full})
output = np.minimum(np.maximum(output, 0), 1)
print("Converted!")

# squeeze the batch dimension, so we get just 3 channels
output = output[0, :, :, :]

# save the image
out_path = os.path.join(result_dir, f"{fname_no_extension}_out.png")
print(f"Saving output to {out_path}")
scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(out_path)
print(f"Output saved! Done.")
