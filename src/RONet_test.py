'''
Created on Apr 30, 2020

@author: Shangqi Gao
'''
import sys
sys.path.append('../')
import os 
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage import io, color
from PIL import Image
from common.eval_noshift import multi_compute as measure_noshift
from common.eval_shift import multi_compute as measure_shift
from skimage.metrics import peak_signal_noise_ratio as PSNR

from RONet_model import RONet

class Tester(RONet):
    def __init__(self):
        RONet.__init__(self, FLAGS)

    def read_image(self, lr_path, gt_path):
        """
        Args;
            lr_path: The path of a lr image
            gt_path: The path of ground truth
        Returns:
            lr_img: an image, narray, float32
            gt_img: a label, narray, float32
        """
        if FLAGS.task == 'DEN':
            gt_img = io.imread(gt_path).astype(np.float32) / 255.
            if FLAGS.out_channel == 1:
                gt_img = np.expand_dims(gt_img, axis=2)
            np.random.seed(seed=1234)
            noise = np.random.normal(size=gt_img.shape)*(FLAGS.sigma/255.)
            lr_img = gt_img + noise
        else:
            lr_img = io.imread(lr_path) / 255.
            if len(lr_img.shape) == 2:
                lr_img = np.stack([lr_img]*3, axis=2)
            gt_img = io.imread(gt_path)
            if len(gt_img.shape) == 2:
                gt_img = np.stack([gt_img]*3, axis=2)
            
        return lr_img, gt_img
    
    def flip(self, image):
        images = [image]
        images.append(image[::-1, :, :])
        images.append(image[:, ::-1, :])
        images.append(image[::-1, ::-1, :])
        images = np.stack(images)
        return images
    
    def mean_of_flipped(self, images):
        image = (images[0] + images[1, ::-1, :, :] + images[2, :, ::-1, :] +
                 images[3, ::-1, ::-1, :])*0.25
        return image
    
    def rotation(self, images):
        return np.swapaxes(images, 1, 2)
 
    def run_test(self):
        test_lr_dir = {'DEN': '{}/{}'.format(FLAGS.input_data_dir, FLAGS.dataset),
                       'BiSR': '{}/{}/LR_bicubic/X{}'.format(FLAGS.input_data_dir, FLAGS.dataset, FLAGS.upscale),
                       'ReSR': '{}/{}/LR_mild/X{}'.format(FLAGS.input_data_dir, FLAGS.dataset, FLAGS.upscale)
                        }[FLAGS.task]
        test_gt_dir = {'DEN': '{}/{}'.format(FLAGS.input_data_dir, FLAGS.dataset),
                       'BiSR': '{}/{}/HR'.format(FLAGS.input_data_dir, FLAGS.dataset),
                       'ReSR': '{}/{}/HR'.format(FLAGS.input_data_dir, FLAGS.dataset)
                       }[FLAGS.task]
        img_mode = 'Gray' if FLAGS.out_channel == 1 else 'RGB'
        test_sr_dir = '{}/{}_RONet_{}_{}_x{}_sigma{}'.format(FLAGS.save_dir, FLAGS.dataset, img_mode, FLAGS.task, FLAGS.upscale, FLAGS.sigma)
        
        if tf.gfile.Exists(test_sr_dir):
            tf.gfile.DeleteRecursively(test_sr_dir)
        tf.gfile.MakeDirs(test_sr_dir)
        
        with tf.Graph().as_default():
            image_pl = tf.placeholder(tf.float32, shape=(1, 64, 64, FLAGS.out_channel))
            output = self.test_infer(image_pl)

            RONet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'decomp') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'RORec')
            RONet_saver = tf.train.Saver(RONet_vars)
            sess = tf.Session()
            RONet_saver.restore(sess, FLAGS.RONet_checkpoint)
            
            lr_names = sorted(os.listdir(test_lr_dir))
            gt_names = sorted(os.listdir(test_gt_dir))
            start = time.time()
            for number in tqdm(range(len(gt_names))):
                lr_path = os.path.join(test_lr_dir, lr_names[number])
                gt_path = os.path.join(test_gt_dir, gt_names[number])
                gt_name = gt_names[number]
                image, label = self.read_image(lr_path, gt_path)
                shape = image.shape
                if FLAGS.ensemble:
                    image_pl0 = tf.placeholder(tf.float32, shape=(4, shape[0], shape[1], shape[2]))
                    image_pl1 = tf.placeholder(tf.float32, shape=(4, shape[1], shape[0], shape[2]))
                    output0 = self.test_infer(image_pl0)
                    output1 = self.test_infer(image_pl1)
                    input_images = self.flip(image)
                    feed_dict = {image_pl0 : input_images,}
                    output_image0 = self.mean_of_flipped(sess.run(output0, feed_dict))
                    feed_dict = {image_pl1 : self.rotation(input_images)}
                    output_image1 = self.mean_of_flipped(self.rotation(sess.run(output1, feed_dict)))
                    output_image = (output_image0 + output_image1)*0.5
                else:
                    image_pl = tf.placeholder(tf.float32, shape=(1, shape[0], shape[1], shape[2])) 
                    output = self.test_infer(image_pl)
                    input_images = np.expand_dims(image, 0)
                    feed_dict = {image_pl : input_images}
                    output_image = sess.run(output, feed_dict)[0]
                sr_img = np.around(output_image*255.0).astype(np.uint8)
                io.imsave(os.path.join(test_sr_dir, gt_name), np.squeeze(sr_img))
            duration = time.time() - start
            mean_dura = duration / len(gt_names)
            print(f'Avg_reconstruction_time_per_image: {mean_dura:0.2f}')
        if FLAGS.task == 'ReSR':
            measure_shift(test_gt_dir, test_sr_dir)
        elif FLAGS.task == 'BiSR':
            measure_noshift(test_gt_dir, test_sr_dir, FLAGS.upscale, 'ycbcr')
        else:
            measure_noshift(test_gt_dir, test_sr_dir, FLAGS.upscale, 'rgb')
            
def main(_):
    test = Tester()
    test.run_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['DEN', 'BiSR', 'ReSR'], required=True,
                        help='Image restoration task'
                        )
    parser.add_argument('--dataset', type=str, default='Set5',
                        help='Test dataset'
                        )
    parser.add_argument('--input_data_dir', type=str, default='./data',
                        help='Directory of test datasets'
                        )
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory of saving reconstructions'
                        )
    parser.add_argument('--RONet_checkpoint', type=str, default='./model',
                        help='Path of pre-trained RONet model'
                        )
    parser.add_argument('--net_type', choices=['baseline', 'net_den', 'net_sr'], required=True,
                        help='Which network structure'
                        )
    parser.add_argument('--deep_scale', type=int, default=1,
                        help='which is used to increase the number of deep features'
                        )
    parser.add_argument('--depth_RODec', type=int, default=3,
                        help='the number of RO projections in RODec'
                        )
    parser.add_argument('--depth_RecROs', type=int, default=3,
                        help='the number of residual blocks in RecROs'
                        )
    parser.add_argument('--depth_RecRes', type=int, default=6,
                        help='the number of RO projections in RecRes'
                        )
    parser.add_argument('--depth_RecFus', type=int, default=3,
                        help='the number of RO projections in RecFus'
                        )
    parser.add_argument('--out_channel', type=int, default=1,
                        help='output channels, 1 for grayscale and 3 for rgb'
                        )
    parser.add_argument('--regularizer', choices=['l1', 'l2', 'None'], default='None',
                        help='Which regularizer is used to restrict kernels'
                        )
    parser.add_argument('--sigma', type=int, default=0,
                        help='Noise level'
                        )
    parser.add_argument('--upscale', type=int, default=1,
                        help='upscaling factor'
                        )
    parser.add_argument('--ensemble', action='store_true',
                        help='If set, use data ensemble'
                        )
    parser.add_argument('--GPU_ids', type=str, default = '0',
                        help = 'Ids of GPUs'
                        )
    FLAGS, unparsed = parser.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(FLAGS.GPU_ids)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
