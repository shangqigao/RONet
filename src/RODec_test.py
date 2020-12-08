'''
Created on 2020/4/21

@author: GaoShangqi
'''
import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from tqdm import tqdm
from RODec_model import RODec

class Tester(RODec):
    def __init__(self):
        RODec.__init__(self, FLAGS)

    def read_image(self, img_path):
        """
        Args;
            HR_path: The path of HR image
            mode: the mode of image
            sigma: The noise level of white Gaussian noise
        Returns:
            observation: The patch corrupted by noise
            groundtruth: The ground truth corresponding oservation
        """
        img = Image.open(img_path)
        size = FLAGS.patch_size // 2
        if FLAGS.out_channel == 1:
            img = img.convert('L')
            img = np.asarray(img, dtype=np.float32)/255.0
            shape = img.shape
            img = img[shape[0]//2-size:shape[0]//2+size,shape[1]//2-size:shape[1]//2+size]
            img = np.expand_dims(img, axis=2)
        elif FLAGS.out_channel == 3:
            img = np.asarray(img, dtype=np.float32)/255.0
            shape = img.shape
            img = img[shape[0]//2-size:shape[0]//2+size,shape[1]//2-size:shape[1]//2+size, :]
        else:
            raise ValueError('Invalid out channels, must be 1 or 3')
        
        np.random.seed(seed=0)
        noise = np.random.normal(size=img.shape)*(FLAGS.sigma/255.0)
        obs = img + noise
        return obs, img

    def run_test(self):
        test_HR_dir = '{}/{}/HR'.format(FLAGS.input_data_dir, FLAGS.dataset)
        psnrs, ssims, durations = [], [], []
        HR_names = sorted(os.listdir(test_HR_dir))

        with tf.Graph().as_default():
            image_pl = tf.placeholder(tf.float32, shape=(1, FLAGS.patch_size, FLAGS.patch_size, FLAGS.out_channel))
            sess = tf.Session()
            if FLAGS.DecMethod == 'RODec':
                Res, ROs = self.test_infer(image_pl)
                output = tf.concat([ROs, Res], axis=0)
                saver = tf.train.Saver()
                saver.restore(sess, FLAGS.RODec_checkpoint)
            else:
                ROs, ROs_sum = self.batch_svd(image_pl, self.num_decomp)
                Res = image_pl - ROs_sum
                output = tf.concat([ROs, Res], axis=0)

            for number in tqdm(range(10)):
                start_time = time.time()
                HR_path = os.path.join(test_HR_dir, HR_names[number])
                ob, gt = self.read_image(HR_path)
                input_ob = np.expand_dims(ob, 0)
                feed_dict = {image_pl : input_ob}
                ob_dec = sess.run(output, feed_dict=feed_dict)
                input_gt = np.expand_dims(gt, 0)
                feed_dict = {image_pl : input_gt}
                gt_dec = sess.run(output, feed_dict=feed_dict)
                duration = time.time() - start_time
                durations.append(duration)
                    
                #compute psnr and ssim
                psnr = []
                ssim = []
                for i in range(self.num_decomp + 1):
                    if self.out_channel == 1:
                        p = PSNR(ob_dec[i,:,:,0], gt_dec[i,:,:,0], data_range=1.0)
                        s = SSIM(ob_dec[i,:,:,0], gt_dec[i,:,:,0], data_range=1.0, multichannel=False)
                    elif self.out_channel == 3:
                        p = PSNR(ob_dec[i], gt_dec[i], data_range=1.0)
                        s = SSIM(ob_dec[i], gt_dec[i], data_range=1.0, multichannel=True) 
                    else:
                        raise ValueError('Invalid out channel, must be 1 or 3')
                    psnr.append(p)
                    ssim.append(s)
                psnrs.append(psnr)
                ssims.append(ssim)
            dur_mean = np.mean(np.array(durations))
            dur_std = np.std(np.array(durations))
            aver_psnr = np.mean(np.array(psnrs), axis=0)
            aver_ssim = np.mean(np.array(ssims), axis=0)
            print('-------------------------------------')
            print('Runtime -> mean: %0.1f  std: %0.2f' % (dur_mean, dur_std))
            print('-------------------------------------')
            for i in range(self.num_decomp):
                print('Average PSNR for X%d : %0.2f' % (i, aver_psnr[i]))
            print('Average PSNR for E%d : %0.2f' % (self.num_decomp, aver_psnr[self.num_decomp]))
            print('-------------------------------------')
            for i in range(self.num_decomp):
                print('Average SSIM for X%d : %0.4f' % (i, aver_ssim[i]))
            print('Average SSIM for E%d : %0.4f' % (self.num_decomp, aver_ssim[self.num_decomp]))
        

def main(_):
    test = Tester()
    test.run_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DecMethod', choices=['RODec', 'SVD'], required=True,
                        help='Method of rank-one decomposition'
                        )
    parser.add_argument('--train_mode', choices=['supervised', 'unsupervised'], 
                        default='unsupervised',
                        help='Method of training RO decomposition network'
                        )
    parser.add_argument('--num_ROPs', type=int, default=3,
                        help='Numer of RO projections'
                        )
    parser.add_argument('--patch_size', type=int, default=1024,
                        help='Size of cropped pathches'
                        )
    parser.add_argument('--dataset', type=str, default='DIV2K_mild',
                        help='Test dataset'
                        )
    parser.add_argument('--input_data_dir', type=str, default='./data',
                        help='Directory of test datasets'
                        )
    parser.add_argument('--RODec_checkpoint', type=str, default='./model',
                        help='Path of pre-trained RODec model'
                        )
    parser.add_argument('--out_channel', type=int, default=1,
                        help='output channels, 1 for grayscale and 3 for rgb'
                        )
    parser.add_argument('--sigma', type=int, default=0,
                        help='Noise level'
                        )
    parser.add_argument('--GPU_ids', type=str, default = '0',
                        help = 'Ids of GPUs'
                        )
    FLAGS, unparsed = parser.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(FLAGS.GPU_ids)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) 
    

