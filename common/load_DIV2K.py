'''
Created on Apr 30, 2020

@author: gsq
'''
import os, cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class Dataloader:
    def __init__(self, FLAGS):
        #if True, use data augmentation
        self.augment = FLAGS.augment
        self.task = FLAGS.task
        self.upscale = FLAGS.upscale
        self.out_channel = FLAGS.out_channel
        self.sigma = FLAGS.sigma
        #if True, the noise level ranges in [0, sigma]
        self.range = FLAGS.range
        self.TRAIN_LR_DIR = {'DEN': 'DIV2K/DIV2K_HR/DIV2K_HR_train',
                             #'DEN': 'BSDS500/train',
                             'BiSR': 'DIV2K/DIV2K_LR_bicubic/DIV2K_LR_train/X{}'.format(self.upscale),
                             'ReSR': 'DIV2K/DIV2K_LR_mild/DIV2K_LR_train/X{}'.format(self.upscale)
        }[self.task]
        
        self.TRAIN_HR_DIR = 'DIV2K/DIV2K_HR/DIV2K_HR_train'
        #self.TRAIN_HR_DIR = 'BSDS500/train'

        self.VALID_LR_DIR = {'DEN': 'DIV2K/DIV2K_HR/DIV2K_HR_valid',
                             'BiSR': 'DIV2K/DIV2K_LR_bicubic/DIV2K_LR_valid/X{}'.format(self.upscale),
                             'ReSR': 'DIV2K/DIV2K_LR_mild/DIV2K_LR_valid/X{}'.format(self.upscale)
        }[self.task]
        
        self.VALID_HR_DIR = 'DIV2K/DIV2K_HR/DIV2K_HR_valid'
         
        self.train_batch_size = FLAGS.train_batch_size 
        self.valid_batch_size = FLAGS.valid_batch_size

        self.train_patch_size = FLAGS.train_patch_size
        self.valid_patch_size = FLAGS.valid_patch_size
        self.train_label_size = self.train_patch_size*self.upscale
        self.valid_label_size = self.valid_patch_size*self.upscale
        
        self.num_data_threads = FLAGS.threads
        self.shuffle_buffer_size  = 800 #DIV2K-800, Flickr2K-2650, BSDS-400

        #The maximal sliding steps for aligning LR and HR paires
        self.align_step = {'DEN': 0, 'DEB': 0, 'BiSR': 0, 'ReSR': 10}[self.task]


    def extract_image(self, input_data_dir, mode):
        '''The function to extract images
        Arg:
            input_data_dir: the dir of dataset
            mode: 'train' or 'validation' or 'test'
        return:
            datatset: A tensor of size [2, num_img, height, width, channels]
        '''
        lr_dir = {'train': os.path.join(input_data_dir, self.TRAIN_LR_DIR),
                  'valid': os.path.join(input_data_dir, self.VALID_LR_DIR),
                  }[mode]
        hr_dir = {'train': os.path.join(input_data_dir, self.TRAIN_HR_DIR),
                  'valid': os.path.join(input_data_dir, self.VALID_HR_DIR),
                  }[mode]
        if not tf.gfile.Exists(lr_dir) or not tf.gfile.Exists(hr_dir):
            raise ValueError(f'{lr_dir} or {hr_dir} does not exit.')

        def list_files(d):
            files = sorted(os.listdir(d))
            files = [os.path.join(d, f) for f in files]
            return files
    
        lr_files = list_files(lr_dir)
        hr_files = list_files(hr_dir)
        num_files = len(hr_files)
        print(f'Total {mode} data: {num_files}')
        dataset = tf.data.Dataset.from_tensor_slices((lr_files, hr_files))
    
        def _read_image(lr_file, hr_file):
            print('Reading images!')
            lr_image = tf.image.decode_image(tf.read_file(lr_file), channels=self.out_channel)
            hr_image = tf.image.decode_image(tf.read_file(hr_file), channels=self.out_channel)

            def _alignment(lr_image, hr_image, step=self.align_step):
                lr_shape = tf.shape(lr_image)
                height = lr_shape[0] - 2*step
                width = lr_shape[1] - 2*step
                hr = tf.expand_dims(hr_image, 0)
                bicubic_lr = tf.image.resize_bicubic(hr, [lr_shape[0], lr_shape[1]])[0]
                initial_lr = tf.slice(bicubic_lr, [step, step, 0], [height, width, -1])
                lr = tf.slice(lr_image, [step, step, 0], [height, width, -1])
                lr = tf.cast(lr, tf.float32)
                mse = tf.reduce_mean(tf.square(lr - initial_lr))
                shift_up = 0
                shift_left = 0
                for row in range(-step, step):
                    for column in range(-step, step):
                        new_lr = tf.slice(bicubic_lr, [step + row, step + column, 0], [height, width, -1])
                        new_lr = tf.cast(new_lr, tf.float32)
                        new_mse = tf.reduce_mean(tf.square(lr - new_lr))
                        def _true_fn():
                            return row, column, new_mse
                        def _false_fn():
                            return shift_up, shift_left, mse
                        shift_up, shift_left, mse = tf.cond(new_mse < mse, true_fn=_true_fn, false_fn=_false_fn)
                return shift_up, shift_left
            
            #In realistic case, data is preprocessed using alignment
            if self.task == 'ReSR':
                shift_up, shift_left = _alignment(lr_image, hr_image)
            else:
                shift_up, shift_left = 0, 0

            return lr_image, hr_image, shift_up, shift_left
    
        dataset = dataset.map(_read_image,
                              num_parallel_calls=self.num_data_threads,
                              )
        dataset = dataset.cache()
        return dataset

    def extract_batch(self, dataset, mode):
        '''The function to extract sub-images for training
        '''
        if mode == 'train':
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.repeat()
    
        def _preprocess(lr, hr, shift_up, shift_left):
            def _flip_rotation(values, fn):
                def _done():
                    return [fn(v) for v in values]
                def _notdone():
                    return values
    
                pred = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
                values = tf.cond(pred, _done, _notdone)
                return values

            def _img_noise(img): 
                if self.range and mode == 'train':
                    sigma = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)*self.sigma
                else:
                    sigma = self.sigma
                noise = tf.random_normal(shape=tf.shape(img))*(sigma/255.)
                img_noise = img + noise
                return img_noise
            
            lr_shape = tf.shape(lr)
            if mode == 'train':
                lr_up = tf.random_uniform(
                    shape=[],
                    minval=self.align_step,
                    maxval=lr_shape[0] - self.train_patch_size - self.align_step,
                    dtype=tf.int32)
                lr_left = tf.random_uniform(
                    shape=[],
                    minval=self.align_step,
                    maxval=lr_shape[1] - self.train_patch_size - self.align_step,
                    dtype=tf.int32)
                lr = tf.slice(lr, [lr_up, lr_left, 0], [self.train_patch_size, self.train_patch_size, -1])
                hr_up = (lr_up + shift_up)*self.upscale
                hr_left = (lr_left + shift_left)*self.upscale
                hr = tf.slice(hr, [hr_up, hr_left, 0], [self.train_label_size, self.train_label_size, -1])
            else:
                lr_up = (lr_shape[0] - self.valid_patch_size) // 2
                lr_left = (lr_shape[1] - self.valid_patch_size) // 2
                lr = tf.slice(lr, [lr_up, lr_left, 0], [self.valid_patch_size, self.valid_patch_size, -1])
                hr_up = (lr_up + shift_up)*self.upscale
                hr_left = (lr_left + shift_left)*self.upscale
                hr = tf.slice(hr, [hr_up, hr_left, 0], [self.valid_label_size, self.valid_label_size, -1])
    
            if mode == 'train' and self.augment:
                lr, hr = _flip_rotation([lr, hr], tf.image.flip_left_right)
                lr, hr = _flip_rotation([lr, hr], tf.image.flip_up_down)
                lr, hr = _flip_rotation([lr, hr], tf.image.rot90)
    
            lr = tf.image.convert_image_dtype(lr, tf.float32)
            hr = tf.image.convert_image_dtype(hr, tf.float32)
            
            if self.sigma > 0:
                lr = _img_noise(lr)

            return lr, hr
        
        dataset = dataset.map(
            _preprocess,
            num_parallel_calls=self.num_data_threads,
            )
        batch_size = {
            'train': self.train_batch_size,
            'valid': self.valid_batch_size,
            }[mode]
        drop_remainder = {
            'train': True,
            'valid': True,
            }[mode]
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        return dataset
    
    def generate_dataset(self, input_data_dir):

        train_dataset = self.extract_image(input_data_dir, 'train')
        valid_dataset = self.extract_image(input_data_dir, 'valid')
        train_dataset = self.extract_batch(train_dataset, 'train')
        valid_dataset = self.extract_batch(valid_dataset, 'valid')
        return train_dataset, valid_dataset
