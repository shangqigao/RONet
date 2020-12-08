'''
Created on May 9, 2019

@author: Shangqi Gao
This is an implementation of rank-one network (RONet)
'''
import tensorflow as tf
import numpy as np
from src import vgg
import tensorflow.contrib.slim as slim

class RONet:
    def __init__(self, FLAGS):
        #The upscaling factor: 4 for super-resolution and 1 for denoising
        self.upscale = FLAGS.upscale
        #The number of output channel
        self.out_channel = FLAGS.out_channel
        self.task = FLAGS.task
        #Used to increase the number of deep features, 1 for 'sr', 
        #16 for RGB 'den', 48 for grayscale 'den' 
        self.deep_scale = FLAGS.deep_scale
        #The size of convolutional kernel
        self.kernel_size = 3
        self.num_decomp = FLAGS.depth_RODec
        self.num_recons = FLAGS.depth_RecROs
        self.num_err = FLAGS.depth_RecRes
        self.num_output = FLAGS.depth_RecFus
        self.filter_recons, self.filter_err = self.set_filters(FLAGS.net_type)
        self.task = FLAGS.task
        #RO decompoaition method: 'SVD' or 'RODec'
        self.DecMethod = 'RODec'
        self.res_scale = 1.0
        #self.initializer = tf.variance_scaling_initializer(1.0, 'fan_avg', 'truncated_normal')
        self.initializer = tf.glorot_uniform_initializer()
        self.regularizer = self.set_regularizer(FLAGS.regularizer)
        #VGG19 checkpoints
        self.content_layer = 'vgg_19/conv4/conv4_2'

    def set_filters(self, net_type):
        if net_type == 'baseline':
            filter1, filter2 = [48, 48], [64, 64]
        elif net_type == 'net_den':
            filter1, filter2 = [48, 96], [64, 128]
        elif net_type == 'net_sr':
            filter1, filter2 = [48, 192], [64, 256]
        else:
            raise ValueError(f'Invalid network type: {net_type}')
        return filter1, filter2

    def set_regularizer(self, reg_type):
        if reg_type == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(scale=1e-8)
        elif reg_type == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-8)
        else:
            regularizer = None
        return regularizer

    def perceptual_loss(self, img_batch, label_batch):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points_img = vgg.vgg_19(img_batch, num_classes=None, is_training=False)
            _, end_points_label = vgg.vgg_19(label_batch, num_classes=None, is_training=False)
        content_img = end_points_img[self.content_layer]
        content_label = end_points_label[self.content_layer]
        content_loss = tf.losses.absolute_difference(content_label, content_img)
        return content_loss
        

    def outer_product(self, left_batch, right_batch):
        '''Compute the outer product of two four-way tensors
        '''
        left_batch = tf.transpose(left_batch, [0, 3, 1, 2])
        left_shape = left_batch.shape.as_list()
        left_batch = tf.reshape(left_batch, [-1, left_shape[2], left_shape[3]])
        right_batch = tf.transpose(right_batch, [0, 3, 1, 2])
        right_shape = right_batch.shape.as_list()
        right_batch = tf.reshape(right_batch, [-1, right_shape[2], right_shape[3]])
        outerProduct = tf.matmul(left_batch, right_batch)
        outerProduct = tf.reshape(outerProduct, [left_shape[0], left_shape[1], left_shape[2], right_shape[3]])
        outerProduct = tf.transpose(outerProduct, [0, 2, 3, 1])    
        return outerProduct

    def batch_svd(self, img_batch, num):
        '''compute svd of a batch of images
        '''
        img_shape= img_batch.shape.as_list()
        img_batch = tf.transpose(img_batch, [0, 3, 1, 2])
        img_batch = tf.reshape(img_batch, [-1, img_shape[1], img_shape[2]])
        sigma_batch, U_batch, V_batch = tf.linalg.svd(img_batch)
        eigenImg_batch = tf.expand_dims(tf.expand_dims(sigma_batch[:, 0], axis=1), axis=2)*tf.matmul(tf.expand_dims(U_batch[:, :, 0], axis=2), 
                                        tf.expand_dims(V_batch[:, :, 0], axis=2), transpose_b=True)
        eigenImg_batch = tf.reshape(eigenImg_batch, [img_shape[0], img_shape[3], img_shape[1], img_shape[2]])
        eigenImg_batch = tf.transpose(eigenImg_batch, [0, 2, 3, 1])
        lowrank_image = eigenImg_batch
        for i in range(1, num):
            eigenImg = tf.expand_dims(tf.expand_dims(sigma_batch[:, i], axis=1), axis=2)*tf.matmul(tf.expand_dims(U_batch[:, :, i], axis=2), 
                                      tf.expand_dims(V_batch[:, :, i], axis=2), transpose_b=True)
            eigenImg = tf.reshape(eigenImg, [img_shape[0], img_shape[3], img_shape[1], img_shape[2]])
            eigenImg = tf.transpose(eigenImg, [0, 2, 3, 1])
            lowrank_image += eigenImg
            eigenImg_batch = tf.concat([eigenImg_batch, eigenImg], axis=0)
        return eigenImg_batch, lowrank_image
        
    def rankone_block(self, left_img_batch, right_img_batch, is_training):
        left_img_batch = tf.layers.conv2d(left_img_batch,
                                     filters=256, kernel_size=(1, self.kernel_size), strides=1,
                                     padding='SAME', use_bias=True, trainable=is_training, name='left_conv0')
        left_img_batch = tf.nn.relu(left_img_batch)
        left_img_batch = tf.layers.conv2d(left_img_batch,
                                     filters=64, kernel_size=(1, self.kernel_size), strides=1,
                                     padding='SAME', use_bias=True, trainable=is_training, name='left_conv1')
        
        right_img_batch = tf.layers.conv2d(right_img_batch,
                                     filters=256, kernel_size=(self.kernel_size, 1), strides=1,
                                     padding='SAME', use_bias=True, trainable=is_training, name='right_conv0')
        right_img_batch = tf.nn.relu(right_img_batch)
        right_img_batch = tf.layers.conv2d(right_img_batch,
                                     filters=64, kernel_size=(self.kernel_size, 1), strides=1,
                                     padding='SAME', use_bias=True, trainable=is_training, name='right_conv1')
        return left_img_batch, right_img_batch

    def decomp_block(self, img_batch, iter_num=3, is_training=True):
        '''Decomposition network which is used to generate an rank-one matrix
        Args:
            img_batch: an input image batch 
        Return:
            left_img_batch: the left vector of rank-one decomposition
            right_img_batch: the right vector of rank-one decomposition
            img_batch: the result of input batch subtracts a rank-one batch
        '''
        left_img_batch = img_batch
        right_img_batch = img_batch
        for num in range(iter_num):
            with tf.variable_scope('decomp_img{}'.format(num), reuse=tf.AUTO_REUSE):
                left_img_batch, right_img_batch = self.rankone_block(left_img_batch, right_img_batch, is_training)
        with tf.variable_scope('decomp_output', reuse=tf.AUTO_REUSE):
            left_img_batch = tf.layers.conv2d(left_img_batch,
                                              filters=self.out_channel, kernel_size=(1, self.kernel_size), strides=1,
                                              padding='SAME', use_bias=True, trainable=is_training, name='left_conv0')
            right_img_batch = tf.layers.conv2d(right_img_batch,
                                              filters=self.out_channel, kernel_size=(self.kernel_size, 1), strides=1,
                                              padding='SAME', use_bias=True, trainable=is_training, name='right_conv0')
        
        left_img_batch = tf.reduce_mean(left_img_batch, axis=2, keepdims=True)
        right_img_batch = tf.reduce_mean(right_img_batch, axis=1, keepdims=True)
        
        return left_img_batch, right_img_batch

    def res_blcok(self, img_batch, filter_size, is_training):
        identity = img_batch
        img_batch = tf.layers.conv2d(img_batch,
                                     filters=filter_size[0], kernel_size=self.kernel_size, strides=1,
                                     kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                     padding='SAME', use_bias=True, bias_regularizer=None,
                                     trainable=is_training, name='conv0')
        img_batch = tf.layers.batch_normalization(img_batch, momentum=0.99, beta_regularizer=None, 
                                                  gamma_regularizer=None, trainable=is_training)
        img_batch = tf.nn.relu(img_batch)
        img_batch = tf.layers.conv2d(img_batch,
                                     filters=filter_size[1], kernel_size=self.kernel_size, strides=1,
                                     kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                     padding='SAME', use_bias=True, bias_regularizer=None,
                                     trainable=is_training, name='conv1')
        return img_batch*self.res_scale + identity

    def recons_body(self, img_batch, filter_size, num_block, out_channel, sname, is_training):
        skip = img_batch   
        with tf.variable_scope(sname+'_input', reuse=tf.AUTO_REUSE):
            img_batch = tf.layers.conv2d(img_batch,
                                         filters = filter_size[1], kernel_size=self.kernel_size, strides=1,
                                         kernel_initializer=self.initializer, kernel_regularizer=None,
                                         padding='SAME', use_bias=True, bias_regularizer=None,
                                         trainable=is_training, name='conv0')
        for num in range(num_block):
            with tf.variable_scope(sname+'_block{}'.format(num), reuse=tf.AUTO_REUSE):
                img_batch = self.res_blcok(img_batch, filter_size, is_training)
                
        with tf.variable_scope(sname+'_output', reuse=tf.AUTO_REUSE):
            img_batch = tf.layers.conv2d(img_batch,
                                         filters = out_channel, kernel_size=self.kernel_size, strides=1,
                                         kernel_initializer=self.initializer, kernel_regularizer=None,
                                         padding='SAME', use_bias=True, bias_regularizer=None,
                                         trainable=is_training, name='conv0')
            skip = tf.layers.conv2d(skip,
                                    filters=out_channel, kernel_size=self.kernel_size, strides=1,
                                    kernel_initializer=self.initializer, kernel_regularizer=None,
                                    padding='SAME', use_bias=True, bias_regularizer=None,
                                    trainable=is_training, name='conv1')
        return img_batch + skip

    def pixel_shuffle(self, output_batch, filter_size, is_training):
        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            output_batch = tf.layers.conv2d(output_batch,
                                            filters = filter_size[1], kernel_size=self.kernel_size, strides=1,
                                            kernel_initializer=self.initializer, kernel_regularizer=None, 
                                            padding='SAME', use_bias=True, bias_regularizer=None,
                                            trainable=is_training, name='conv0')
            output_batch = tf.depth_to_space(output_batch, self.upscale//2)
            output_batch = tf.layers.conv2d(output_batch,
                                            filters = filter_size[1], kernel_size=self.kernel_size, strides=1,
                                            kernel_initializer=self.initializer, kernel_regularizer=None,
                                            padding='SAME', use_bias=True, bias_regularizer=None,
                                            trainable=is_training, name='conv1')
            output_batch = tf.depth_to_space(output_batch, self.upscale//2)
            output_batch = tf.layers.conv2d(output_batch,
                                            filters = self.out_channel, kernel_size=9, strides=1,
                                            kernel_initializer=self.initializer, kernel_regularizer=None,
                                            padding='SAME', use_bias=True, bias_regularizer=None, 
                                            trainable=is_training, name='conv2')
        return output_batch

    def infer(self, img_batch, label_batch, summary=False):
        '''Image SR based on rank one decomposition and reconstruction
        Args:
            img_batch: an tensor of size [batch_size, patch_size, patch_size, last_channel]
        
        Return:
            img_batch: the left error after decomposition
            output_batch: the SR of img_batch with size [batch_size, label_size, label_size, last_channel]
        
        '''
        #Source RO decomposition
        if self.DecMethod == 'RODec':
            with tf.variable_scope('decomp', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('decomp_block0', reuse=tf.AUTO_REUSE):
                    column_vector_batch, row_vector_batch = self.decomp_block(img_batch, is_training=False)
                    img_ROs_batch = self.outer_product(column_vector_batch, row_vector_batch)
                    img_batch -= img_ROs_batch
                    recons_batch = img_ROs_batch

                for num in range(1, self.num_decomp):
                    with tf.variable_scope('decomp_block{}'.format(num), reuse=tf.AUTO_REUSE):
                        column_vector_batch, row_vector_batch = self.decomp_block(img_batch, is_training=False)
                        img_RO_batch = self.outer_product(column_vector_batch, row_vector_batch)
                        img_batch -= img_RO_batch
                        recons_batch += img_RO_batch
                        img_ROs_batch = tf.concat([img_ROs_batch, img_RO_batch], axis=0)   
        elif self.DecMethod == 'SVD':
            _, recons_batch = self.batch_svd(img_batch, self.num_decomp)
            img_batch -= recons_batch
        else:
            raise Exception('Invalid RO decomposition method!')
        #Target RO decomposition
        if self.DecMethod == 'RODec':
            with tf.variable_scope('decomp', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('decomp_block0', reuse=tf.AUTO_REUSE):
                    column_vector_batch, row_vector_batch = self.decomp_block(label_batch, is_training=False)
                    label_ROs_batch = self.outer_product(column_vector_batch, row_vector_batch)
                    label_batch -= label_ROs_batch
                    true_batch = label_ROs_batch

                for num in range(1, self.num_decomp):
                    with tf.variable_scope('decomp_block{}'.format(num), reuse=tf.AUTO_REUSE):
                        column_vector_batch, row_vector_batch = self.decomp_block(label_batch, is_training=False)
                        label_RO_batch = self.outer_product(column_vector_batch, row_vector_batch)
                        label_batch -= label_RO_batch
                        true_batch += label_RO_batch
                        label_ROs_batch = tf.concat([label_ROs_batch, label_RO_batch], axis=0)
        elif self.DecMethod == 'SVD':
            _, true_batch = self.batch_svd(label_batch, self.num_decomp)
            label_batch -= true_batch
        else:
            raise Exception('Invalid RO decomposition method!')
        #RO reconstruction network
        with tf.variable_scope('RORec', reuse=tf.AUTO_REUSE):
            #Reconstruct RO components
            recons_batch = self.recons_body(recons_batch, self.filter_recons, self.num_recons, 
                    self.deep_scale*self.out_channel*self.upscale**2, sname='recons', is_training=True)
            #reconstruct residual
            img_batch = self.recons_body(img_batch, self.filter_err, self.num_err, 
                    self.deep_scale*self.out_channel*self.upscale**2, sname='error', is_training=True)
            #Fuse ROs and residual
            output_batch = tf.concat([recons_batch, img_batch], axis=3)
            output_batch = self.recons_body(output_batch, self.filter_err, self.num_output, 
                    self.out_channel*self.upscale**2, sname='concat', is_training=True)
            #upsample features
            if self.task in ['BiSR', 'ReSR']:
                recons_batch = tf.depth_to_space(recons_batch, self.upscale)
                img_batch = tf.depth_to_space(img_batch, self.upscale)    
                output_batch = self.pixel_shuffle(output_batch, self.filter_err, is_training=True)

        #summary histograms and images
        if summary:
            #summarize kernels
            tf.summary.histogram('RORec_input',
                    tf.get_default_graph().get_tensor_by_name('RORec/error_input/conv0/kernel:0'))
            tf.summary.histogram('RORec_output',
                    tf.get_default_graph().get_tensor_by_name('RORec/concat_output/conv0/kernel:0'))
            #summarize images
            tf.summary.image('Residual', tf.expand_dims(img_batch[...,0], axis=3), max_outputs=1)
            tf.summary.image('ROs_Sum', tf.expand_dims(recons_batch[...,0], axis=3), max_outputs=1)
        
        return [img_batch, label_batch, recons_batch, true_batch], tf.clip_by_value(output_batch, 0.0, 1.0)


    def test_infer(self, img_batch):
        '''Image SR based on rank one decomposition and reconstruction
        Args:
            img_batch: an tensor of size [batch_size, patch_size, patch_size, last_channel]
        
        Return:
            img_batch: the left error after decomposition
            output_batch: the SR of img_batch with size [batch_size, label_size, label_size, last_channel]
        
        '''
    #     mean = np.asarray([0.4485, 0.4375, 0.4045], dtype=np.float32)
    #     img_batch -= mean
        #Rank-one decomposition network
        if self.DecMethod == 'RODec':
            with tf.variable_scope('decomp', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('decomp_block0', reuse=tf.AUTO_REUSE):
                    column_vector_batch, row_vector_batch = self.decomp_block(img_batch, is_training=False)
                    img_ROs_batch = self.outer_product(column_vector_batch, row_vector_batch)
                    img_batch -= img_ROs_batch
                    recons_batch = img_ROs_batch

                for num in range(1, self.num_decomp):
                    with tf.variable_scope('decomp_block{}'.format(num), reuse=tf.AUTO_REUSE):
                        column_vector_batch, row_vector_batch = self.decomp_block(img_batch, is_training=False)
                        img_RO_batch = self.outer_product(column_vector_batch, row_vector_batch)
                        img_batch -= img_RO_batch
                        recons_batch += img_RO_batch
                        img_ROs_batch = tf.concat([img_ROs_batch, img_RO_batch], axis=0)
        elif self.DecMethod == 'SVD':
            _, recons_batch = self.batch_svd(img_batch, self.num_decomp)
            img_batch -= recons_batch
        else:
            raise Exception('Invalid RO decomposition method!')

        #Rank-one reconstruction network
        with tf.variable_scope('RORec', reuse=tf.AUTO_REUSE):
            #reconstruct RO components
            recons_batch = self.recons_body(recons_batch, self.filter_recons, self.num_recons, 
                    self.deep_scale*self.out_channel*self.upscale**2, sname='recons', is_training=False)
            #reconstruct residual
            img_batch = self.recons_body(img_batch, self.filter_err, self.num_err, 
                    self.deep_scale*self.out_channel*self.upscale**2, sname='error', is_training=False)
            #fuse RO components and residual
            output_batch = tf.concat([recons_batch, img_batch], axis=3)
            output_batch = self.recons_body(output_batch, self.filter_err, self.num_output, 
                    self.out_channel*self.upscale**2, sname='concat', is_training=False)
            if self.task in ['BiSR', 'ReSR']:
                recons_batch = tf.depth_to_space(recons_batch, self.upscale)
                img_batch = tf.depth_to_space(img_batch, self.upscale)
                output_batch = self.pixel_shuffle(output_batch, self.filter_err, is_training=False)

        return tf.clip_by_value(output_batch, 0.0, 1.0)

    def loss(self, feature_err, output_batch, label_batch, reg_parameter):
        '''Define the loss function for training
        Args:
            feature_err: the left error after rank-one decomposition
            output_batch: the output after rank-one reconstruction
            label_batch: the corresponding labels
            reg_parameter: the parameter for balancing decomposition and reconstruction errors
        '''
        if self.task in ['BiSR', 'ReSR']:
            decomp_error = tf.losses.absolute_difference(feature_err[0], feature_err[1]) + tf.losses.absolute_difference(feature_err[2], feature_err[3])
            recons_error = tf.losses.absolute_difference(output_batch, label_batch) + 1e-3*self.perceptual_loss(output_batch, label_batch)
            total_loss = reg_parameter*decomp_error + (1-reg_parameter)*recons_error
        elif self.task in ['DEN']:
            recons_error = tf.losses.mean_squared_error(output_batch, label_batch)
            #reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            total_loss = recons_error
        else:
            raise Exception('Invalid image restoration task!')
        return total_loss

    def training(self, loss, learning_rate, global_step):
        """Set up the training operations 
        Args:
            loss: loss function returned from loss()
            learning_rate: The learning rate to use for gradient descent.
            global_step: the global step for training
        Returns:
            train_op: the operation for training.
        
        """
        #Add a scalar summary for the loss.
        tf.summary.scalar('loss', loss)
        
        #Create the gradient descent optimizer with the given learning rate.
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
         
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'RORec')
        #train_vars = [var for var in train_vars if 'concat' in var.name]
        train_op = optimizer.minimize(loss, 
                                      var_list=train_vars, 
                                      global_step=global_step)
        return train_op

    def evaluation(self, output, labels, name):
        """Evaluate the quality of the output in reconstructing the label.
        
        Args:
            output: Reconstructed images
            labels: corresponding labels
            
        Returns:
            psnr: peak signal to noise ratio
            ssim: structural similarity
            
        """
        #Calculate measures
        psnr = tf.reduce_mean(tf.image.psnr(output, labels, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(output, labels, max_val=1.0))
        tf.summary.scalar(name+'-psnr', psnr)
        tf.summary.scalar(name+'-ssim', ssim)
        return psnr, ssim
        
        
    
            
    
        
    
