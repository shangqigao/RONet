'''
Created on May 9, 2019

@author: gsq
'''
import tensorflow as tf
import numpy as np

class RODec:
    def __init__(self, FLAGS):
        self.out_channel = FLAGS.out_channel
        self.kernel_size = 3
        self.num_decomp = FLAGS.num_ROPs
        #Training RODec: supervised or unsupervised
        self.mode = FLAGS.train_mode 


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

    def rankone_block(self, left_img_batch, right_img_batch, is_training):
        left_img_batch = tf.layers.conv2d(left_img_batch,
                                     filters=256, kernel_size=(1, self.kernel_size), strides=1,
                                     padding='SAME', use_bias=True, trainable=is_training, name='left_conv0')
        left_img_batch = tf.layers.batch_normalization(left_img_batch, trainable=is_training)
        left_img_batch = tf.nn.relu(left_img_batch)
        left_img_batch = tf.layers.conv2d(left_img_batch,
                                     filters=64, kernel_size=(1, self.kernel_size), strides=1,
                                     padding='SAME', use_bias=True, trainable=is_training, name='left_conv1')
        
        right_img_batch = tf.layers.conv2d(right_img_batch,
                                     filters=256, kernel_size=(self.kernel_size, 1), strides=1,
                                     padding='SAME', use_bias=True, trainable=is_training, name='right_conv0')
        right_img_batch = tf.layers.batch_normalization(right_img_batch, trainable=is_training)
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

    def batch_svd(self, img_batch, num): 
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
              

    def infer(self, img_batch, label_batch, summary=False):
        '''Image SR based on rank one decomposition and reconstruction
        Args:
            img_batch: an tensor of size [batch_size, patch_size, patch_size, last_channel]
        
        Return:
            img_batch: the left error after decomposition
            output_batch: the SR of img_batch with size [batch_size, label_size, label_size, last_channel]
        
        '''
        #RO decomposition network
        true_batch, _ = self.batch_svd(img_batch, self.num_decomp)
        with tf.variable_scope('decomp', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('decomp_block0', reuse=tf.AUTO_REUSE):
                column_vector_batch, row_vector_batch = self.decomp_block(img_batch)
                ROs_batch = self.outer_product(column_vector_batch, row_vector_batch)
                img_batch -= ROs_batch
                Res_batch = img_batch 
                
            for num in range(1, self.num_decomp):
                with tf.variable_scope('decomp_block{}'.format(num), reuse=tf.AUTO_REUSE):
                    column_vector_batch, row_vector_batch = self.decomp_block(img_batch)
                    RO_batch = self.outer_product(column_vector_batch, row_vector_batch)
                    img_batch -= RO_batch
                    ROs_batch = tf.concat([ROs_batch, RO_batch], axis=0)
                    Res_batch = tf.concat([Res_batch, img_batch], axis=0)
        
        if summary:
            #summarize kernels
            tf.summary.histogram('RODec_input',
                    tf.get_default_graph().get_tensor_by_name('decomp/decomp_block0/decomp_img0/left_conv0/kernel:0'))
            tf.summary.histogram('RODec_output',
                    tf.get_default_graph().get_tensor_by_name('decomp/decomp_block5/decomp_output/right_conv0/kernel:0'))
            #summarize images
            tf.summary.image('Residual', img_batch, max_outputs=1)
            tf.summary.image('RO_Components', ROs_batch, max_outputs=1)

        return img_batch, [ROs_batch, true_batch, Res_batch]

    def test_infer(self, img_batch):
        '''Image SR based on rank one decomposition and reconstruction
        Args:
            img_batch: an tensor of size [batch_size, patch_size, patch_size, last_channel]
        
        Return:
            img_batch: the left error after decomposition
            output_batch: the SR of img_batch with size [batch_size, label_size, label_size, last_channel]
        
        '''
        #RO decomposition network
        
        with tf.variable_scope('decomp', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('decomp_block0', reuse=tf.AUTO_REUSE):
                column_vector_batch, row_vector_batch = self.decomp_block(img_batch)
                ROs_batch = self.outer_product(column_vector_batch, row_vector_batch)
                img_batch -= ROs_batch

            for num in range(1, self.num_decomp):
                with tf.variable_scope('decomp_block{}'.format(num), reuse=tf.AUTO_REUSE):
                    column_vector_batch, row_vector_batch = self.decomp_block(img_batch)
                    RO_batch = self.outer_product(column_vector_batch, row_vector_batch)
                    img_batch -= RO_batch
                    ROs_batch = tf.concat([ROs_batch, RO_batch], axis=0)
        
        
        return img_batch, ROs_batch 


    def loss(self, img_batch, output_batch, label_batch):
        '''Define the loss function for training
        Args:
            img_batch: the left error after rank-one decomposition
            output_batch: the output after rank-one reconstruction
            label_batch: the corresponding label batch
        '''
        if self.mode == 'unsupervised':
            RODec_loss = tf.reduce_mean(tf.square(output_batch[2]))
        elif self.mode == 'supervised':
            RODec_loss = tf.reduce_mean(tf.square(output_batch[0] - output_batch[1]))
        else:
            raise Exception('Invalid training mode!')
        return RODec_loss

    def training(self, loss, learning_rate, global_step):
        """Set up the training operations
        Create a summarizer to track the loss over time in TensorBoard.
        Create an optimizer and apply gradients to all trainable variables.
        
        The operation returned by this function is what must be passed to
        "sess.run()" call to cause the model to train
        
        Args:
            loss: L_2 loss tensor from loss()
            learning_rate: The learning rate to use for gradient descent.
            
        Returns:
            train_op: the operation for training.
        
        """
        #Add a scalar summary for the loss.
        tf.summary.scalar('loss', loss)
        
        #Create the gradient descent optimizer with the given learning rate.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
         
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, 
                                      var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'decomp'),
                                      global_step=global_step)
        return train_op

    def evaluation(self, output, labels, name):
        """Evaluate the quality of the output at reconstructing the label.
        
        Args:
            output: Reconstructed tensor, float, --[batch_size, IMAGE_PIXELS]
            labels: Lables tensor, float, --[batch_size, IMAGE_PIXELS]
            
        Returns: A scalar int32 tensor with the number of examples for which
        their mean squared errors are all less than a given value.
            
        """
        #Calculate PSNR and SSIM
        output1 = output[0]
        output2 = output[1]
        psnr = tf.reduce_mean(tf.image.psnr(output1, output2, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(output1, output2, max_val=1.0))
        tf.summary.scalar(name+'-psnr', psnr)
        tf.summary.scalar(name+'-ssim', ssim)
        return psnr, ssim
        
        
    
            
    
        
    
