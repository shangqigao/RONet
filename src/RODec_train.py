'''
Created on May 9, 2019

@author: gsq

Trains and evaluates rank-one network
'''
import sys
sys.path.append('../')
import argparse
import os.path
import time
import logging

import tensorflow as tf
import numpy as np

from common.load_DIV2K import Dataloader
from RODec_model import RODec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer(RODec):
    def __init__(self, FLAGS):
        RODec.__init__(self, FLAGS)

    def placeholder_inputs(self, batch_size, img_size, lab_size, channels):
        """Generate placeholder variables to represent the input tensors
        
        These placeholder are used as inputs by the rest of the model building code.
        
        Args:
            batch_size: The batch size will be bakes into both placeholders.
            
        Returns:
            images_placeholder: Images placeholder.
            labels_placeholder: Labels placeholder.
            regularization_placeholder: regularization placeholder
        
        """
        #Note that the shapes of the placeholders match the shapes of the full image
        # and label tensors, except the first dimension is the batch size rather than 
        #the full size of the train or test data set
        
        images_placeholder = tf.placeholder(tf.float32, 
                                            shape=(batch_size, img_size, img_size, channels))
        labels_placeholder = tf.placeholder(tf.float32, 
                                            shape=(batch_size, lab_size, lab_size, channels))
        return images_placeholder, labels_placeholder

    def fill_feed_dict(self, sess, img_batch, images_pl, labels_pl):
        """Fill the feed_dict for traning the given step
        
        A feed_dict takes the form of 
        feed_dict = {
            <placeholder> <values of tensor to be passed to placeholder>
            ....
        }
        
        Args:
            data_set:The set of images and labels from input_dec
            images_pl: The image placeholder from placeholder_input()
            labels_pl: The label placeholder from placeholder_input()
            
        Returns:
            feed_dict: The feed dictionary mapping from placeholder to values
        
        """
        #Create the feed_dict for placeholders
        images_feed, labels_feed = sess.run(img_batch)
        
        feed_dict = {
            images_pl : images_feed,
            labels_pl : labels_feed,   
        }
        return feed_dict

        
    def run_training(self):
        """Train given data set for a number of steps."""
        #Tell Tensorflow that the model will be built into the default graph
        with tf.Graph().as_default():
            dataloader = Dataloader(FLAGS)
            train_dataset, valid_dataset = dataloader.generate_dataset(FLAGS.input_data_dir)
            train_batch = train_dataset.make_one_shot_iterator().get_next()
            valid_batch = valid_dataset.make_one_shot_iterator().get_next()
            #Generate placeholder for images and labels.
            images_pl, labels_pl = self.placeholder_inputs(FLAGS.train_batch_size, FLAGS.train_patch_size,
                    FLAGS.train_patch_size*FLAGS.upscale, self.out_channel)
            images_pl_valid, labels_pl_valid = self.placeholder_inputs(FLAGS.valid_batch_size, FLAGS.valid_patch_size,
                    FLAGS.valid_patch_size*FLAGS.upscale, self.out_channel)

            #Build a graph that compute reconstructions for the inference model.
            err_decomp, output = self.infer(images_pl, labels_pl, summary=True)
            _, valid_output = self.infer(images_pl_valid, labels_pl_valid, summary=False)

            #Add to the graph the operations for loss calculation.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rates = [1e-4, 1e-5]
            boundaries = [400000]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rates)

            loss = self.loss(err_decomp, output, labels_pl) 
            
            #Add to the graph the operations calculate and apply gradient.
            train_op = self.training(loss, learning_rate, global_step)
            
            #Add to the graph the operations for comparing logits with labels.
            eval_perform = self.evaluation(output, labels_pl, 'Train')
            eval_valid = self.evaluation(valid_output, labels_pl_valid, 'Test')
            
            #BUild the summary tensor based on Tensorflow collection of summaries.
            summary = tf.summary.merge_all()
            #Add the variable initializer operation.
            all_vars = tf.global_variables()
            
            #Create a saver for an overview of variables, saving and restoring.
            saver = tf.train.Saver(all_vars, max_to_keep=1) 
            
            #Create a session for running operations on the graph.
            sess = tf.Session() 
            
            #Instantiate a SummaryWriter to output summaries and graph.
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            
            #calculate the number of trainable variables
            total_paras = 0
            for ele in tf.trainable_variables():
                #print(ele.name)
                total_paras += np.prod(np.array(ele.get_shape(), np.int32))
            total_paras = float(total_paras) / 1e6
            print(f'Total trainable parameters: {total_paras:0.2f}M')

            #Run the operation to initialize variables.
            sess.run(tf.variables_initializer(all_vars))
            
            #Restore variables from disk
            if FLAGS.resume:
                saver.restore(sess, FLAGS.RODec_checkpoint)

            #Start the training loop.
            init_step = sess.run(global_step)
            best_psnr, best_ssim = 0., 0.
            for step in range(init_step + 1, FLAGS.max_steps + 1):
                start_time = time.time()
                
                #Fill a feed dictionary with the actual set of images and labels for this training step.
                feed_dict = self.fill_feed_dict(sess, train_batch, images_pl, labels_pl)

                #Save a checkpoint and evaluate the model periodically.
                if step == init_step + 1 or step % 1000 == 0:
                    #save checkpoint
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model')
                    saver.save(sess, checkpoint_file, global_step=step)

                    #update feed dict
                    valid_dict = self.fill_feed_dict(sess, valid_batch, images_pl_valid, labels_pl_valid)
                    feed_dict.update(valid_dict)

                    #Evaluate against the training set.
                    train_psnr, train_ssim = sess.run(eval_perform, feed_dict)
                    logging.info('Train evaluation: PSNR=%0.04f  SSIM=%0.04f' % (train_psnr, train_ssim))

                    #Evaluate against the validation set.
                    new_psnr, new_ssim = sess.run(eval_valid, feed_dict)
                    logging.info('Valid evaluation: PSNR=%0.04f  SSIM=%0.04f' % (new_psnr, new_ssim))

                    #Save the best model
                    is_new_best = new_psnr > best_psnr or new_ssim > best_ssim
                    best_psnr = max(best_psnr, new_psnr)
                    best_ssim = max(best_ssim, new_ssim)
                    if is_new_best:
                        checkpoint_file = os.path.join(FLAGS.log_dir, 'best_model')
                        saver.save(sess, checkpoint_file)
                    logging.info('Best  evaluation: PSNR=%0.04f  SSIM=%0.04f' % (best_psnr, best_ssim))


                    #Update the events file
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                #Run one step of the model. The return values are activations from the train_op
                #(which is discarded) and the loss operations.
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time
                lr = sess.run(learning_rate)
                #Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    #Print status
                    logging.info('%s | step:[%7d/%7d] | loss=%0.04f | lr=%1.0e (%0.03f sec)' % \
                            (time.strftime("%Y-%m-%d %H:%M:%S"), step, FLAGS.max_steps, loss_value, lr, duration))
 
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir) and not FLAGS.resume:
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    trainer = Trainer(FLAGS)
    trainer.run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=500000,
                        help='Number of steps to run trainer'
                        )
    parser.add_argument('--upscale', type=int, default=1,
                        help='upsclaing factor'
                        )
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Train batch size. Must divide evenly into dataset size'
                        )
    parser.add_argument('--valid_batch_size', type=int, default=32,
                        help='validation batch size. Must divide evenly into dataset size'
                        )
    parser.add_argument('--train_patch_size', type=int, default=64,
                        help='the size of patches for training'
                        )
    parser.add_argument('--valid_patch_size', type=int, default=64,
                        help='the size of patches for validation'
                        )
    parser.add_argument('--threads', type=int, default=10,
                        help='Number of threads for loading data'
                        )
    parser.add_argument('--task', type=str, default='DEN',
                        help='Which image restoration task'
                        )
    parser.add_argument('--train_mode', choices=['supervised', 'unsupervised'], required=True,
                        help='Which image restoration task'
                        )
    parser.add_argument('--num_ROPs', type=int, default=6,
                        help='the maximal number of RO projections in RODec'
                        )
    parser.add_argument('--out_channel', type=int, default=1,
                        help='which is 1 for grayscale and 3 for RGB'
                        )
    parser.add_argument('--input_data_dir', type=str, default='./data',
                        help='Directory to put the input data'
                        )
    parser.add_argument('--augment', action='store_true',
                        help='whether use data augmentation'
                        )
    parser.add_argument('--sigma', type=int, default=0,
                        help='the noise level ranges in [0, 255]'
                        )
    parser.add_argument('--range', action='store_true',
                        help='whether let the noise level range in [0, sigma]'
                        )
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to put the log data'
                        )
    parser.add_argument('--resume', action='store_true',
                        help = 'If set, resume the training from a previous model checkpoint'
                        )
    parser.add_argument('--RODec_checkpoint', type=str, default='./models',
                        help='Path of RO decomposition model'
                        )
    parser.add_argument('--GPU_ids', type=str, default = '0',
                        help = 'Ids of GPUs'
                        )
    FLAGS, unparsed = parser.parse_known_args()
    print('--------------------------Arguments-----------------------')
    print(FLAGS)
    print('---------------------------End----------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(FLAGS.GPU_ids)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
