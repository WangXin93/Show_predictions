# Import dependencies
import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
import matplotlib.pyplot as plt

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

# Set parameter
FILE_DIR = '/home/wangx/project/show_single_pred/images'
IMAGE_SIZE = vgg.vgg_16.default_image_size
NUM_EPOCHS = 10
CHECKPOINTS_DIR = '/home/wangx/project/show_single_pred/checkpoints'
names = imagenet.create_readable_names_for_imagenet_labels()
NUM_IMAGES = 3    # how many image you want to show

# Helper function to get list of filenames, labels
def get_files(file_dir):
    '''
    Args:
        file_dir: directory include images
    Returns:
        filenames: list of path of images
        labels: list of labels corresponding to images
    '''
    filenames = []
    labels = []

    for line in open('test.txt'):
        line = line.strip('\n') + '.jpg'
        labels.append(line.split('/')[0])
        filenames.append(os.path.join(file_dir, line))

    print('There are totally {} files.'.format(len(labels)))
    print('For example: ',filenames[0])
    return filenames, labels

# Map function to dataset
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded,
                                           [IMAGE_SIZE,IMAGE_SIZE])
    processed_img = vgg_preprocessing.preprocess_image(image_decoded,
                                                        IMAGE_SIZE,
                                                        IMAGE_SIZE,
                                                        is_training=False)
    raw_img = image_resized
    return raw_img, processed_img, label

# Get filename list and label list
filenames, labels = get_files(FILE_DIR)

with tf.Graph().as_default():

    # Use tf.contrib.data.Dataset to load batch samples
    dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    #dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(1)
    dataset = dataset.repeat(NUM_EPOCHS)
    iterator = dataset.make_one_shot_iterator()

    next_raw_img, next_processed_img, next_label = iterator.get_next()
    
    # Use model to infer probabilities
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logit, _ = vgg.vgg_16(next_processed_img, num_classes=1000, is_training=False)

    probabilities = tf.nn.softmax(logit)
    
    # Set function to restore weights
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(CHECKPOINTS_DIR, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
        
    with tf.train.MonitoredTrainingSession() as sess:

        # Load weights
        init_fn(sess)

        # Set loop numbers here
        i = 0

        while not sess.should_stop() and i<NUM_IMAGES:
            np_raw_img, np_processed_img, np_probabilities = sess.run([next_raw_img,
                                                                       next_processed_img,
                                                                       probabilities])
            np_probabilities = np_probabilities[0, 0:]
            
            # Get an ordered list of index by descending probabilities
            sorted_idx = [i[0] for i in sorted(
                enumerate(-np_probabilities), key=lambda x:x[1])]
            
            # Print predictions and save it to draw
            pred_class = []
            pred_prob = []
            for t in range(5):
                index = sorted_idx[t]
                print("Probability %0.2f => [%s]" % (np_probabilities[index], names[index+1]))
                pred_class.append(names[index+1])
                pred_prob.append(np_probabilities[index])
             
            # Show raw image and preprocessed image and top-5 predictions   
            plt.rcdefaults()
            fig = plt.figure(figsize=(16,9))
            ax1 = plt.subplot2grid((2,2),(0,0))
            ax2 = plt.subplot2grid((2,2),(0,1))
            ax3 = plt.subplot2grid((2,2),(1,0), colspan=2)
            ax1.imshow(np_raw_img[0].astype(np.uint8))
            ax1.set_title('Downloaded Image')
            ax1.axis('off')
            ax2.imshow(np_processed_img[0]/(
                np_processed_img[0].max() - np_processed_img[0].min()))
            ax2.set_title('Preprocessed Image')
            ax2.axis('off')
            
            y_pos = np.arange(len(pred_class))
            ax3.barh(y_pos, pred_prob, color='green')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(pred_class)
            ax3.invert_yaxis()
            ax3.set_xlabel('Probabilities')
            ax3.set_title('Top-5 Predictions')
            plt.tight_layout()
            plt.show()
            
            i += 1
