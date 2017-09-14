from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
import urllib
from scipy import misc

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

checkpoints_dir = 'checkpoints'

slim = tf.contrib.slim

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = vgg.vgg_16.default_image_size

names = os.listdir('test')
name = 'test/' + names[7]
with tf.Graph().as_default():
    
    image_string = misc.imread(name)
    
    image = tf.convert_to_tensor(image_string, np.float32)
    
    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_image = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logit, _ = vgg.vgg_16(processed_image,
                             num_classes=1000,
                             is_training=False)
    
    probabilities = tf.nn.softmax(logit)
    
    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    
    with tf.Session() as sess:
        
        # Load weights
        init_fn(sess)
        
        # We want to get predictions, image as numpy matrix
        # and resized and cropped piece that is actually
        # being fed to the network.
        np_image, network_input, probabilities = sess.run([image,
                                                          processed_image,
                                                          probabilities])
        probabilities = probabilities[0, 0:]
        sorted_indx = [i[0] for i in sorted(enumerate(-probabilities),
                                                     key=lambda x:x[1])]
        
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np_image.astype(np.uint8))
    plt.title("Downloaded", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Show the image that is actually being fed to the network
    # The image was resized while preserving aspect ratio and then
    # cropped. After that, the mean pixel value was subtracted from
    # each pixel of that crop. We normalize the image to be between [-1, 1]
    # to show the image.
    plt.subplot(1,2,2)
    plt.imshow( network_input[0] / (network_input.max() - network_input.min()) )
    plt.title("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    names = imagenet.create_readable_names_for_imagenet_labels()
    
    for i in range(5):
        index = sorted_indx[i]
        # Now we print the top-5 predictions that the network gives us with
        # corresponding probabilities. Pay attention that the index with
        # class names is shifted by 1 -- this is because some networks
        # were trained on 1000 classes and others on 1001. VGG-16 was trained
        # on 1000 classes.
        print("Probability %0.2f => [%s]" % (probabilities[index], names[index+1]))