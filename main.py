import time
import numpy as np

from tf_image_models.dcgan.dcgan import DCGAN
from tf_utils.manage_data import *
from tf_utils.process_images import *
from tf_utils.python_utils import *

#############################
#   General Parameters      #
#############################
model_name = "dc_gan"
batch_size = 2
epochs = 1000
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 320
img_shape = 200
img_channels = 3

load_model = False
start_epoch = 0

test_images_proportion = 0.3 # In case of training GAN, set to 0
num_examples_to_generate = 4 # Num Images to generate during training to show process
seed = None # Only in GAN cases, if None, seed is set randomly with length num_examples_to_generate. Setting a seed makes easier to visualize the progress of GAN training

# Create model
model = DCGAN(latent_dim=latent_dim, image_shape=img_shape, image_channels=img_channels, model_name=model_name, seed=seed, seed_length=num_examples_to_generate)
model = CVAE(latent_dim, image_shape, image_channels, load_model = load_model)

#############################
#        Get Data           #
#############################
#### Download a test dataset #####
# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
########## Load a video ##########
# train_images = extract_video_frames("data_in/video_cortito.mp4", img_shape=img_shape, img_channels=img_channels)
###### Load a set of images ######
train_images = load_image_dataset("data_in/selulitis/", img_shape = img_shape, img_channels = img_channels)
# train_images = load_image_dataset("data_in/landscapes/", img_shape = img_shape, img_channels = img_channels)
###################################

#############################
#      Preprocess Data      #
#############################
train_images = shuffle_dataset(train_images)
train_images, test_images = split_dataset(train_images, test_proportion=test_images_proportion)
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)
train_size = len(train_images); test_size = len(test_images)
# Print data info:
print("\nTrain Data -> shape: " + str(np.shape(train_images)) + " -> Img.type -> " + str(type(train_images[0])))
print("--------------------------------------------------------------------------")
print("| Num.images  | Img.shape (px)  | Num.Channels  | Max.Value  | Min.Value |")
print("| " + str(train_size) + " | " + str(np.shape(train_images)[1:-2]) + " | " + \
        str(np.shape(train_images)[3]) + " | " + str(np.max(train_images)) + " | " + \
        str(np.min(train_images)) + " |")
print("\nTest Data -> shape: " + str(np.shape(test_images)) + " -> Img.type -> " + str(type(test_images[0])))
print("--------------------------------------------------------------------------")
print("| Num.images  | Img.shape (px)  | Num.Channels  | Max.Value  | Min.Value |")
print("| " + str(test_size) + " | " + str(np.shape(test_images)[1:-2]) + " | " + \
        str(np.shape(test_images)[3]) + " | " + str(np.max(test_images)) + " | " + \
        str(np.min(test_images)) + " |")

# Batch and shuffle the data, create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
test_dataset  = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)
del train_images
del test_images
gc.collect()

if load_model:
  model.load_weights("{epoch:04d}".format(epoch=start_epoch))

#############################
#        Train Model        #
#############################
for epoch in range(start_epoch, epochs+start_epoch):
    start = time.time()

    mean_loss = [0.0]*len(model.loss_names)
    for image_batch in train_dataset:
        loss = model.train_step(image_batch, batch_size)
        for i in range(len(model.loss_names)):
            mean_loss[i]+=loss[i]
    mean_loss = np.array(mean_loss)/len(train_dataset)

    # Produce images for the GIF as you go
    save_image_matrix(model.generate_images(model.seed), img_path ='data_out/{}-image_at_epoch_{:04d}'.format(model_name, epoch))
    # Save the model every 15 epochs
    if (epoch) % 15 == 0:
        model.save_weights("{epoch:04d}".format(epoch=epoch))

    print('-- EPOCH {}'.format(epoch))
    print('Execution Time {} sec'.format(time.time()-start))
    print('Mean Loss:')
    for l, n in zip(mean_loss, model.loss_names):
        print('{} : {}'.format(n,l))

# Save model after final epoch
model.save_weights("{epoch:04d}".format(epoch=epochs))
# Generate after the final epoch
save_image_matrix(model.generate_images(model.seed), img_path ='data_out/{}-image_at_epoch_{:04d}'.format(model_name, epochs))
# Generate GIF and video to show training progress
save_gif('data_out/'+model_name, re_images_name='data_out/{}-image_at_epoch_*.png'.format(model_name))
save_mp4('data_out/'+model_name, re_images_name='data_out/{}-image_at_epoch_*.png'.format(model_name))