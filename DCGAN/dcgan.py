"""
DCGAN model to generate digits (trained on MNIST dataset)
"""

import getopt
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Flatten, LeakyReLU, MaxPooling2D, Reshape)
from keras.models import Sequential, load_model
from tqdm import tqdm

from utils import (create_collage, create_gif, create_plot, generate_images,
                   save_model, create_directory, get_last_photo_number)


def get_generator(input_dim=100):
    """ Build sequential model that will generate 28x28 images of digits """
    model = Sequential()
    # -------- layer 1 -----------------------
    model.add(Dense(7*7*4, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Reshape((7, 7, 4)))
    # -------- layer 2 -----------------------
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # -------- layer 3 -----------------------
    model.add(Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def get_discriminator(input_shape=(28, 28, 1)):
    """ Build sequential model that will try to answer if given image is real or fake """
    model = Sequential()
    # -------- layer 1 -----------------------
    model.add(Conv2D(128, (5, 5), input_shape=input_shape))
    model.add(LeakyReLU())
    # -------- layer 2 -----------------------
    model.add(Conv2D(256, (5, 5)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    # -------- layer 3 -----------------------
    model.add(Conv2D(128, (5, 5)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    # -------- layer 4 -----------------------
    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU())
    # -------- output -----------------------
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_gan(generator, discriminator):
    """
    Build gan model.
    Generator will try to generate images and trick discriminator but discriminator will try to distinguish
    real and fake images.
    """
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def train(discriminator, gan, real_data, model_dir, batch_size=32, epochs=1, history=None):
    """ 
    Train gan model.
    Discriminator have to be compiled with accuracy metrics
    """
    try:
        total_steps = history.shape[0]
        last_epoch = get_last_photo_number(dir=f'{model_dir}/images')
    except AttributeError:
        history = pd.DataFrame(columns=['d_loss', 'd_acc', 'g_loss'])
        last_epoch = 0
        total_steps =0
    half_batch_size = int(batch_size/2)
    steps = int(len(real_data)/batch_size)
    #steps = 1
    for epoch in range(epochs):
        print(f'Epoch  {epoch + 1}')
        index = 0  # it will indicate the index in real_data to get appropriate batch of images
        for step in tqdm(range(steps)):
            y = np.zeros(batch_size)    # fake images have label 0
            y[half_batch_size:] = 0.9   # labels for real images
            noise = np.random.normal(0, 1, size=(half_batch_size, 100))
            generated_images = generate_images(
                generator=gan.layers[0], count=half_batch_size)
            # Conv2D layer takes as an input 3 dimensions so we want to have shape equal (28, 28, 1)
            generated_images = np.expand_dims(generated_images, axis=3)
            real_images = real_data[index: index+half_batch_size]
            x = np.concatenate((generated_images, real_images))
            discriminator.trainable = True
            d_metrics = discriminator.train_on_batch(x, y)
            discriminator.trainable = False
            another_noise = np.random.normal(0, 1, size=(batch_size, 100))
            gan_loss = gan.train_on_batch(another_noise, np.ones(batch_size))
            history.loc[total_steps] = [d_metrics[0], d_metrics[1], gan_loss]
            total_steps += 1
        save_model(gan.layers[0], discriminator, gan, dir=f'{model_dir}/model')
        # visualizing progress of training
        print(history.loc[total_steps-1])
        sample_images = generate_images(generator=gan.layers[0], count=3*4)
        create_collage(
            sample_images, f'{model_dir}/images/epoch_{epoch+last_epoch}.png', rows=3, cols=4)
    return history


def get_data():
    (data, _), (_, _) = mnist.load_data()
    # normalization - each pixel hava value in range (-1, 1)
    data = (data / 255) - 1
    # Conv2D layer takes as an input 3 dimensions so we want to have shape equal (28, 28, 1)
    data = np.expand_dims(data, axis=3)
    return data


if __name__ == '__main__':
    # default settings
    model_directory = 'model'
    history = None
    generator = None
    # command line options
    arguments_list = sys.argv[1:]
    short_opt = 'm:o'
    long_opt = ['model_dir=', 'output_dir=']
    try:
        arguments, values = getopt.getopt(arguments_list, short_opt, long_opt)
    except getopt.error as err:
        print(err)
        sys.exit(1)
    # settings depends on given options
    for current_arg, current_val in arguments:
        if current_arg in ('-m', '--model_dir'):
            try:
                model_directory = current_val
                generator = load_model(f'{current_val}/model/generator.h5')
                discriminator = load_model(f'{current_val}/model/discriminator.h5')
                gan = load_model(f'{current_val}/model/gan.h5')
                history = pd.read_csv(f'{model_directory}/history.csv')
                print('Models loaded succesfull')
            except OSError:
                print('Given directory does not contain models')
        elif current_arg in ('-o', '--output_dir'):
            if not os.path.exists(current_val):
                create_directory(dir=current_val)
            model_directory = current_val

    if not generator:
        generator = get_generator()
        discriminator = get_discriminator()
        gan = get_gan(generator, discriminator)
    training_data = get_data()
    history = train(discriminator, gan, training_data, model_dir=model_directory,
                    batch_size=128, epochs=5, history=history)
    # save training data
    create_plot(history['d_acc'], path=f'{model_directory}/plots/discriminator_accuracy.png',
                title='Discriminator Accuracy', x_label='step', y_label='acc')
    create_plot(
        history['d_loss'], path=f'{model_directory}/plots/discriminator_loss.png',
        title='Discriminator Loss', x_label='step', y_label='loss')
    create_plot(history['g_loss'], path=f'{model_directory}/plots/gan_loss.png',
                title='GAN Loss', x_label='step', y_label='loss')
    history.to_csv(f'{model_directory}/history.csv', index=False)
    save_model(generator, discriminator, gan, dir=f'{model_directory}/model')
    create_gif(f'{model_directory}/images', f'{model_directory}/training.gif')
