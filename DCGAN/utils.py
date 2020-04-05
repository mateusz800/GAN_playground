"""
Module gives some basic tools to train and evaluate models
"""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_images(generator, input_dim=100, count=1):
    """ Use generator to generate images """
    noise = np.random.normal(size=[count, input_dim])
    images = generator.predict(noise)
    images = images.reshape(
        images.shape[0], images.shape[1], images.shape[2])  # cut last axis
    return images


def save_model(generator, discriminator, gan, dir='model'):
    create_directory(dir=dir)
    generator.save(f'{dir}/generator.h5')
    discriminator.save(f'{dir}/discriminator.h5')
    gan.save(f'{dir}/gan.h5')


def create_collage(images, path, rows=5, cols=5):
    """ Save image containing several images inside """
    create_directory(path=path)  # make sure that directory exist
    plt.figure()
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.axis('off')
        plt.imshow(images[i], cmap='gray_r', interpolation='nearest')
    plt.savefig(path)


def create_gif(dir, output_path):
    """ 
    Create gif from images that are in folder specifide by dir parameter.
    Images in given directory must be in png format.
    """
    create_directory(path=output_path)  # make sure that directory exist
    images = []
    for file_name in os.listdir(dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(dir, file_name)
            images.append(Image.open(file_path))
    imageio.mimsave(output_path, images, fps=4)


def create_plot(data, path, title='', x_label='', y_label=''):
    """ Create plot of given data and save it to file """
    create_directory(path=path)
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(data)
    plt.savefig(path)


def create_directory(path=None, dir=None):
    """ Split path to get directory and if not exist make it """
    if path:
        dir_array = path.split('/')[:-1]
        dir = '/'.join(elem for elem in dir_array)
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except FileNotFoundError:
            child_dir = dir.split('/')[:-1]
            child_dir = '/'.join(elem for elem in child_dir)
            create_directory(dir=child_dir)
            create_directory(dir=dir)


def get_last_photo_number(dir) -> int:
    """ Calculate last photo number usin files count """
    try:
        return len(os.listdir(dir))
    except FileNotFoundError:
        return 0