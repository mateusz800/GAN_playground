import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model

from utils import generate_images, create_collage

if __name__ == '__main__':

    data = pd.read_csv('model2/history.csv')
    plt.yscale('log')
    plt.plot(data['d_loss'])
    plt.show()
    """
    generator = load_model('model/model/generator.h5')
    images = generate_images(generator, count=12)
    create_collage(images, 'test/test.png', rows=3, cols=4)
    """