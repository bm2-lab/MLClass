import numpy as np
import matplotlib.pyplot as plt

def show_mnist_fig(prefix, arr, y):
    arr = arr.reshape(28, 28)
    y = np.argmax(y)
    plt.imshow(arr, cmap='Greys', vmax=1, vmin=0)
    plt.savefig(f'{prefix}_{y}.png')

