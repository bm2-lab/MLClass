import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs

x = np.linspace(-10, 10, 100)
y = scs.expit(x)

plt.plot(x, y)
plt.show()