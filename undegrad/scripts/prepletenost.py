import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

x = np.pi * np.linspace(0, 1.5, 200)
y = np.cos(x)**2
z = np.sin(x)**2

fig, ax = plt.subplots()
ax.plot(x, y, 'k-', lw=1, label=r'$p_{01}$')
ax.plot(x, z, 'r-', lw=1, label=r'$p_{10}$')
ax.set_ylabel('verjetnost za meritev', fontsize=14)
ax.set_xlabel('čas $gt$', fontsize=14)
ax.legend(loc='center right', fontsize=14)
plt.show()