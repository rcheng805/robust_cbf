import numpy as np
import matplotlib.pyplot as plt

# Load stored parameter values (and corresponding loss)
allvals = []
allparams = []
N = 7
for i in range(N):
    vals = np.load('likelihood_vals_v' + str(i) + '.npy')
    params = np.load('parameters_v' + str(i) + '.npy', allow_pickle=True)
    allvals.append(np.array(vals))
    allparams.append(params)
colors = ['r', 'g', 'b', 'k', 'skyblue','y','m']

# Plot loss function
plt.subplot(131)
for i in range(N):
    plt.plot(np.array(allvals[i]), color=colors[i])
    plt.ylim(0, 60000)
# Plot sigma
plt.subplot(132)
for i in range(N):
    sigma = allparams[i][1]
    plt.plot(sigma, color=colors[i])
# Plot length scale (l)
plt.subplot(133)
for i in range(N):
    l = allparams[i][2]
    plt.plot(l, color=colors[i])
plt.show()

