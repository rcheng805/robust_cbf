import numpy as np
import matplotlib.pyplot as plt
import pickle

kPlot = False

# Load stored parameter values (and corresponding loss)
allvals = []
allparams = []
mean_vals = []
N = 1
for i in range(N):
    vals = np.load('likelihood_vals_human_v' + str(i) + '.npy')
    params = np.load('parameters_human_v' + str(i) + '.npy', allow_pickle=True)
    allvals.append(np.array(vals))
    allparams.append(params)
    mean_vals.append(np.mean(np.array(allvals[i][-100:-1])))
val_idx = np.argsort(np.array(mean_vals))
omega_h = allparams[val_idx[0]][0][-1]
sigma_h = allparams[val_idx[0]][1][-1]
l_h = allparams[val_idx[0]][2][-1]

if (kPlot):
    colors = ['dimgray', 'g', 'b', 'brown', 'peru', 'k', 'skyblue', 'y','indigo', 'goldenrod', 'lawngreen', 'r', 'orange', 'b']
    # Plot loss function
    plt.subplot(131)
    for i in range(N):
        plt.plot(np.array(allvals[i]), color=colors[i])
        # plt.ylim(-2000, 20000)
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


# Load stored parameter values (and corresponding loss)      
allvals = []
allparams = []
mean_vals = []
N = 1
for i in range(N):
    vals = np.load('likelihood_vals_robot_v' + str(i) + '.npy')
    params = np.load('parameters_robot_v' + str(i) + '.npy', allow_pickle=True)
    allvals.append(np.array(vals))
    allparams.append(params)
    mean_vals.append(np.mean(np.array(allvals[i][-100:-1])))
val_idx = np.argsort(np.array(mean_vals))
omega_r = allparams[val_idx[0]][0][-1]
sigma_r = allparams[val_idx[0]][1][-1]
l_r = allparams[val_idx[0]][2][-1]

if (kPlot):
    colors = ['dimgray', 'g', 'b', 'brown', 'peru', 'k', 'skyblue', 'y','indigo', 'goldenrod', 'lawngreen', 'r', 'orange', 'b']
    # Plot loss function                                                                                                                  
    plt.subplot(131)
    for i in range(N):
        plt.plot(np.array(allvals[i]), color=colors[i])
        # plt.ylim(-2000, 20000)
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

# Save optimized hyperparameters
result_human = {'omega' : omega_h, 'sigma' : sigma_h, 'l' : l_h}
f = open("hyperparameters_human.pkl", "wb")
pickle.dump(result_human, f)
f.close()

result_robot = {'omega' : omega_r, 'sigma' : sigma_r, 'l' : l_r}
f = open("hyperparameters_robot.pkl", "wb")
pickle.dump(result_robot, f)
f.close()

'''
sigma_h = 10.75
l_h = 44.5
omega_h = np.array([[ 2.14168939e+00, -1.91468950e-03,  4.36759561e-03,  1.95112290e-03], [-1.92039823e-03, 2.14430236e+00, -7.22641086e-03,  6.12010022e-03], [ 3.09204576e-03, -7.36122232e-03,  9.24875327e-02,  5.49446192e-01], [ 2.21627719e-03,  5.49250113e-04, -5.32344418e-01,  1.14454823e-01]])

sigma_r = 0.06
l_r = 0.77
omega_r = np.array([[ 3.71130568e-01,  1.06507988e-01,  6.91692193e-03, -4.81956016e-03], [ 1.06507988e-01,  3.06215802e-02,  3.10027825e-03 , 1.09952609e-03], [ 6.91692193e-03,  3.10027825e-03,  1.34292990e+00,  1.29547135e-02], [-4.81956016e-03,  1.09952609e-03,  1.29547135e-02 , 1.37563698e+00]])

print(omega_r)
print(omega_h)
vr, wr = np.linalg.eig(omega_r)
vh, wh = np.linalg.eig(omega_h)
print(vr)
print(vh)
'''
