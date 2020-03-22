import numpy as np
import matplotlib.pyplot as plt

result1 = np.load('comparison_results_v1.npy', allow_pickle=True)
result2 = np.load('comparison_results_v2.npy', allow_pickle=True)
result3 = np.load('comparison_results_v3.npy', allow_pickle=True)
result4 = np.load('comparison_results_v4.npy', allow_pickle=True)
result5 = np.load('comparison_results_v5.npy', allow_pickle=True)
result6 = np.load('comparison_results_v6.npy', allow_pickle=True)
results = np.concatenate((result1, result2, result3, result4, result5, result6), axis=1)
results = results[:,0:1000]

success_robust, success_primal = results[0,:], results[1,:]
collision_robust, collision_primal = results[2,:], results[3,:]
L_robust, L_primal = results[4,:], results[5,:]
dist_robust, dist_primal = results[6,:], results[7,:]

N = results.shape[1]
print(N)
'''
for i in range(N):
    if (collision_robust[i] and not collision_primal[i]):
        print("Collision with robust but not primal in trial " + str(i) + " with distance " + str(dist_robust[i]) + " vs. " + str(dist_primal[i]))
'''

# print("Robust CBF number of successes: " + str(sum(success_robust)) + ". Primal CBF number of successes: " + str(sum(success_primal)))
# print("Robust CBF number of collisions: " + str(sum(collision_robust)) + ". Primal CBF number of collisions: " + str(sum(collision_primal)))
print("Robust CBF collisions rate: %.3f%%. Primal CBF collision rate: %.3f%%." %  (sum(collision_robust) / N, sum(collision_primal) / N))

idx_robust, idx_primal = np.where(success_robust == True)[0], np.where(success_primal == True)[0]
idx_success = np.intersect1d(idx_robust, idx_primal)
L_diff, L_std = np.mean(L_robust[idx_success] - L_primal[idx_success]), np.std(L_robust[idx_success] - L_primal[idx_success])
print("Mean +- Std Path Length Difference for Successful Trials:  %.3f  +-  %.3f." %  (L_diff, L_std))

coll_threshold = 5.0
idx_robust, idx_primal = np.where(collision_robust == True)[0], np.where(collision_primal == True)[0]
idx_collision = np.union1d(idx_robust, idx_primal)
a = 0.0*np.ones(len(idx_collision))
dist_robust_mean = np.mean(np.minimum(dist_robust[idx_collision] - coll_threshold, a))
dist_robust_std = np.std(np.minimum(dist_robust[idx_collision] - coll_threshold, a))
dist_primal_mean = np.mean(np.minimum(dist_primal[idx_collision] - coll_threshold, a))
dist_primal_std = np.std(np.minimum(dist_primal[idx_collision] - coll_threshold, a))
print("For collision trials, Average violation of CBF (robust) :  %.3f  +-  %.3f. Average violation of CBF (primal) :  %.3f  +-  %.3f." %  (dist_robust_mean, dist_robust_std, dist_primal_mean, dist_primal_std))

dist_primal[dist_primal > 4.99] = 6
dist_robust[dist_robust > 4.99] = 6
total_dist = dist_primal + dist_robust
sort_idx = np.argsort(total_dist)
dist_primal = dist_primal[sort_idx]
dist_robust = dist_robust[sort_idx]
X = np.arange(N)

plt.rcParams.update({'font.size': 18})
plt.scatter(X, dist_primal,s=100.0,facecolors='none',edgecolors=(0.3,0.3,1.0))
plt.scatter(X, dist_robust,s=100.0,facecolors='none',edgecolors=(1.0,0.5,0.0))
plt.plot([-3,200],[5,5],'--',linewidth=3.0,c='black')
plt.xlim([-3,170])
plt.ylim([0,5.5])
plt.title('Collision Distance vs. Trial')
plt.xlabel('Trial Index (sorted)')
plt.ylabel('Minimum Distance')
plt.legend(['Collision Threshold','Nominal CBF', 'Robust CBF'])
plt.tight_layout()
plt.show()