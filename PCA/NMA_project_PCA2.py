# Maya Z Wang 07 2020

# this program load the PCA permutation and bootstrap results

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# from PCA_functions import *
# funcs = PCA_functions
##
area = pickle.load(open('PCA_results.p', "rb"))
sigTrlPC = area[0]
sigITI = area[1]
sigStim = area[2]
sigCho = area[3]
# del area
## plot all
usedAreas = ['CA1', 'VISam', 'PL', 'MOs']
epochs = ['Whole Trial', 'Pre-Stim', 'Stim-Cho', 'Cho-Fdbk']

fig, axs = plt.subplots(4, 8)
fig.figsize = (20, 16)
for tn in range(len(area)):
    tests = area[tn]
    for an in range(len(usedAreas)):
        perm = tests[an, 1]
        btsp = tests[an, 2]
        eigRep = perm.shape[0]
        mcRep = btsp.shape[1]
        sigN = np.zeros([eigRep, mcRep])
        for i in range(eigRep - 1):
            aa = perm[i, :]
            for j in range(mcRep - 1):
                bb = btsp[i, j, :]
                sigN[i, j] = (aa > np.percentile(bb, 95)).sum()

        # eigenvalue plot
        bbb = np.matrix.flatten(bb)
        axs[an, tn*2].hist(bbb)
        aaa = np.matrix.flatten(aa)
        axs[an, tn*2].hist(aaa)
        # if tn==0:
        #     axs[an, 0].set(ylabel=usedAreas[an])
        if an==0:
            axs[0, tn*2].title.set_text(epochs[tn])
        # if tn==0 and an==0:
        #     axs[an, tn * 2].legend(('bootstrap: Eigenvalue', 'data: Eigenvalue'), loc="upper right")

        # plot sig dim
        sigDim = np.median(np.median(sigN, axis=1))
        axs[an, tn*2+1].hist(np.matrix.flatten(sigN))
        axs[an, tn*2+1].vlines(sigDim,0,6000, linestyles="dashed", colors="r")
        axs[an, tn*2+1].annotate('Sig_Dim=' + str(sigDim), xy=(0.1, 5000))

fig.legend(('bootstrap: Eigenvalue', 'data: Eigenvalue'), loc = 'upper right')
for ax, row in zip(axs[:, 0], usedAreas):
    ax.set_ylabel(row, size='large')
plt.show()



##

## pick one to look at
# usedAreas = ['CA1', 'VISam', 'PL', 'MOs']
iarea = 0
# sigITI sigStim sigCho sigTrlPC
dd = sigITI
perm = dd[iarea, 1]
btsp = dd[iarea, 2]
eigRep = perm.shape[0]
mcRep = btsp.shape[1]
##
sigN = np.zeros([eigRep, mcRep])
for i in range(eigRep-1):
    aa = perm[i, :]
    for j in range(mcRep-1):
        bb = btsp[i, j, :]
        sigN[i, j] = (aa > np.percentile(bb, 95)).sum()

##
bbb = np.matrix.flatten(bb)
plt.hist(bbb)
aaa = np.matrix.flatten(aa)
plt.hist(aaa)
plt.legend(('bootstrap: Eigenvalue','data: Eigenvalue'))
# plt.title('Pre-stimuli: ' + usedAreas[iarea])
# plt.title('Stimuli-Choice: ' + usedAreas[iarea])
# plt.title('Choice-Feedback: ' + usedAreas[iarea])
plt.title('Whole trial: ' + usedAreas[iarea])
##
sigDim = np.median(np.median(sigN, axis=1))
plt.hist(np.matrix.flatten(sigN))
plt.title('Significant dimension: ' + str(sigDim), y=1.0)

# plt.suptitle('Pre-stimuli: ' + usedAreas[iarea])
# plt.suptitle('Stimuli-Choice: ' + usedAreas[iarea])
# plt.suptitle('Choice-Feedback: ' + usedAreas[iarea])
plt.suptitle('Whole trial: ' + usedAreas[iarea])

##  ############## figure out task parameters #################
# left recorded
rp = np.array((dat['response']))

choose_l = np.array((dat['response'] == 1))
choose_r = np.array((dat['response'] == -1))

nogo = np.array((dat['response'] == 0))
go = np.array((dat['response'] != 0))

contrast_l = dat['contrast_left']
contrast_r = dat['contrast_right']

contrast_l_high = contrast_r < contrast_l
contrast_r_high = contrast_r > contrast_l
contrast_equal = contrast_r == contrast_l

correct = np.zeros_like(choose_l)
for i in range(len(choose_l)):
    if contrast_l[i] > contrast_r[i] and rp[i] == 1:
        correct[i] = 1
    elif contrast_l[i] < contrast_r[i] and rp[i] == -1:
        correct[i] = 1
    elif contrast_l[i] == contrast_r[i] and rp[i] == 0:
        correct[i] = 1
