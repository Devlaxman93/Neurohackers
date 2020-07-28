# Maya Z Wang 07 2020

import pickle
import numpy as np
import matplotlib.pyplot as plt

from PCA_functions import *
funcs = PCA_functions

# use sessions: 13, 39 (actual session # not index)
## load only one session
dat = pickle.load(open("ses13.p", "rb"))
# dat = pickle.load(open("ses39.p", "rb"))
spk = dat["spks"]
# print(dat.keys())
##
allAreas = dat["brain_area"]
usedAreas = ['CA1', 'VISam', 'PL', 'MOs']
area = dict()
cellN = np.zeros(4)
for i in range(len(usedAreas)):
    areaId = allAreas == usedAreas[i]
    tmp_Area = spk[areaId, :, :]
    print(tmp_Area.shape)
    area[i] = funcs.frNormalization(tmp_Area)
    cellN[i] = tmp_Area.shape[0]

sampleN = int(np.min(cellN))
# print(np.unique(area[1]))
##
stepSize = 10
eigRep = 100
mcRep = 100
sigpc = dict()
sigITI = dict()
sigStim = dict()
sigCho = dict()

for ia in range(len(usedAreas)):

    # whole trial
    dd = funcs.moveMean_step(stepSize, area[ia])
    sigPC_n, sigEigVals, bstpEigVals = funcs.PCA_sigPC_est(dd, sampleN, eigRep, mcRep, 0)
    sigpc[ia, 0] = sigPC_n
    sigpc[ia, 1] = sigEigVals
    sigpc[ia, 2] = bstpEigVals

    # ITI epoch
    sigPC_n, sigEigVals, bstpEigVals = funcs.PCA_sigPC_est(area[ia], sampleN, eigRep, mcRep, 1)
    sigITI[ia, 0] = sigPC_n
    sigITI[ia, 1] = sigEigVals
    sigITI[ia, 2] = bstpEigVals

    # stimulus - choice epoch
    sigPC_n, sigEigVals, bstpEigVals = funcs.PCA_sigPC_est(area[ia], sampleN, eigRep, mcRep, 2, params=dat['gocue'])
    sigStim[ia, 0] = sigPC_n
    sigStim[ia, 1] = sigEigVals
    sigStim[ia, 2] = bstpEigVals

    # choice - feedback
    tmp = np.array([dat['gocue'], dat['feedback_time']])
    sigPC_n, sigEigVals, bstpEigVals = funcs.PCA_sigPC_est(area[ia], sampleN, eigRep, mcRep, 3, params=tmp)
    sigCho[ia, 0] = sigPC_n
    sigCho[ia, 1] = sigEigVals
    sigCho[ia, 2] = bstpEigVals


##
pca_results = dict()
pca_results[0] = sigpc
pca_results[1] = sigITI
pca_results[2] = sigStim
pca_results[3] = sigCho
# pickle.dump(pca_results, open('PCA_results', "wb"))
##
#
# ## ITI epoch
# aa = np.nanmean(area[1][:, :, 0:49], axis=2).T
# print(aa.shape)
# ## sti - choice epoch
# bb = np.zeros_like(aa)
# ix = np.round(dat['gocue'] * 1000 / 10) + 49
# for tn in range(len(dat['gocue'])):
#
#     bb[tn, :] = np.nanmean(area[1][:, tn, 50:int(ix[tn])], axis=1).T
# print(bb.shape)
# ## choice - feedback
# cc = np.zeros_like(aa)
# ix = np.round(dat['gocue'] * 1000 / 10 + 49)
# ip = np.round(dat['feedback_time'] * 1000 / 10 + 49)
# for tn in range(len(dat['gocue'])):
#     if ip[tn] > 250:
#         ip[tn] = 250
#     cc[tn, :] = np.nanmean(area[1][:, tn, int(ix[tn]):int(ip[tn])-1], axis=1).T
# print(cc.shape)
#



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

