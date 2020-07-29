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
##
pickle.dump(pca_results, open('PCA_results.p', "wb"))
