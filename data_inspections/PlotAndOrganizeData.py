# use sessions: 13, 39 (actual session # not index)
# MZW 2020 07
## load only one session
import pickle
import numpy as np
from scipy import stats
##
dat = pickle.load(open("ses39.p", "rb"))
spk = dat["spks"]
# print(dat.keys())

## normalization
# log scaling can't work on neg or 0
# sqrt(raw_spk) --> z-score
def frNormalization(spk):
    spkz=np.sqrt(spk)
    spk_norm = np.zeros_like(spk)
    for cn in range(spkz.shape[0]):
        psth = spkz[cn, :, :]
        for trn in range(psth.shape[0]):
            # trlSpk = psth[trn, :] - np.nanmean(psth[trn, :])
            trlSpk = stats.mstats.zscore(psth[trn, :], nan_policy='omit')
            spk_norm[cn, trn, :] = trlSpk

            del trlSpk
        del psth
    return spk_norm
# from scipy import stats
# spk_z3=stats.mstats.zscore(spk, axis=2, ddof=0, nan_policy='omit') # normalization dimension & nan problem
##
allAreas = dat["brain_area"]
usedAreas = ['CA1', 'VISam', 'PL', 'MOs']
area=dict()
cellN=np.zeros(4)
for i in range(len(usedAreas)):
    areaId = allAreas == usedAreas[i]
    tmp_Area = spk[areaId, :, :]
    print(tmp_Area.shape)
    area[i] = frNormalization(tmp_Area)
    cellN[i]=tmp_Area.shape[0]

# spk_norm=frNormalization(spk)

##
# use plot function to plot raster for single cell, raster for ensemble, or cell-FR for ensemble
from data_inspection_functions import *
funcs = data_inspection_functions
iA = 2
subplotColN = 6
subplotRowN = int(np.ceil(area[iA].shape[0] / subplotColN))
figureSupTitle = 'Ses 39:  ' + usedAreas[iA]
funcs.Raster_plot_all(area[iA], usedAreas[iA], subplotRowN, subplotColN, figureSupTitle)
##
# smpCellN=int(np.min(cellN))
# MOs = area[len(area)-1]
# cell_i = np.random.choice(np.arange(0,area[0].shape[0]+1), smpCellN, replace=True)
# MOs_sampled = MOs[cell_i,:,:]
# for i in range(MOs_sampled.shape[1]):
#     if i == 0:
#         MOs_mat = np.rollaxis(MOs_sampled[:,i,:], 1, 0)
#     else:
#         tmp = np.rollaxis(MOs_sampled[:,1,:], 1, 0)
#         MOs_mat = np.append(MOs_mat, tmp, axis=0)


##
# export .mat just in case
# import scipy.io
# scipy.io.savemat('S39.mat', area)

