
import pickle
import numpy as np
from scipy import stats
import tensortools as tt
import matplotlib.pyplot as plt

# use sessions: 13, 39 (actual session # not index)
## load only one session
dat = pickle.load(open("ses13.p", "rb"))
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

## figure out task parameters
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

##
print(np.unique(area[0]))
##
def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    # >>> m = np.arange(0,100,1).reshape((10,10))
    # >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    # >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray


##
# organize data: neuron by temporal (50 ms bins) by trial
new_n_bins = int(area[iA].shape[2]/5)
FR = dict()
for iA in range(len(area)):
  iN = area[iA].shape[0]
  iT = area[iA].shape[1]
  # take sum and divide by 0.1 to get firing rate in Hz
  tmp=bin_ndarray(area[iA],new_shape=(iN,iT,new_n_bins),operation='mean') / 0.1
  FR[iA]=np.rollaxis(tmp, 2, 1)

  print(FR[iA].shape)

##
# usedAreas = ['CA1', 'VISam', 'PL', 'MOs']
iA=3
X= FR[iA]
I, J, K,= FR[iA].shape
np.unique(FR[0])

##
# this script is for similarity score and error plot from cpd_ensemble.py (Alex- Github)

# Fit ensembles of tensor decompositions.
methods = (
  'cp_als',    # fits unconstrained tensor decomposition.
  # 'ncp_bcd',   # fits nonnegative tensor decomposition.
  # 'ncp_hals',  # fits nonnegative tensor decomposition.
)


ensembles = {}
for m in methods:
    ensembles[m] = tt.Ensemble(fit_method=m, fit_options=dict(tol=1e-4))
    ensembles[m].fit(X, ranks=range(1, 30), replicates=3)

# Plotting options for the unconstrained and nonnegative models.
plot_options = {
  'cp_als': {
    'line_kw': {
      'color': 'black',
      'label': 'cp_als',
    },
    'scatter_kw': {
      'color': 'black',
    },
  },
  'ncp_hals': {
    'line_kw': {
      'color': 'blue',
      'alpha': 0.5,
      'label': 'ncp_hals',
    },
    'scatter_kw': {
      'color': 'blue',
      'alpha': 0.5,
    },
  },
  'ncp_bcd': {
    'line_kw': {
      'color': 'red',
      'alpha': 0.5,
      'label': 'ncp_bcd',
    },
    'scatter_kw': {
      'color': 'red',
      'alpha': 0.5,
    },
  },
}
##
# Plot similarity and error plots.
R = 1

plt.figure()
for m in methods:
    tt.plot_objective(ensembles[m], **plot_options[m])
plt.axvline(x=R, linewidth=2.5, color='g')
plt.legend()

plt.figure()
for m in methods:
    tt.plot_similarity(ensembles[m], **plot_options[m])
plt.axvline(x=R, linewidth=2.5, color='g')
plt.legend()
plt.show()


##
# Fit CP tensor decomposition (two times).
U = tt.cp_als(X, rank=R, verbose=True)
V = tt.cp_als(X, rank=R, verbose=True)
##
# Compare the low-dimensional factors from the two fits.
fig, ax, po = tt.plot_factors(U.factors)

tt.plot_factors(V.factors, fig=fig)
fig.suptitle("raw models", y=1)
fig.tight_layout()

##
# Align the two fits and print a similarity score.
sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
# sim is the similarity score
print(sim)

# Plot the results again to see alignment.
fig, ax, po = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)
fig.suptitle("aligned models", y=1)
fig.tight_layout()

##
# U : KTensor Kruskal tensor to be plotted.
# plots : str or list One of {'bar','line','scatter'} to specify the type of plot for each factor. The default is 'line'.
# fig : matplotlib Figure object If provided, add plots to the specified figure.
# The figure must have a sufficient number of axes objects.
# axes : 2d numpy array of matplotlib Axes objects If provided, add plots to the specified figure.
# scatter_kw : dict or sequence of dicts Keyword arguments provided to scatterplots.
# If a single dict is provided, these options are broadcasted to all modes.
# line_kw : dict or sequence of dicts Keyword arguments provided to line plots.
# If a single dict is provided, these options are broadcasted to all modes.
# bar_kw : dict or sequence of dicts Keyword arguments provided to bar plots.
# If a single dict is provided, these options are broadcasted to all modes.
# **kwargs : dict
# Additional keyword parameters are passed to the `subplots(...)` function
# to specify options such as `figsize` and `gridspec_kw`.
# See `matplotlib.pyplot.subplots(...)` documentation for more info.

# neuron factor
tt.plot_factors(U.factors, plots='bar')
# temporal factor - within trial
tt.plot_factors(U.factors, plots='line')
# trial factor - across trial
tt.plot_factors(U.factors, plots='scatter')
