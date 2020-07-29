# Maya Z Wang 2020 07

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



class PCA_functions:

    def __init__(self):
        self.memo = 'functions: normalization; movingAverage; PCA'

    # normalization
    # log scaling can't work on neg or 0
    # sqrt(raw_spk) --> z-score
    def frNormalization(spk):
        spkz = np.sqrt(spk)
        spk_norm = np.zeros_like(spk)
        for cn in range(spkz.shape[0]):
            psth = spkz[cn, :, :]
            for trn in range(psth.shape[0]):
                # trlSpk = psth[trn, :] - np.nanmean(psth[trn, :])
                trlSpk = stats.mstats.zscore(psth[trn, :], nan_policy='omit')
                # scale to Hz
                spk_norm[cn, trn, :] = trlSpk

                del trlSpk
            del psth
        return spk_norm

    # from scipy import stats
    # spk_z3=stats.mstats.zscore(spk, axis=2, ddof=0, nan_policy='omit') # normalization dimension & nan problem

    # calculate mean FR in Hz across stepSize time bins (temporal, within trial)
    def moveMean_step(stepSize, data):
        starts = np.arange(0, data.shape[2], stepSize)
        FR = np.zeros((data.shape[0], data.shape[1], len(starts)))
        for cn in range(data.shape[0]):
            psth = data[cn, :, :]
            for trn in range(psth.shape[0]):
                fr = np.zeros((data.shape[1], len(starts)))
                for tp in range(len(starts) - 1):
                    fr[trn, tp] = np.nanmean(psth[trn, starts[tp]:starts[tp] + stepSize]) / 0.01
                    # scale to Hz
                FR[cn, :, :] = fr

            del psth
        return FR

    # PCA
    def get_sample_cov_matrix(X):
        """
      Returns the sample covariance matrix of data X

      Args:
        X (numpy array of floats) : Data matrix each column corresponds to a
                                    different random variable

      Returns:
        (numpy array of floats)   : Covariance matrix
      """

        # Subtract the mean of X % columns are neurons so mean for each neuron
        X = X - np.mean(X, 0)
        # Calculate the covariance matrix (hint: use np.matmul)
        cov_matrix = 1 / len(X) * X.T @ X

        return cov_matrix

    def sort_evals_descending(evals, evectors):
        """
      Sorts eigenvalues and eigenvectors in decreasing order. Also aligns first two
      eigenvectors to be in first two quadrants (if 2D).

      Args:
        evals (numpy array of floats)    : Vector of eigenvalues
        evectors (numpy array of floats) : Corresponding matrix of eigenvectors
                                            each column corresponds to a different
                                            eigenvalue

      Returns:
        (numpy array of floats)          : Vector of eigenvalues after sorting
        (numpy array of floats)          : Matrix of eigenvectors after sorting
      """

        index = np.flip(np.argsort(evals))
        evals = evals[index]
        evectors = evectors[:, index]
        if evals.shape[0] == 2:
            if np.arccos(np.matmul(evectors[:, 0],
                                   1 / np.sqrt(2) * np.array([1, 1]))) > np.pi / 2:
                evectors[:, 0] = -evectors[:, 0]
            if np.arccos(np.matmul(evectors[:, 1],
                                   1 / np.sqrt(2) * np.array([-1, 1]))) > np.pi / 2:
                evectors[:, 1] = -evectors[:, 1]
        return evals, evectors

    def pca(X):
        """
      Sorts eigenvalues and eigenvectors in decreasing order.

      Args:
        X (numpy array of floats): Data matrix each column corresponds to a
                                   different random variable

      Returns:
        (numpy array of floats)  : Data projected onto the new basis
        (numpy array of floats)  : Vector of eigenvalues
        (numpy array of floats)  : Corresponding matrix of eigenvectors

      """
        # Subtract the mean of X
        X = X - np.mean(X, 0)
        # Calculate the sample covariance matrix
        cov_matrix = PCA_functions.get_sample_cov_matrix(X)
        # Calculate the eigenvalues and eigenvectors
        evals, evectors = evals, evectors = np.linalg.eigh(cov_matrix)
        # Sort the eigenvalues in descending order
        evals, evectors = PCA_functions.sort_evals_descending(evals, evectors)
        # Project the data onto the new eigenvector basis
        score = X @ evectors

        return score, evectors, evals

    # #############  PCA: significant component estimation #############
    # funcs.moveMean_step take average in across 100 ms (10 time bins) for each trial and each cell
    # for each repetition, each region
    #   down sample to the smallest cell number (min-Cell-num) in the session
    #   Permutation, min-Cell-num without replacement -> PCA --> sig test
    #   Bootstrap min-Cell-num with replacement -> PCA --> against distribution

    def PCA_sigPC_est(dd, sampleN, eigRep, mcRep, eventAvgd, params=0):
        # dd = data, organized in time (time bin by trial) by cell number matrix
        # sampleN = min-Cell-num
        # eigRep: times of permutation test
        # mcRep: times of mc-bootstrap test
        c_array = np.arange(0, dd.shape[0])
        sigEigVals = np.zeros([eigRep, sampleN])
        bstpEigVals = np.zeros([eigRep, mcRep, sampleN])
        sigPC_n = np.zeros([eigRep])

        for ei in np.arange(0, eigRep):
            print('ei is ' + str(ei))
            ind = np.random.choice(c_array, sampleN, replace=False)
            FR = dd[ind, :, :]
            if eventAvgd == 0:
                for i in range(FR.shape[1]):
                    if i == 0:
                        fr_mat = np.rollaxis(FR[:, i, :], 1, 0)
                    else:
                        tmp = np.rollaxis(FR[:, i, :], 1, 0)
                        fr_mat = np.append(fr_mat, tmp, axis=0)
            elif eventAvgd == 1:
                # ITI
                fr_mat = np.nanmean(FR[:, :, 0:49], axis=2).T
            elif eventAvgd == 2:
                # stimulus - choice , params = dat['gocue']
                fr_mat = np.zeros([FR.shape[1], sampleN])
                ix = np.round(params * 1000 / 10) + 49
                for tn in range(len(params)):
                    fr_mat[tn, :] = np.nanmean(FR[:, tn, 50:int(ix[tn])], axis=1).T
            elif eventAvgd == 3:
                # choice -  feedback params = np.array([dat['gocue'], dat['feedback_time']])
                fr_mat = np.zeros([FR.shape[1], sampleN])
                ix = np.round(params[0, :] * 1000 / 10 + 49)
                ip = np.round(params[1, :]  * 1000 / 10 + 49)
                for tn in range(len(FR)):
                    if ip[tn] > 250:
                        ip[tn] = 250
                    fr_mat[tn, :] = np.nanmean(FR[:, tn, int(ix[tn]):int(ip[tn]) - 1], axis=1).T

            _, _, evals = PCA_functions.pca(fr_mat)
            sigEigVals[ei, :] = evals

            # bootstrap
            for bi in np.arange(0, mcRep):
                ind = np.random.choice(c_array, sampleN, replace=True)
                FR = dd[ind, :, :]
                if eventAvgd == 0:
                    for i in range(FR.shape[1]):
                        if i == 0:
                            fr_mat = np.rollaxis(FR[:, i, :], 1, 0)
                        else:
                            tmp = np.rollaxis(FR[:, i, :], 1, 0)
                            fr_mat = np.append(fr_mat, tmp, axis=0)
                elif eventAvgd == 1:
                    # ITI
                    fr_mat = np.nanmean(FR[:, :, 0:49], axis=2).T
                elif eventAvgd == 2:
                    # stimulus - choice , params = dat['gocue']
                    fr_mat = np.zeros([FR.shape[1], sampleN])
                    ix = np.round(params * 1000 / 10) + 49
                    for tn in range(len(params)):
                        fr_mat[tn, :] = np.nanmean(FR[:, tn, 50:int(ix[tn])], axis=1).T
                elif eventAvgd == 3:
                    # choice -  feedback params = np.array([dat['gocue'], dat['feedback_time']])
                    fr_mat = np.zeros([FR.shape[1], sampleN])
                    ix = np.round(params[0, :] * 1000 / 10 + 49)
                    ip = np.round(params[1, :] * 1000 / 10 + 49)
                    for tn in range(len(FR)):
                        if ip[tn] > 250:
                            ip[tn] = 250
                        fr_mat[tn, :] = np.nanmean(FR[:, tn, int(ix[tn]):int(ip[tn]) - 1], axis=1).T

                _, _, evals = PCA_functions.pca(fr_mat)
                bstpEigVals[ei, bi, :] = evals

        sigPC_n[ei] = (sigEigVals[ei, :] > np.percentile(np.mean(bstpEigVals[ei, :, :], axis=0), 95)).sum()


        return sigPC_n, sigEigVals, bstpEigVals

