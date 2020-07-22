# -*- coding: utf-8 -*-
"""
Created on Tue July 21st 2020

@author: dev
"""

"""PCA on Steinmetz data

Original file is located at
    https://colab.research.google.com/drive/14aT_xJZEnTYndO0B9MXxsoFEGzHFHYg8

"""

#@title Data retrieval
import os, requests
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fname = ['steinmetz_NMA_part1.npz']
fname.append('steinmetz_NMA_part2.npz')
url = ["https://osf.io/ex9zk/download"]
url.append("https://osf.io/cvjrf/download")

for j in range(2):
  if not os.path.isfile(fname[j]):
    try:
      r = requests.get(url[j])
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname[j], "wb") as fid:
          fid.write(r.content)

#@title Data loading

alldat = np.load('steinmetz_NMA_part1.npz', allow_pickle=True)['dat']
alldat = np.hstack((alldat, np.load('steinmetz_NMA_part2.npz', allow_pickle=True)['dat']))


# Rat ID: Lederberg Date: 2017-12-07
interested_session = alldat[13]

all_spikes = interested_session['spks']

#Grabbing the neurons from the brain region interested
RSP_neurons = [i for i,item in enumerate(interested_session['brain_area']) if "RSP" in item]

All_RSP_neurons = all_spikes[RSP_neurons[0],:,:]

for i in range(len(RSP_neurons)-1):

    RSP_neuron = all_spikes[RSP_neurons[i+1],:,:]

    All_RSP_neurons = np.dstack([All_RSP_neurons, RSP_neuron])

All_RSP_neurons = np.reshape(All_RSP_neurons, (67000, 84))

# Calculate firing rate
All_RSP_neurons = All_RSP_neurons*100

#Calculate Z-score
z_scored_All_RSP_neurons = stats.zscore(All_RSP_neurons, axis=0)

#PCA
pca_model = PCA(n_components=50)  # Initializes PCA
pc_comp = pca_model.fit_transform(z_scored_All_RSP_neurons)  # Performs PCA
principal_components = pca_model.components_
explained_variance = pca_model.explained_variance_

# Plot the PCA
plt.figure(figsize=(20,12))
plt.scatter(pc_comp[:,0], pc_comp[:,1], c='blue', alpha=0.5, s=4)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Steinmetz - Retrosplenial Cortex PCA')
plt.show()

