# MZW 2020 07

import numpy as np
import matplotlib.pyplot as plt

class data_inspection_functions:

    def __init__(self):
        self.memo ='contians funcs for single neuron raster, ensemble raster, mean FR'


    def Raster_plot(raster, cn, areaName):
        for trl in range(raster.shape[0]):
            for bn in range(raster.shape[1]):
                fr_count = raster[trl, bn]
                if fr_count > 0:
                    for frn in range(fr_count):
                        gitter = np.random.random(1)
                        x1 = [trl, trl + 1]
                        x2 = [bn + gitter, bn + gitter]
                        plt.plot(x2, x1, color='black')

        plt.xlabel('Time Bin')
        plt.ylabel('Trial')
        plt.title(areaName + ' Cell ' + str(cn))
        pass

    def Raster_plot_all(ensembleSpk, areaName, subplotRowN, subplotColN, figureSupTitle):
        Celltotal = ensembleSpk.shape[0]
        fig, axs = plt.subplots(subplotRowN, subplotColN, sharex='all', sharey='all')
        fig.suptitle(figureSupTitle)
        for i in range(int(subplotRowN)):
            for j in range(int(subplotColN)):
                if i * subplotColN + j < Celltotal:
                    cn = i * subplotColN + j
                    print(cn)
                    raster = ensembleSpk[cn, :, :]
                    for trl in range(raster.shape[0]):
                        for bn in range(raster.shape[1]):
                            fr_count = raster[trl, bn]
                            if fr_count > 0:
                                for frn in range(fr_count):
                                    gitter = np.random.random(1)
                                    x1 = [trl, trl + 1]
                                    x2 = [bn + gitter, bn + gitter]
                                    axs[i, j].plot(x2, x1, color='black')
                                    axs[i, j].set_title('Cell ' + str(cn + 1), fontsize=8)
                                    if not (not (i == int(subplotRowN) - 1) or not (j == 0)):
                                        axs[i, j].set(xlabel='Time Bin', ylabel='Trials')


    def plot_mean_FR(ensembleSpk, figureTitle):
        for cn in range(ensembleSpk.shape[0]):
            raster = ensembleSpk[cn, :, :]
            meanFR = np.nanmean(raster, axis=0)
            plt.plot(meanFR)
        plt.title(figureTitle)



