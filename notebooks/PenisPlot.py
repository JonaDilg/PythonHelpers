import numpy as np
from numpy import array as arr

import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pylandau


# ----------------------------------------------------------
ikrum = 4.57
corrected = False
ikrum_map = {3.30:"635", 4.57:"639", 8.96:"640"}
ikrum_name = "{:.2f} nA".format(ikrum) 
if corrected:
    dataPath = "/home/jona/DESY/analysis_python/input/"+ikrum_map[ikrum]+"_corrected.csv"
else:
    dataPath = "/home/jona/DESY/analysis_python/input/"+ikrum_map[ikrum]+".csv"

threshold = 0.015

dataRaw = np.genfromtxt(dataPath, delimiter=',', skip_header=1)

nEvents = int(dataRaw[:,0].size / 4)
nEntries = dataRaw[0].size
data = np.zeros([nEvents,4,nEntries])
map = [2,0,3,1]
for i in range(4):
    data[:,map[i],:] = dataRaw[i::4,]
    
    
    
# ----------------------------------------------------------
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import AxesImage

binRange = np.array([[20,220],[0,500]])
binN = np.array([50,50])

title = "ToT vs Amplitude"

fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(8,6))
ax = ax.flatten()

map = [1,3,0,2]
pix_names = ["00","01","10","11"]
for i in range(4):
    i_pix = map[i]
    mask = data[:,i_pix,7]>threshold
    # mask = np.ones(data[:,i_pix,7].size, dtype=bool)
    dataArrX = data[:,i_pix,7][mask] * 1000
    dataArrY = data[:,i_pix,10][mask]
    hist = ax[i].hist2d(dataArrX, dataArrY, bins=binN, norm=LogNorm(), range=binRange, cmap="viridis_r", zorder=100)
    
    ax[i].grid()
    ax[i].set_title("pix "+pix_names[i_pix] + ", N={:n}".format(hist[0].sum()))
    ax[i].tick_params("both", direction="in", top=True, right=True)
    
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes('right', size='5%', pad=0.03)
    im = ax[i].imshow(arr([dataArrX[:],dataArrY[:]]), cmap='viridis_r', norm=LogNorm())
    fig.colorbar(im, cax=cax, orientation='vertical', fraction=0.05, pad=0.01, anchor=(0.0,0.9))

for i in [2,3]:
    ax[i].set_xlabel("Amplitude [mV]", loc="right")
for i in [0,2]:
    ax[i].set_ylabel("ToT [ns]", loc="top")
    
# fig.text(0.96,0.95,"Sample ?, Fe55, krum_bias_trim="+ikrum_name+", i_krum=~2nA via Carboard,\nv_dummypix=350 mV, v_krumref=400 mV, ", ha="right", zorder=100, size="small")
if corrected:
    fig.text(0.96,0.96,r"Sample #10 | Fe55 Oct 2024"+"\n"+"krum_bias_trim="+ikrum_name+", i_krum=2nA via Carboard, bias=-3.6V\nBALLISTIC DEFICIT CORRETION VIA LINEAR FIT", ha="right", zorder=100, size="small")
else:
    fig.text(0.96,0.96,r"Sample #10 | Fe55 Oct 2024"+"\n"+"krum_bias_trim="+ikrum_name+", i_krum=2nA via Carboard, bias=-3.6V", ha="right", zorder=100, size="small")
# fig.suptitle("ER1 ToT vs Amplitude", ha="left", x=0, fontdict={})
fig.suptitle("DESYER1 - "+title, ha="left", x=0, fontweight="bold")
# fig.tight_layout()
name = "ToTvsA" + ("_corrected" if corrected else "") + ".pdf"
fig.savefig("/home/jona/DESY/analysis_python/output/"+name)