import numpy as np
import matplotlib.pyplot as plt

def create_fig(figsize=(8,6)):
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=figsize)
    ax = ax.flatten()
    for i in range(4):
        ax[i].grid(True)
        ax[i].tick_params("both", direction="in", top=True, right=True)
    return fig, ax

def finalize_fig(single_run, fig, ax, xlabel, ylabel, title, logy=False):
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]

    for i in range(4):
        ax[i].grid(zorder = 0)
        ax[i].set_title("pix "+pix_names[map[i]])
        if len(ax[i].get_legend_handles_labels()[0]) > 0:
            ax[i].legend(prop={'size': 8})
        if logy:
            ax[i].set_yscale("log")
    ax[2].set_xlabel(xlabel, loc='right')
    ax[3].set_xlabel(xlabel, loc='right')
    ax[0].set_ylabel(ylabel, loc='top')
    ax[2].set_ylabel(ylabel, loc='top')
    if title:
        fig.suptitle("DESYER1 - "+title, ha="left", x=0, fontweight="bold")
    else:
        fig.suptitle("DESYER1 - "+xlabel, ha="left", x=0, fontweight="bold")
        
    draw_parameter_string_(single_run, fig)
        
    fig.tight_layout()
    fig.subplots_adjust(hspace=.15, wspace=.07)
    return
    
def get_parameter_string_(single_run):
    txt = "Sample {:}".format(single_run["sample"]) +" | {:}".format(single_run["data_type"]) +", {:}".format(single_run["setting"])
    txt += "\nkrum_bias_trim=" + str(single_run["krum_bias_trim"]) + "nA; i_krum=" + str(single_run["i_krum"]) + "; bias=" + str(single_run["bias_v"]) + "V"
    return txt  
    
def draw_parameter_string_(single_run, fig):
    txt = get_parameter_string_(single_run)
    fig.text(0.96, 0.96, txt, ha='right', fontsize=8)
    
    
    
    
    
    
    
def draw(single_run, fig, ax, dataColumn, binN=50, binRange=None, mask=None, color="black", label=None):
    
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]
    data = single_run["data"]
    
    if mask is None:
        mask = np.ones(data[:,:,dataColumn].shape, dtype=bool)
        
    for i, i_pix in enumerate(map):
        if binRange==None:
            binRange = [np.min(data[:,i_pix,dataColumn]), np.max(data[:,i_pix,dataColumn])]
        hist, bins, _ = ax[i].hist(data[:,i_pix,dataColumn][mask[:,i_pix]], bins=binN, range=binRange, histtype="step", lw=1.5, zorder=100, color=color, label=label+" (N={:.1f}k)".format(np.sum(mask[:,i_pix]/1000)))
        ax[i].stairs(hist, bins, zorder=50, fill=True, alpha=0.3, color=color, lw=0)
        ax[i].grid()
        ax[i].set_xlim(binRange)
        











# drawing multiple histograms on the same plot

def createMask(single_run, dataColumn, limits):
    data = single_run["data"]
    masks = []
    for i in range(len(limits)-1):
        masks.append(np.logical_and(data[:,:,dataColumn]>limits[i], data[:,:,dataColumn]<limits[i+1]))
    return masks

def drawStacked(single_run, fig, ax, dataColumn, binN=50, binRange=None, masks=None, colors=None, labels=None):
    
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]
    data = single_run["data"]
    
    if not colors:
        colors = np.array([["C{:d}".format(i)] for i in range(len(masks))]).flatten()
    if not masks:
        print("No masks given. Exiting drawStacked.")
    if not labels:
        labels = ["Mask {:d}".format(i) for i in range(len(masks))]

    for i, i_pix in enumerate(map):
        if binRange==None:
            binRange = [np.min(data[:,i_pix,dataColumn]), np.max(data[:,i_pix,dataColumn])]
        base = np.zeros(binN)
        
        for j, mask in enumerate(masks):
            hist, bins = np.histogram(data[:,i_pix,dataColumn][mask[:,i_pix]], bins=binN, range=binRange)
            
            ax[i].stairs(hist+base, bins, zorder=100-j, fill=True, alpha=1, color=colors[j], lw=0, label=labels[j]+" (N={:.1f}k)".format(np.sum(mask[:,i_pix]/1000)))
            
            # hist, bins, _ = ax[i].hist(data[:,i_pix,dataColumn][mask[:,i_pix]], bins=binN, range=binRange, histtype="step", lw=1.5, zorder=100, color=colors[j], label=label)
            base += hist
        ax[i].set_xlim(binRange)
        
def drawMultiple(single_run, fig, ax, dataColumn, binN=50, binRange=None, masks=None, colors=None, labels=None):
    
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]
    data = single_run["data"]
    
    if not colors:
        colors = np.array([["C{:d}".format(i)] for i in range(len(masks))]).flatten()
    if not masks:
        print("No masks given. Exiting drawStacked.")
    if not labels:
        labels = ["Mask {:d}".format(i) for i in range(len(masks))]

    for i, i_pix in enumerate(map):
        if binRange==None:
            binRange = [np.min(data[:,i_pix,dataColumn]), np.max(data[:,i_pix,dataColumn])]
        
        for j, mask in enumerate(masks):
            hist, bins = np.histogram(data[:,i_pix,dataColumn][mask[:,i_pix]], bins=binN, range=binRange)
            
            ax[i].stairs(hist, bins, zorder=100+j, fill=False, alpha=1, color=colors[j], lw=1.5, label=labels[j]+" (N={:.1f}k)".format(np.sum(mask[:,i_pix]/1000)))
            ax[i].stairs(hist, bins, zorder=100-len(masks)+j-1, fill=True, alpha=.3, color=colors[j], lw=0)
            
        ax[i].set_xlim(binRange)