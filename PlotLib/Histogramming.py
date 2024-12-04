import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# entries can be either 
# - (a) a number -> generates n evenly spaced colors from the colormap
# - (b) an array -> maps the values in n to the colormap
def get_color_range(entries, invert=False, mapName="plasma", maxLightness=0.85):
    cmap = colormaps[mapName]
    if type(entries) == int or type(entries) == float:
        entries = np.linspace(0,maxLightness,entries)
        # linspace starts at 0, ends at maxLightness - nice.
    elif type(entries) == np.ndarray or type(entries) == list:
        entries = entries / np.max(entries) * maxLightness
    else:
        raise TypeError("n must be a number or a list/np.ndarray")
    colors  = cmap(entries)
    if invert:
        colors = colors[::-1]
    return colors

def create_fig(cols=2, rows=2, figsize=(8,6), sharex=True, sharey=True, width_ratios=None, height_ratios=None):
    fig, ax = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=figsize, width_ratios=width_ratios, height_ratios=height_ratios)
    if (rows==1) and (cols==1):
        ax.grid(True)
        ax.tick_params("both", direction="in", top=True, right=True)
    else:
        ax = np.array(ax)
        if ax.ndim > 1:
            ax = ax.flatten()
        for i in range(len(ax)):
            ax[i].grid(True)
            ax[i].tick_params("both", direction="in", top=True, right=True)
    return fig, ax

def finalize(single_run, fig, ax, xlabel, ylabel, title, subtitles=["0"], measurement="TB", logy=False, subplots_adjust=None, legend_loc="best", param_narrow=False, param_fontsize=8):
        
    
        
    if type(ax) is np.ndarray:
        for i in range(len(ax)):
            ax[i].grid(zorder = 0)
            if (subtitles != ["0"]) and (len(subtitles) == len(ax)) :
                ax[i].set_title(subtitles[i], {'size': 9})
            if len(ax[i].get_legend_handles_labels()[0]) > 0:
                ax[i].legend(prop={'size': 8}, loc=legend_loc)
            if logy:
                ax[i].set_yscale("log")
        
        # assumes exactly 2 columns of plots, any number of rows.
        for i in range(len(ax)-2, len(ax)):
            ax[i].set_xlabel(xlabel, loc='right')
        for i in range(0,len(ax),2):
            ax[i].set_ylabel(ylabel, loc='top')
        
        # suptitle_x0 = ax[0].get_position().x0
        # suptitle_x1 = np.array([ax[i].get_position().x1 for i in range(0,len(ax),2)]).max()
    else:
        ax.grid(zorder = 0)
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(prop={'size': 8}, loc=legend_loc)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, loc='right')
        ax.set_ylabel(ylabel, loc='top')

        # suptitle_x0 = ax.get_position().x0
        # suptitle_x1 = ax.get_position().x1
        
    axs = fig.get_axes()   
    # suptitle_x0 = np.array([ax.get_position().x0 for ax in axs]).min()
    suptitle_x0 = np.array([ax.get_tightbbox().transformed(fig.transFigure.inverted()).x0 for ax in axs]).min()
    suptitle_x1 = np.array([ax.get_tightbbox().transformed(fig.transFigure.inverted()).x1 for ax in axs]).max() - 0.03
        
    if title:
        fig.suptitle("DESYER1 - "+title, ha="left", va="top", fontweight="bold", x=suptitle_x0, y=0.995)
    else:
        fig.suptitle("DESYER1 - "+xlabel, ha="left", va="top", fontweight="bold", x=suptitle_x0, y=0.995)
    
    if measurement=="TB":
        draw_parameter_string_(single_run, fig, showDict={"data_type":True}, x=suptitle_x1, narrow=param_narrow, fontsize=param_fontsize)
    elif measurement=="Fe55_all":
        draw_parameter_string_(single_run, fig, showDict={"data_type":True, "krum_bias_trim":False}, x=suptitle_x1, narrow=param_narrow, fontsize=param_fontsize)
    else:
        print("ERROR(Histogramming): measurement type \""+measurement+ "\" unknown.")
        
    fig.tight_layout()
    if (subplots_adjust is not None and len(subplots_adjust)==2):
        fig.subplots_adjust(hspace=subplots_adjust[0], wspace=subplots_adjust[1], top=0.935)
    elif (subplots_adjust is not None and len(subplots_adjust)==3):
        fig.subplots_adjust(hspace=subplots_adjust[0], wspace=subplots_adjust[1], top=subplots_adjust[2])
    elif (type(ax) is np.ndarray):
        if len(ax)==8:
            fig.subplots_adjust(hspace=.15, wspace=.09, top=0.935)
        else:
            fig.subplots_adjust(hspace=.15, wspace=.09)
    return

def get_hist(run, dataString, binN=50, binRange=None, mask=None):
    dataColumn = run[runID]["M"][dataString]
    if mask is not None:
        hist, bins = np.histogram(run[dataString][mask], bins=binN, range=binRange)
    hist, bins = np.histogram(run[dataString], bins=binN, range=binRange)
    return hist, binss

def finalize_pixelwise(single_run, fig, ax, xlabel, ylabel, title, measurement="TB", logy=False):
    if len(ax) != 4:
        print("ERROR(Histogramming.py): supplied fig,ax does not have 4 subplots.")
        return
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]

    subtitles = ["pix "+pix_names[map[i]] for i in range(4)]
    finalize(single_run, fig, ax, xlabel, ylabel, title, subtitles, measurement, logy)

def get_parameter_string_(single_run, showDict={}, narrow=False):
    showDict = {"sample":True, "data_type":False, "krum_bias_trim":True, "i_krum":True, "bias":True} | showDict
    txt = ""
    linefilled = False
    if(showDict["sample"]):
        txt += "Sample {:}".format(single_run["sample"])
        linefilled = True
    if(showDict["data_type"]):
        if linefilled:
            txt += " | "
        txt += "{:}".format(single_run["data_type"])
        linefilled = True
    txt += "\n"
    linefilled = False
    if not narrow:
        if(showDict["krum_bias_trim"]):
            txt += "krum_trimming=" + str(single_run["krum_trim"]) + " DAC"
            linefilled = True
    else:
        if(showDict["krum_bias_trim"]):
            txt += "krum_trim=" + str(single_run["krum_trim"]) + " DAC\n"
            
    if(showDict["i_krum"]):
        if linefilled:
            txt += " | "
        txt += "i_krum=" + str(single_run["i_krum"])
        linefilled = True
    if(showDict["bias"]):
        if linefilled:
            txt += " | "
        txt += "bias=" + str(single_run["bias_v"]) + "V"
        linefilled = True
    return txt
    
def draw_parameter_string_(single_run, fig, showDict={}, x=0.95, narrow=False, fontsize=8):
    txt = get_parameter_string_(single_run, showDict, narrow)
    fig.text(x, 0.995, txt, ha='right', fontsize=fontsize, va="top")
        
def draw(ax, hist, bins, color="black", label=None, fill_alpha=0.2):
    ax.stairs(hist, bins, fill=False, alpha=1, color=color, lw=1.5, label=label)
    ax.stairs(hist, bins, fill=True, alpha=fill_alpha, color=color, lw=0)
def draw_pixelwise(single_run, ax, dataColumn, binN=50, binRange=None, mask=None, color="black", label=None):
    
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]
    data = single_run["data"]
    
    if mask is None:
        mask = np.ones(data[:,:,dataColumn].shape, dtype=bool)
        
    for i, i_pix in enumerate(map):
        if binRange==None:
            binRange = [np.min(data[:,i_pix,dataColumn]), np.max(data[:,i_pix,dataColumn])]
    
        Entries = np.sum(0.001*
            np.logical_and(
                mask[:,i_pix],
                np.logical_and(
                    data[:,i_pix,dataColumn]>binRange[0],
                    data[:,i_pix,dataColumn]<binRange[1]))) # in k
        hist, bins, _ = ax[i].hist(data[:,i_pix,dataColumn][mask[:,i_pix]], bins=binN, range=binRange, histtype="step", lw=1.5, zorder=100, color=color, label=label+" (N={:.1f}k)".format(Entries))
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
        
def fit_wrapper(hist, bins, mask, fitfunc, sigma=None, **kwargs):
    from scipy.optimize import curve_fit
    from scipy.stats import norm
    
    # fit function needs to be defined as f(x, *p)
    # p0 is the initial guess for the fit parameters
    # mask is the mask to apply to the data
    
    if sigma is None:
        sigma = np.sqrt(hist)
    
    mask_nonzero = hist>0
    mask = np.logical_and(mask, mask_nonzero)
    hist = hist[mask]
    bins = bins[:-1]+(bins[1]-bins[0])/2
    bins = bins[mask]
    sigma = sigma[mask]
    
    popt, pcov = curve_fit(fitfunc, bins, hist, sigma=sigma, check_finite=True, absolute_sigma=True, **kwargs)
    
    perr = np.sqrt(np.diag(pcov))
    dx = fitfunc(bins, *popt) - hist
    chi2 = np.sum(dx**2 / hist)
    
    ndeg = len(hist) - len(popt)
        
    return popt, perr, chi2, ndeg