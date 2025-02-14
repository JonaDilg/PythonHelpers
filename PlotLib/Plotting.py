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

def create_fig(cols=1, rows=1, figsize=None, sharex=True, sharey=True, width_ratios=None, height_ratios=None, **kwargs):
    if figsize is None:
        figsize = (5+1.5*(cols-1),3.5+1.5*(rows-1))
    fig, ax = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=figsize, **kwargs)
    if (rows==1) and (cols==1):
        ax.tick_params("both", direction="in", top=True, right=True)
    else:
        ax = np.array(ax)
        if ax.ndim > 1:
            ax = ax.flatten()
        for i in range(len(ax)):
            ax[i].tick_params("both", direction="in", top=True, right=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    return fig, ax

def get_hist(run, dataString, binN=50, binRange=None, mask=None):
    dataColumn = run[runID]["M"][dataString]
    if mask is not None:
        hist, bins = np.histogram(run[dataString][mask], bins=binN, range=binRange)
    hist, bins = np.histogram(run[dataString], bins=binN, range=binRange)
    return hist, bins

# --- finalize ---

def finalize(single_run, fig, ax,
    title, xlabel, ylabel, subtitles=["0"],
    title_linebreak=True, ER1=True,
    legend_loc="best", legend_fontdict={"size":"medium"},
    param_dict={},
    grid=False, logy=False, subplots_adjust=None, labelpad=1.):
    
    # -- Suptitle & Parameters -- 
    
    title_x0 = 0.    
    title_y0 = 1.007
    
    title_x1 = 0.
    for child in fig.get_axes():
        title_x1 = max(title_x1, child.get_position().x1)
    title_x1 += 0.001
    
    if ER1:
        title_str = r"$\bf{DESY ER1}$"
        if title_linebreak:
            title_str += "\n"
        else:
            title_str += " – "
    else:
        title_str = ""
    if title:
        title_str += title
    else:
        title_str += xlabel
    fig.text(title_x0, title_y0, title_str, ha='left', fontdict={"size":"large"}, va="bottom", linespacing=1.7)
    
    param_dict = {"campaign":True, "sample":False, "ER1Param":True, "recoParam":False, "fontsize":8} | param_dict
    draw_parameter_string_(single_run, fig, showDict=param_dict, x=title_x1, y=title_y0)

    # -- Adjustments --
    
    if type(ax) is np.ndarray:
        for i in range(len(ax)):
            if grid:
                ax[i].grid(zorder = 0)
            if (subtitles != ["0"]) and (len(subtitles) == len(ax)) :
                ax[i].set_title(subtitles[i], {'size': 9})
            if len(ax[i].get_legend_handles_labels()[0]) > 0:
                ax[i].legend(prop=legend_fontdict, loc=legend_loc, frameon=False)
            if logy:
                ax[i].set_yscale("log")
        
        # assumes exactly 2 columns of plots, any number of rows.
        for i in range(len(ax)-2, len(ax)):
            ax[i].set_xlabel(xlabel, loc='right', labelpad=labelpad)
        for i in range(0,len(ax),2):
            ax[i].set_ylabel(ylabel, loc='top', labelpad=labelpad)
    else:
        if grid:
            ax.grid(zorder = 0)
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(prop=legend_fontdict, loc=legend_loc, frameon=False)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, loc='right', labelpad=labelpad)
        ax.set_ylabel(ylabel, loc='top', labelpad=labelpad)
    
    
    
    if subplots_adjust is not None and type(ax) is np.ndarray:
        if len(subplots_adjust)==2:
            fig.subplots_adjust(hspace=subplots_adjust[0], wspace=subplots_adjust[1])
        else:
            raise ValueError("ERROR(Histogramming): subplots_adjust must be a list of 2 elements.")
    if type(ax) is np.ndarray:
        fig.subplots_adjust(hspace=0.05, wspace=0.02)
        
    
        
    return

def pull_up_ax(fig, ax):
    left, bottom, widht, height = ax.get_position().bounds
    bottom = 1 - height
    ax.set_position([left, bottom, widht, height])

def savefig(fig, filename, path="/home/jona/DESY/analysis_python/output/", dpi=300):
    if not path.endswith("/"):
        path += "/"
    if filename.find(".pdf")>0:
        filename.replace(".pdf", "")
        print("WARNING(Histogramming): filename should not contain file extension. Removed .pdf")
    if filename.find(".png")>0:
        print("WARNING(Histogramming): filename should not contain file extension. Removed .png")
        filename.replace(".png", "")
    if filename.find(".jpg")>0:
        print("WARNING(Histogramming): filename should not contain file extension. Removed .jpg")
        filename.replace(".jpg", "")
    if filename.find(".")>0:
        print("WARNING(Histogramming): filename should not contain file extension (or periods). Removed everything after the period.")
        filename = filename.split(".")[0]
    
    fig.savefig(path+filename+".pdf", bbox_inches='tight', pad_inches=0.01, dpi=dpi)
    fig.savefig(path+filename+".png", bbox_inches='tight', pad_inches=0.01, dpi=dpi)

def draw_parameter_string_(single_run, fig, showDict, x=0.95, y=1.008):
    def add_entry_(txt, to_add, sep=" | "):
        if not len(txt)<=4:
            if not txt[-1]=="\n":
                txt += sep
        txt += to_add
        return txt
    
    minEntries = ["campaign", "sample", "ER1Param", "recoParam", "fontsize"]
    for entry in minEntries:
        if not entry in showDict:
            raise ValueError("ERROR(Histogramming): key \""+entry+"\" not found in showDict. Is needed to be set to True or False.")
    
    txt = r"$\,$"

    if (showDict["campaign"]):
        txt = add_entry_(txt, single_run["data_type"] + ", run " + str(single_run["runID"])) 
    if (showDict["sample"]):
        txt = add_entry_(txt, f"Sample {single_run["sample"]}")
    if (showDict["ER1Param"]):
        txt += r"$\,$"+"\n"
        txt = add_entry_(txt, "i_krum=" + str(single_run["i_krum"]))
        txt = add_entry_(txt, "bias=" + str(single_run["bias_v"]) + r"$\,$V")
        
        if (not "krum_trim" in showDict) or (showDict["krum_trim"] is not None) or (showDict["thr"] is not None):
            txt += r"$\,$"+"\n"
        if "krum_trim" not in showDict:
            txt = add_entry_(txt, "krum_trim=" + str(single_run["krum_trim"]) + r"$\,$DAC")
        elif showDict["krum_trim"] is not None:
            txt = add_entry_(txt, "krum_trim=" + str(showDict["krum_trim"]) + r"$\,$DAC")
        if "thr" not in showDict:
            raise ValueError("ERROR(Histogramming): key \"thr\" not found in showDict. Is needed if \"ER1Param\" is True. Set to None if not needed.")
        if showDict["thr"] is not None:
            txt = add_entry_(txt, "thr=" + str(showDict["thr"]) + r"$\,e^-$")
    if showDict["recoParam"]:
        txt += r"$\,$"+"\n"
        minEntries = ["edgeCut", "minTrkPlanes"]
        for entry in minEntries:
            if not entry in showDict:
                raise ValueError("ERROR(Histogramming): key \""+entry+"\" not found in showDict. Set to value, or None if not needed.")
        if showDict["edgeCut"] is not None:
            txt = add_entry_(txt, "SensorEdgeCut=" + str(showDict["edgeCut"]))
        if showDict["minTrkPlanes"] is not None:
            txt = add_entry_(txt, "minTrkPlanes=" + str(showDict["minTrkPlanes"]))

    fig.text(x, y, txt, ha='right', fontsize=showDict["fontsize"], va="bottom", linespacing=1.1)

    
# --- draw histogram ---

def draw(ax, hist, bins, color="black", label=None, fill_alpha=0.2, **kwargs):
    ax.stairs(hist, bins, fill=False, alpha=1, color=color, lw=1.5, label=label, **kwargs)
    ax.stairs(hist, bins, fill=True, alpha=fill_alpha, color=color, lw=0, **kwargs)
    
# --- drawing multiple histograms on the same plot ---

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
        
# --- fitting ---
        
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