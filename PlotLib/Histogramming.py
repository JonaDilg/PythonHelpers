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

def create_fig(cols=2, rows=2, figsize=None, sharex=True, sharey=True, width_ratios=None, height_ratios=None):
    if figsize is None:
        if cols==1 and rows==1:
            figsize = (5,3.8)
        else:
            figsize = (8,6)
    fig, ax = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=figsize, width_ratios=width_ratios, height_ratios=height_ratios)
    if (rows==1) and (cols==1):
        ax.tick_params("both", direction="in", top=True, right=True)
    else:
        ax = np.array(ax)
        if ax.ndim > 1:
            ax = ax.flatten()
        for i in range(len(ax)):
            ax[i].tick_params("both", direction="in", top=True, right=True)
    return fig, ax

def get_hist(run, dataString, binN=50, binRange=None, mask=None):
    dataColumn = run[runID]["M"][dataString]
    if mask is not None:
        hist, bins = np.histogram(run[dataString][mask], bins=binN, range=binRange)
    hist, bins = np.histogram(run[dataString], bins=binN, range=binRange)
    return hist, bins

# --- finalize ---

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
    
    fig.savefig(path+filename+".pdf", bbox_inches='tight', pad_inches=0.)
    fig.savefig(path+filename+".png", bbox_inches='tight', pad_inches=0., dpi=dpi)

def finalize(single_run, fig, ax, xlabel, ylabel, title, subtitles=["0"], measurement="TB", logy=False, subplots_adjust=None, legend_loc="best", legend_fontdict={"size":"medium"}, param_narrow=True, param_fontsize=8, grid=False, tight_layout=False, title_linebreak=True, labelpad=1., thr=None, ER1=True):
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

        # suptitle_x0 = ax.get_position().x0
        # suptitle_x1 = ax.get_position().x1
    
    
    title_y = 1.0088
    # -- Suptitle -- 
    if ER1:
        title_str = r"$\bf{DESYER1}$"
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
    fig.text(0, title_y, title, ha='left', fontdict={"size":"large"}, va="bottom")
    
    # -- Parameters (top right) --
    suptitle_x1 = 1.
    if measurement=="TB":
        draw_parameter_string_(single_run, fig, showDict={"data_type":True, "thr":thr}, x=1., y=title_y, narrow=param_narrow, fontsize=param_fontsize)
    elif measurement=="Fe55_all":
        draw_parameter_string_(single_run, fig, showDict={"data_type":True, "krum_bias_trim":False, "thr":thr}, x=1., narrow=param_narrow, fontsize=param_fontsize)
    else:
        raise ValueError("ERROR(Histogramming): measurement type \""+measurement+ "\" unknown.")
        
    if tight_layout:
        fig.tight_layout()
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    if subplots_adjust is not None and type(ax) is np.ndarray:
        if len(subplots_adjust)==2:
            fig.subplots_adjust(hspace=subplots_adjust[0], wspace=subplots_adjust[1])
        else:
            raise ValueError("ERROR(Histogramming): subplots_adjust must be a list of 2 elements.")
    if type(ax) is np.ndarray:
        fig.subplots_adjust(hspace=0.05, wspace=0.02)
    return

def get_parameter_string_(single_run, showDict={}, narrow=False):
    showDict = {"sample":True, "data_type":False, "krum_bias_trim":True, "i_krum":True, "bias":True} | showDict
    txt = ""
    lineHasContent = False
    # first line
    if(showDict["sample"]):
        txt += "Sample {:}".format(single_run["sample"])
        lineHasContent = True
    if(showDict["data_type"]):
        if lineHasContent:
            txt += " | "
        txt += "{:}".format(single_run["data_type"])
        lineHasContent = True
    
    # line 2
    txt += "\n"
    lineHasContent = False
    if(showDict["i_krum"]):
        if lineHasContent:
            txt += " | "
        txt += "i_krum=" + str(single_run["i_krum"])
        lineHasContent = True
    if(showDict["bias"]):
        if lineHasContent:
            txt += " | "
        txt += "bias=" + str(single_run["bias_v"]) + r"$\,$V"
        lineHasContent = True
            
    # line 3   
    if (showDict["krum_bias_trim"]) or (showDict["thr"] is not None):
        txt += "\n"
    lineHasContent = False
    if(showDict["krum_bias_trim"]):
        txt += "krum_trim=" + str(single_run["krum_trim"]) + r"$\,$DAC"
        lineHasContent = True
    if showDict["thr"] is not None:
        if lineHasContent:
            txt += " | "
        txt += "thr=" + str(showDict["thr"]) + r"$\,e^-$"
        lineHasContent = True
            
    
            
    return txt
    
def draw_parameter_string_(single_run, fig, showDict={}, x=0.95, y=1.008, narrow=False, fontsize=8):
    txt = get_parameter_string_(single_run, showDict, narrow)
    fig.text(x, y, txt, ha='right', fontsize=fontsize, va="bottom")    

def finalize_pixelwise(single_run, fig, ax, xlabel, ylabel, title, measurement="TB", logy=False):
    if len(ax) != 4:
        print("ERROR(Histogramming.py): supplied fig,ax does not have 4 subplots.")
        return
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]

    subtitles = ["pix "+pix_names[map[i]] for i in range(4)]
    finalize(single_run, fig, ax, xlabel, ylabel, title, subtitles, measurement, logy)
    
# --- draw histogram ---

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