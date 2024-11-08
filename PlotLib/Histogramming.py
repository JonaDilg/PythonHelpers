import numpy as np
import matplotlib.pyplot as plt

def create_fig(cols=2, rows=2, figsize=(8,6)):
    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=figsize)
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

def finalize(single_run, fig, ax, xlabel, ylabel, title, subtitles=["0"], measurement="TB", logy=False):
        
    if type(ax) is np.ndarray:
        for i in range(len(ax)):
            ax[i].grid(zorder = 0)
            if (subtitles != ["0"]) and (len(subtitles) == len(ax)) :
                ax[i].set_title(subtitles[i], {'size': 9})
            if len(ax[i].get_legend_handles_labels()[0]) > 0:
                ax[i].legend(prop={'size': 8})
            if logy:
                ax[i].set_yscale("log")
        
        # assumes exactly 2 columns of plots, any number of rows.
        for i in range(len(ax)-2, len(ax)):
            ax[i].set_xlabel(xlabel, loc='right')
        for i in range(0,len(ax),2):
            ax[i].set_ylabel(ylabel, loc='top')
    else:
        ax.grid(zorder = 0)
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(prop={'size': 8})
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, loc='right')
        ax.set_ylabel(ylabel, loc='top')
        
    if title:
        fig.suptitle("DESYER1 - "+title, ha="left", x=0, fontweight="bold")
    else:
        fig.suptitle("DESYER1 - "+xlabel, ha="left", x=0, fontweight="bold")
    
    if measurement=="TB":
        draw_parameter_string_(single_run, fig, showDict={"data_type":True})
    elif measurement=="Fe55_all":
        draw_parameter_string_(single_run, fig, showDict={"data_type":True, "krum_bias_trim":False})
    else:
        print("ERROR(Histogramming): measurement type \""+measurement+ "\" unknown.")
        
    fig.tight_layout()
    if type(ax) is np.ndarray:
        if len(ax)==8:
            fig.subplots_adjust(hspace=.25, wspace=.09)
        else:
            fig.subplots_adjust(hspace=.15, wspace=.09)
    return
    

def finalize_pixelwise(single_run, fig, ax, xlabel, ylabel, title, measurement="TB", logy=False):
    if len(ax) != 4:
        print("ERROR(Histogramming.py): supplied fig,ax does not have 4 subplots.")
        return
    map = [1,3,0,2] # map pixel 1-4 to the locations in the 2x2 grid
    pix_names = ["00","01","10","11"]

    subtitles = ["pix "+pix_names[map[i]] for i in range(4)]
    finalize(single_run, fig, ax, xlabel, ylabel, title, subtitles, measurement, logy)

    
# def get_parameter_string_TB_(single_run):
#     txt = "Sample {:}".format(single_run["sample"]) +" | {:}".format(single_run["data_type"]) +", {:}".format(single_run["setting"])
#     txt += "\nkrum_bias_trim=" + str(single_run["krum_bias_trim"]) + "nA; i_krum=" + str(single_run["i_krum"]) + "; bias=" + str(single_run["bias_v"]) + "V"
#     return txt  

# def get_parameter_string_Fe55_(single_run):
#     txt = "Sample {:}".format(single_run["sample"]) +" | {:}".format(single_run["data_type"])
#     txt += "\nkrum_bias_trim=" + str(single_run["krum_bias_trim"]) + "nA; i_krum=" + str(single_run["i_krum"]) + "; bias=" + str(single_run["bias_v"]) + "V"
#     return txt  
    
def get_parameter_string_(single_run, showDict={}):
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
    if(showDict["krum_bias_trim"]):
        txt += "krum_bias_trim=" + str(single_run["krum_bias_trim"]) + "nA"
        linefilled = True
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
    
def draw_parameter_string_(single_run, fig, showDict={}):
    txt = get_parameter_string_(single_run, showDict)
    fig.text(0.96, 0.96, txt, ha='right', fontsize=8)
        
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
        
def fit_wrapper(hist, bins, mask, fitfunc, p0, sigma=None):
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
    
    popt, pcov = curve_fit(fitfunc, bins, hist, p0=p0, sigma=sigma)
    
    perr = np.sqrt(np.diag(pcov))
    dx = fitfunc(bins, *popt) - hist
    chi2 = np.sum(dx**2 / hist)
    
    ndeg = len(hist) - len(popt)
        
    return popt, perr, chi2, ndeg