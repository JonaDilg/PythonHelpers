import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import matplotlib as mpl
mpl.rcParams["font.serif"] = "CMU serif"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "CMU serif"
mpl.rcParams["mathtext.it"] = "CMU serif:italic"
mpl.rcParams["mathtext.bf"] = "CMU serif:bold"
mpl.rcParams["font.family"] = "serif"

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

def create_fig(cols=1, rows=1, figsize=None, sharex=True, sharey=True, flatten=True, **kwargs):
    if figsize is None:
        if cols==1 and rows==1:
            figsize = [3.5,2.5]
        elif cols==2 and rows==2:
            figsize = [6,4.5]
        elif cols==2 and rows==4:
            figsize = [6,7.5]
        else:
            figsize = (5+1.5*(cols-1),3.5+1.5*(rows-1))
    fig, ax = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=figsize, **kwargs)
    if (rows==1) and (cols==1):
        ax.tick_params("both", direction="in", top=True, right=True)
    else:
        ax = np.array(ax)
        if flatten:
            ax = ax.flatten()
            for i in range(len(ax)):
                ax[i].tick_params("both", direction="in", top=True, right=True)
        else:
            for i in range(rows):
                for j in range(cols):
                    ax[i,j].tick_params("both", direction="in", top=True, right=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    return fig, ax

def get_hist(run, dataString, binN=50, binRange=None, mask=None):
    dataColumn = run[runID]["M"][dataString]
    if mask is not None:
        hist, bins = np.histogram(run[dataString][mask], bins=binN, range=binRange)
    hist, bins = np.histogram(run[dataString], bins=binN, range=binRange)
    return hist, bins

# --- finalize ---

def finalize_noRun(fig, ax,
    title, xlabel, ylabel, subtitles=None,
    title_linebreak=True, ER1=True,
    legend_loc="best", legend_fontdict={"size":"medium"}, legend_borderaxespad=None,
    grid=False, logy=False, subplots_adjust=None, labelpad=1.):
    
    # -- Suptitle --
    
    if title is not None:
        title_x0 = 0.    
        title_y0 = 1.007
        if ER1:
            title_str = r"$\bf{DESY\;chip\;V2}$"
            if title_linebreak:
                title_str += "\n"
            else:
                title_str += " – "
        else:
            title_str = ""
        title_str += title
        fig.text(title_x0, title_y0, title_str, ha='left', fontdict={"size":"large"}, va="bottom", linespacing=1.3)
    
    
    # -- Adjustments --
    
    if type(ax) is np.ndarray:
        for i in range(len(ax)):
            if grid:
                ax[i].grid(zorder = 0)
            if (subtitles is not None) and (len(subtitles) == len(ax)) :
                # ax[i].set_title(subtitles[i], {'size': 9})
                ax[i].text(0.5, 0.96, subtitles[i], ha='center', va='top', transform=ax[i].transAxes, fontdict={"size":"medium"})
            if (legend_loc is not None) and (len(ax[i].get_legend_handles_labels()[0]) > 0):
                ax[i].legend(prop=legend_fontdict, loc=legend_loc, frameon=False, borderaxespad=legend_borderaxespad)
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
        if (legend_loc is not None) and (len(ax.get_legend_handles_labels()[0]) > 0):
            ax.legend(prop=legend_fontdict, loc=legend_loc, frameon=False, borderaxespad=legend_borderaxespad)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, loc='right', labelpad=labelpad)
        ax.set_ylabel(ylabel, loc='top', labelpad=labelpad)
        
    if not (subplots_adjust is None):
        if type(ax) is np.ndarray:
            if len(subplots_adjust)==2:
                fig.subplots_adjust(hspace=subplots_adjust[0], wspace=subplots_adjust[1])
            else:
                raise ValueError("ERROR(Histogramming): subplots_adjust must be a list of 2 elements.")
    if type(ax) is np.ndarray:
        fig.subplots_adjust(hspace=0.05, wspace=0.02)

def finalize(single_run, fig, ax,
    title, xlabel, ylabel, subtitles=None,
    title_linebreak=True, ER1=True,
    legend_loc="best", legend_fontdict={"size":"medium"}, legend_borderaxespad=None,
    param_dict={},
    grid=False, logy=False, subplots_adjust=None, labelpad=1.):
    
    # -- Parameters -- 
    
    title_x1 = 0.
    for child in fig.get_axes():
        title_x1 = max(title_x1, child.get_position().x1)
    title_x1 += 0.001
    
    title_y0 = 1.007
    
    param_dict = {"campaign":True, "run":True, "sample":False, "ER1Param":True, "recoParam":False, "fontsize":8} | param_dict
    draw_parameter_string_(single_run, fig, showDict=param_dict, x=title_x1, y=title_y0)

    # -- call finalize_noRun --
    
    finalize_noRun(fig, ax,
    title, xlabel, ylabel, subtitles,
    title_linebreak=title_linebreak, ER1=ER1,
    legend_loc=legend_loc, legend_fontdict=legend_fontdict, legend_borderaxespad=legend_borderaxespad,
    grid=grid, logy=logy, subplots_adjust=subplots_adjust, labelpad=labelpad)

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
        txt = add_entry_(txt, single_run["data_type"])
    if (showDict["run"]):
        txt = add_entry_(txt, "run " + str(single_run["runID"]))
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

def draw(ax, hist, bins, color="black", label=None, fill_alpha=0.2, lw=None, **kwargs):
    ax.stairs(hist, bins, fill=False, alpha=1, color=color, lw=lw, label=label, **kwargs)
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