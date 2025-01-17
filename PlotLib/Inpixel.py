import numpy as np
from numpy import array as arr
import uncertainties as unc
from uncertainties import unumpy as unp
from uncertainties.unumpy import uarray as uarr
from uncertainties.unumpy import nominal_values as val
from uncertainties.unumpy import std_devs as dev
from uncertainties import ufloat as uf
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def uzarr(shape):
    return unp.uarray(np.zeros(shape), np.zeros(shape))

class Hist_2D:
    def __init__(self, bins, binRange=None):
        _, self.binsX, self.binsY = np.histogram2d([-10,10], [-10,10], bins=bins, range=binRange)
        self.val = np.zeros((len(self.binsX)+1,len(self.binsY)+1))
        self.err = np.zeros((len(self.binsX)+1,len(self.binsY)+1))
        
    def isNumber(self, val):
        return isinstance(val, (float, np.floating, int, np.integer))
    
    def getBinRangeX(self):
        return arr([self.binsX[0],self.binsX[-1]])
    def getBinRangeY(self):
        return arr([self.binsY[0],self.binsY[-1]])
    
    def getBinNX(self):
        return len(self.binsX)-1
    def getBinNY(self):
        return len(self.binsY)-1
    def getBinN(self):
        return self.binNX()*self.binNY()\
            
    def getBinWidthX(self, binX=1):
        return self.binsX[binX+1]-self.binsX[binX]
    def getBinWidthY(self, binY=1):
        return self.binsY[binY+1]-self.binsY[binY]
            
    def setBin(self, binX, binY, value, err=None): 
        self.val[binX,binY] = value
        if err==None:
            self.poisson_err_(binX, binY)
        else:
            self.err[binX, binY] = err
            
    def getBinsX(self):
        return self.binsX
            
    def getBin(self, x, y):
        return self.getBinCol(x), self.getBinRow(y)
    def getBinCol(self, x):
        try:
            return arr(np.where(np.logical_and(x>=self.binsX[:-1], x<self.binsX[1:]))).flatten()[0]+1
        except:
            return (0 if x<self.binsX[0] else len(self.binsX))
    def getBinRow(self, y):
        try:
            return arr(np.where(np.logical_and(y>=self.binsY[:-1], y<self.binsY[1:]))).flatten()[0]+1
        except:
            return (0 if y<self.binsY[0] else len(self.binsY))
    
    def fill(self, x, y, w=1):
        if self.isNumber(x) and self.isNumber(y):
            binX = self.getBinCol(x)
            binY = self.getBinRow(y)
            self.val[binX, binY] += w
            self.poisson_err_(binX, binY)
            return
        if len(x)!=len(y):
            print("[ERROR] Hist_2D.fill(): x and y need to have the same length. The lengths are:", len(x), len(y))
            return
        for i in range(len(x)):
            binX = self.getBinCol(x[i])
            binY = self.getBinRow(y[i])
            self.val[binX, binY] += w
            self.poisson_err_(binX, binY)
    
    def getBinContent(self, col, row):
        return self.val[col, row]
        
    def hist(self):
        return self.val[1:-1, 1:-1]
    def histErr(self):
        return np.sqrt(self.err[1:-1, 1:-1])
    
    def calcErr_poisson(self, binX, binY):
        self.err[binX,binY] = np.sqrt(abs(self.val[binX,binY]))
        
    def draw(self, ax, ax_cbar=None, extent=None, cmap="viridis", cbar_label=None, overflow=False, **kwargs):
        if not overflow:
            X = self.hist()
            im = ax.imshow(self.hist().T, extent=[self.binsX[0],self.binsX[-1],self.binsY[0],self.binsY[-1]], origin='lower', **kwargs)
        else: 
            X = self.val
            widthX = self.getBinWidthX(0)
            widthY = self.getBinWidthY(0)
            im = ax.imshow(X.T, extent=[self.binsX[0]-widthX, self.binsX[-1]+widthX, self.binsY[0]-widthY, self.binsY[-1]+widthY], origin='lower', **kwargs)
            ax.plot([self.binsX[0],self.binsX[-1],self.binsX[-1],self.binsX[0],self.binsX[0]], [self.binsY[0],self.binsY[0],self.binsY[-1],self.binsY[-1],self.binsY[0]], color='black', linestyle=":")
        ax_cbar = inset_axes(ax, width="5%", height="100%", loc = 'lower left',
            bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
            borderpad = 0)
        #     ax_cbar = ax.inset_axes(bounds = (0.95, 0., 1, 0.05))
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label(cbar_label, loc="top")
        ax_cbar.tick_params(axis="y", direction="in")
        return im, ax_cbar

    def printOverflow(self):
        print("Overflow:")
        
        print(uf(self.val[0,-1], self.err[0,-1]), "|", uarr(self.val[1:-1,0], self.err[1:-1,0]).sum(), "|", uf(self.val[-1,-1], self.err[-1,-1]))
        print(uarr(self.val[0,1:-1], self.err[0,1:-1]).sum(), "|", uarr(self.val[1:-1,1:-1], self.err[1:-1,1:-1]).sum(), "|", uarr(self.val[-1,1:-1], self.err[-1,1:-1]).sum())

class Plot_2D(Hist_2D):
    def __init__(self, bins, binRange=None, mode="Mean"):
        super().__init__(bins, binRange)
        self.mode = mode
        self.data = np.empty([len(self.binsX)+1,len(self.binsY)+1], dtype=object)
        for binX in range(len(self.binsX)+1):
            for binY in range(len(self.binsY)+1):
                self.data[binX, binY] = arr([])
                
    def getBinData(self, col, row):
        return self.data[col, row]
    
    
    def fill_entries(self, xs, ys, zs, MPV_binN=None, MPV_binRange=None, MPV_removeTail=0.95, Bootstrap_N=1000, Bootstrap_draw=False):
        # dataset of tuples of (x,y,z). x,y define the bin, z is averaged for each bin
        if not ((len(xs)==len(ys)) and (len(ys)==len(zs))):
            print("[ERROR] Plot_2D(): fill_entries needs x,y,z arrays of the same lengths. The lengths are:", len(xs), len(ys), len(zs))
        xs = [self.getBinCol(x) for x in xs]
        ys = [self.getBinRow(y) for y in ys]
        for i in range(len(xs)):
            self.data[xs[i],ys[i]] = np.append(self.data[xs[i],ys[i]], [zs[i]]) # appends each z to the list of entries in its corresponding bin
        self.fill_hist_(MPV_binN, MPV_binRange, MPV_removeTail, Bootstrap_N, Bootstrap_draw)
        
    def fill_hist_(self, MPV_binN=None, MPV_binRange=None, removeTail=0.95, Bootstrap_N=1000, Bootstrap_draw=False):
        for binX in range(len(self.binsX)+1):
            for binY in range(len(self.binsY)+1):
                if len(self.data[binX, binY]>0):
                    if self.mode == "Mean":
                        temp = self.getBinMean(binX, binY, removeTail)
                        val = temp.n
                        err = temp.s
                    elif self.mode == "MPV":
                        temp = self.getBinMPV(binX, binY, MPV_binN, MPV_binRange, removeTail)
                        val = temp.n
                        err = temp.s
                    elif self.mode == "MedianBootstrap":
                        if len(self.data[binX, binY]) < 100:
                            print("[WARN] Plot_2D.fill_hist_(): MedianBootstrap needs at least 100 entries per bin")
                        temp = self.getBinMedianBootstrap(binX, binY, Bootstrap_N, Bootstrap_draw)
                        val = temp.n
                        err = temp.s
                    self.setBin(binX, binY, val, err**2)
                else: 
                    self.setBin(binX, binY, 0.,0.)
    
    def getBinMean(self, col, row, removeTail=0.95):
        entries = self.data[col,row]
        if removeTail is not None:
            entries.sort()
            entries = entries[:int(len(entries)*removeTail)]
        v = np.mean(entries)
        e = np.std(entries)
        return uf(v,e)
        
    def getBinMPV(self, col, row, binN=None, binRange=None, removeTail=0.95):
        entries = self.data[col,row]
        if removeTail is not None:
            entries.sort()
            entries = entries[:int(len(entries)*removeTail)]
        if binN is None and binRange is None:
            hist, bins = np.histogram(entries, bins=max(int(np.sqrt(len(entries))),100))
        elif binN is None and binRange is not None:
            hist, bins = np.histogram(entries, bins=max(int(np.sqrt(len(entries))),100), range=binRange)
        else:
            hist, bins = np.histogram(entries, bins=binN, range=binRange)
        v = bins[np.where(hist == hist.max())[0][0]] + (bins[1]-bins[0]) / 2
        e = (bins[1]-bins[0]) / 2
        return uf(v,e)
    
    def getBinMedianBootstrap(self, col, row, Bootstrap_N=1000, Bootstrap_draw=False):
        entries = self.data[col,row]
        medians = np.zeros(Bootstrap_N)
        for i in range(Bootstrap_N):
            a_boot = np.random.choice(entries, len(entries), replace=True)
            medians[i] = np.median(a_boot)
        if Bootstrap_draw:
            fig, ax = plt.subplots()
            hist, bins = np.histogram(medians, bins=20, range=[min(medians),max(medians)])
            _ = ax.stairs(hist, bins, fill=False, color="black")
            _ = ax.stairs(hist, bins, fill=True, color="black", alpha=0.2)
            ax.axvline(np.mean(medians), color="red")
            ax.axvline(np.mean(medians)+np.std(medians), color="red", linestyle="--")
            ax.axvline(np.mean(medians)-np.std(medians), color="red", linestyle="--")
            ax.set_title("Median Bootstrap (N="+str(Bootstrap_N)+")")
            ax.set_xlabel("Median", loc="right")
            ax.set_ylabel("Counts", loc="top")
        return uf(np.mean(medians), np.std(medians))
    
