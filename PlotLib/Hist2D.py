import numpy as np
from numpy import array as arr
import uncertainties as unc
from uncertainties import unumpy as unp
from uncertainties.unumpy import uarray as uarr
from uncertainties.unumpy import nominal_values as val
from uncertainties.unumpy import std_devs as dev
from uncertainties import ufloat as uf
import matplotlib.pyplot as plt

# from PlotLib.Stats import pdfLanGau
import PlotLib.Hist1D as Hist1D

def is_number_(val):
    return isinstance(val, (float, np.floating, int, np.integer))

def poisson_stat_sq_(val):
    if abs(val) < 1: 
        return 1.14**2
    return abs(val)

class Hist_2D:
    def __init__(self, bins, binRange=None, poisson_stat=True):
        _, self.binsX, self.binsY = np.histogram2d([-10,10], [-10,10], bins=bins, range=binRange)
        self.val = np.zeros((len(self.binsX)+1,len(self.binsY)+1))
        self.unc = np.zeros((len(self.binsX)+1,len(self.binsY)+1))
        self.poisson_stat = poisson_stat
    
    def getBinRangeX(self):
        return arr([self.binsX[0],self.binsX[-1]])
    def getBinRangeY(self):
        return arr([self.binsY[0],self.binsY[-1]])
    
    def getBinNX(self):
        return len(self.binsX)-1
    def getBinNY(self):
        return len(self.binsY)-1
    def getBinN(self):
        return self.getBinNX()*self.getBinNY()
            
    def getBinWidthX(self, binX=1):
        return self.binsX[binX+1]-self.binsX[binX]
    def getBinWidthY(self, binY=1):
        return self.binsY[binY+1]-self.binsY[binY]
            
    def getBinsX(self):
        return self.binsX
    def getBinsY(self):
        return self.binsY
            
    def getBinIndex(self, x, y):
        return self.getBinIndexX(x), self.getBinIndexY(y)
    def getBinIndexX(self, x):
        try:
            return arr(np.where(np.logical_and(x>=self.binsX[:-1], x<self.binsX[1:]))).flatten()[0]+1
        except:
            return (0 if x<self.binsX[0] else len(self.binsX))
    def getBinIndexY(self, y):
        try:
            return arr(np.where(np.logical_and(y>=self.binsY[:-1], y<self.binsY[1:]))).flatten()[0]+1
        except:
            return (0 if y<self.binsY[0] else len(self.binsY))
        
    def getIntegral(self, overflow=False):
        if overflow:
            return np.sum(self.val)
        return np.sum(self.val[1:-1,1:-1])
        
    def getBinContent(self, binX, binY):
        return self.val[binX, binY]
    
    def setBinContent(self, binX, binY, value, unc=None): 
        self.val[binX,binY] = value
        if self.poisson_stat:
            if unc!=None:
                raise ValueError("[ERROR] Hist_2D.setBinContent(): Poisson statistics are enabled. The uncertainty is calculated automatically.")
            self.unc[binX, binY] = poisson_stat_sq_(value)
        else:
            if unc==None:
                raise ValueError("[ERROR] Hist_2D.setBinContent(): Poisson statistics are disabled. Please provide an uncertainty.")
            self.unc[binX, binY] = unc
            
    def addBinContent(self, binX, binY, w=1):
        self.val[binX, binY] += w
        if not self.poisson_stat:
            raise ValueError("[ERROR] Hist_2D.addBinContent(): Poisson statistics are disabled. Use setBinContent() with a custom uncertainty instead.")
        self.unc[binX, binY] = poisson_stat_sq_(self.val[binX, binY])
    def fill(self, x, y, w=1):
        if not self.poisson_stat:
            raise SystemError("[ERROR] Hist_2D.fill(): Poisson statistics are disabled. Use setBinContent() with a custom uncertainty instead.")
        if is_number_(x) and is_number_(y):
            binX, binY = self.getBinIndex(x, y)
            self.addBinContent(binX, binY, w)
            return
        if len(x)!=len(y):
            raise ValueError("[ERROR]Hist_2D.fill(): x and y need to have the same length. The lengths are:", len(x), len(y))
        for i in range(len(x)):
            binX, binY = self.getBinIndex(x[i], y[i])
            # binX = self.getBinIndexX(x[i])
            # binY = self.getBinIndexY(y[i])
            self.addBinContent(binX, binY, w)
        
    def draw(self, ax, ax_cbar=None, extent=None, cmap="viridis", cbar_label=None, overflow=False, aspect=None, **kwargs):
        """
        
        kwargs for ax.imshow():
        - vmin:float, vmax:float
            min and max values for the color scale
        """
        if not overflow:
            X = self.val[1:-1,1:-1]
            im = ax.imshow(self.val[1:-1,1:-1].T, extent=[self.binsX[0], self.binsX[-1], self.binsY[0], self.binsY[-1]], origin='lower', **kwargs)
            # im = ax.imshow(self.val.T, extent=[self.binsX[0],self.binsX[-1],self.binsY[0],self.binsY[-1]], origin='lower', **kwargs)
        else: 
            binWidthX, binWidthY = self.getBinWidthX(0), self.getBinWidthY(0)
            im = ax.imshow(self.val.T, extent=[self.binsX[0]-binWidthX, self.binsX[-1]+binWidthX, self.binsY[0]-binWidthY, self.binsY[-1]+binWidthY], origin='lower', **kwargs)
            ax.plot([self.binsX[0],self.binsX[-1],self.binsX[-1],self.binsX[0],self.binsX[0]], [self.binsY[0],self.binsY[0],self.binsY[-1],self.binsY[-1],self.binsY[0]], color='black', linestyle=":")
        ax.set_aspect(aspect, adjustable="box", anchor="NW")
        if ax_cbar is None:
            y0 = ax.get_position().y0
            pad = 0.01
            width = 0.04
            ax_cbar = ax.get_figure().add_axes([1.01, y0, width, 1-y0])   
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label(cbar_label, loc="top", labelpad=0.5)
        ax_cbar.tick_params(axis="y", direction="in", )

        
        return im, ax_cbar

    def printOverflow(self):
        print("Overflow:")
        
        print(uf(self.val[0,-1], self.unc[0,-1]), "|", uarr(self.val[1:-1,0], self.unc[1:-1,0]).sum(), "|", uf(self.val[-1,-1], self.unc[-1,-1]))
        print(uarr(self.val[0,1:-1], self.unc[0,1:-1]).sum(), "|", uarr(self.val[1:-1,1:-1], self.unc[1:-1,1:-1]).sum(), "|", uarr(self.val[-1,1:-1], self.unc[-1,1:-1]).sum())

class Plot_2D(Hist_2D):
    def __init__(self, bins, binRange=None, statistic="Mean", truncate=None,
        inBin_binNdelta=None, inBin_binRange=None,
        Bootstrap_N=None, Bootstrap_draw=False):
        """
        
        Parameters:
        - bins, binRange: same as numpy histogram
        - statistic: str - one of ["Mean", "MedianBootstrap", "LanGau-MPV"]
        - inBin_binNdelta: int - for fitting a LanGau to the content of each bin individually, all the bin entries need to be binned into a histogram. This parameter adds N bins to the number of bins in this in-bin histogram, which has sqrt(N_entries) bins by default. Can be negative.
        - inBin_binRange: 2-length list - range of the in-bin histogram
        """
        super().__init__(bins, binRange, poisson_stat=False)
        allowed_statistics = ["Mean", "MPV", "MedianBootstrap", "LanGau-MPV"]
        if statistic not in allowed_statistics:
            raise ValueError("Plot_2D.__init__(): statistic needs to be one of:", allowed_statistics)
        self.statistic = statistic
        if self.statistic == "MedianBootstrap":
            if Bootstrap_N is None:
                print("[WARNING] Plot_2D.__init__(): statistic=\"MedianBootstrap\", but Bootstrap_N is undefined. Using 1000.")
                Bootstrap_N = 1000
            self.Bootstrap_N = Bootstrap_N # number of bootstrap samples
            self.Bootstrap_draw = Bootstrap_draw # draw the bootstrap distribution for each bin
        elif self.statistic == "LanGau-MPV":
            self.inBin_binNdelta = inBin_binNdelta
            self.inBin_binRange = inBin_binRange
            
        self.truncate = truncate
            
        self.data = np.empty([len(self.binsX)+1,len(self.binsY)+1], dtype=object)
        for binX in range(len(self.binsX)+1):
            for binY in range(len(self.binsY)+1):
                self.data[binX, binY] = arr([])
                
        self.nFits = 0
                
    def getBinData(self, binX, binY):
        return self.data[binX, binY]
    
    def getBinSampleSize(self, binX, binY):
        return len(self.data[binX, binY])
    def getSampleSize(self, overflow=False):
        if overflow:
            return np.sum(
                [[len(self.data[binX, binY]) for binX in range(len(self.binsX)+1)] for binY in range(len(self.binsY)+1)])
        return np.sum(
            [[len(self.data[binX, binY]) for binX in range(1,len(self.binsX))] for binY in range(1,len(self.binsY))])
    
    # load set of data points into the histogram
    def fillData(self, xs, ys, zs):
        # dataset of tuples of (x,y,z). x,y define the bin, z is averaged for each bin
        if not ((len(xs)==len(ys)) and (len(ys)==len(zs))):
            print("[ERROR] Plot_2D(): fill_entries needs x,y,z arrays of the same lengths. The lengths are:", len(xs), len(ys), len(zs))
        xs = [self.getBinIndexX(x) for x in xs]
        ys = [self.getBinIndexY(y) for y in ys]
        for i in range(len(xs)):
            self.data[xs[i],ys[i]] = np.append(self.data[xs[i],ys[i]], [zs[i]]) # appends each z to the list of entries in its corresponding bin
        self.analyseData()
        
    # use the data in each bin to fill the histogram
    def analyseData(self):
        for binX in range(len(self.binsX)+1):
            for binY in range(len(self.binsY)+1):
                if len(self.data[binX, binY]>0):
                    if self.statistic == "Mean":
                        temp = self.get_mean_(self.data[binX, binY])
                        val = temp.n
                        unc = temp.s
                    elif self.statistic == "MPV":
                        temp = self.get_MPV_(self.data[binX, binY])
                        val = temp.n
                        unc = temp.s
                    elif self.statistic == "MedianBootstrap":
                        if len(self.data[binX, binY]) < 100:
                            print("[WARN] Plot_2D.fill_hist_(): MedianBootstrap needs at least 100 entries per bin")
                        temp = self.get_median_bootstrap_(self.data[binX, binY])
                        val = temp.n
                        unc = temp.s
                    elif self.statistic == "LanGau-MPV":
                        temp = self.get_LanGau_MPV(self.data[binX, binY],binXY=[binX,binY])
                        val = temp.n
                        unc = temp.s
                    else:
                        raise SystemError("[ERROR] Plot_2D.fill_hist_(): statistic not implemented")
                    self.setBinContent(binX, binY, val, unc**2)
                else: 
                    self.setBinContent(binX, binY, 0.,0.)
    
    def get_mean_(self, entries):
        if self.truncate is not None:
            entries.sort()
            entries = entries[:int(len(entries)*self.truncate)].copy()
        n = np.mean(entries)
        s = np.std(entries)
        return uf(n,s)
    
    def get_MPV_(self, entries):
        if self.truncate is not None:
            entries.sort()
            entries = entries[:int(len(entries)*self.truncate)].copy()
        if (self.MPV_binN is None) and (self.MPV_binRange is None):
            hist, bins = np.histogram(entries, bins=max(int(np.sqrt(len(entries))),100))
        elif (self.MPV_binN is None) and (self.MPV_binRange is not None):
            hist, bins = np.histogram(entries, bins=max(int(np.sqrt(len(entries))),100), range=self.MPV_binRange)
        else:
            hist, bins = np.histogram(entries, bins=self.MPV_binN, range=self.MPV_binRange)
        n = bins[np.where(hist == hist.max())[0][0]] + (bins[1]-bins[0]) / 2
        s = (bins[1]-bins[0]) / 2
        return uf(n,s)

    def get_median_bootstrap_(self, entries):
        medians = np.zeros(self.Bootstrap_N)
        for i in range(self.Bootstrap_N):
            a_boot = np.random.choice(entries, len(entries), replace=True)
            medians[i] = np.median(a_boot)
        if self.Bootstrap_draw:
            fig, ax = plt.subplots()
            hist, bins = np.histogram(medians, bins=20, range=[min(medians),max(medians)])
            _ = ax.stairs(hist, bins, fill=False, color="black")
            _ = ax.stairs(hist, bins, fill=True, color="black", alpha=0.2)
            ax.axvline(np.mean(medians), color="red")
            ax.axvline(np.mean(medians)+np.std(medians), color="red", linestyle="--")
            ax.axvline(np.mean(medians)-np.std(medians), color="red", linestyle="--")
            ax.set_title("Median Bootstrap (N="+str(self.Bootstrap_N)+")")
            ax.set_xlabel("Median", loc="right")
            ax.set_ylabel("Counts", loc="top")
        return uf(np.mean(medians), np.std(medians))
    
    def get_LanGau_MPV(self, entries, binXY=None):
        binN = int(np.sqrt(len(entries)))
        if self.inBin_binNdelta is not None:
            binN += self.inBin_binNdelta
        hist, bins = np.histogram(entries, bins=binN, range=self.inBin_binRange)
        
        self.nFits += 1
        fitter = Hist1D.fitter(hist, bins, "LanGau", fit_options="QRN")
            # L - LogLikelihood fit (default: chi-square, ignore empty bins. Using chi-square for now because its more stable)
            # E - better error estimation
            # R - range of fit
            # N - do not plot
            # Q - quiet mode
        fitter.fit()
        fitter.draw(f"temp-{binXY[0]}-{binXY[1]}.pdf")
        fitter.print()
        if fitter.getStatus() == 0:
            res = uf(fitter.getMaxX(), 0.)
        elif fitter.getStatus() == 1:
            print(f"[WARN] Plot_2D.get_LanGau_MPV(): Fit {self.nFits} failed with status 1 (Not converged). Returning NaN. Used {len(entries)} data points.")
            fitter.draw(f"failed{self.nFits}.pdf")
            res =  uf(float("nan"), float("nan"))
        elif fitter.getStatus() == 2:
            print(f"[WARN] Plot_2D.get_LanGau_MPV(): Fit {self.nFits} failed with status 2 (Error in uncertainty estimation). Used {len(entries)} data points.")
            fitter.draw(f"failed{self.nFits}.pdf")
            res = uf(fitter.getMaxX(), 0.)
        elif fitter.getStatus() == 3:
            print(f"[WARN] Plot_2D.get_LanGau_MPV(): Fit {self.nFits} failed with status 3 (approximation issue). Returning NaN. Used {len(entries)} data points.")
            fitter.draw(f"failed{self.nFits}.pdf")
            res = uf(float("nan"), float("nan"))
        else:
            print(f"[WARN] Plot_2D.get_LanGau_MPV(): Fit {self.nFits} failed with status {fitter.getStatus():n}, returning NaN. Used {len(entries)} data points.")
            fitter.print()
            fitter.draw(f"failed-{binXY[0]}-{binXY[1]}.pdf")
            res = uf(float("nan"), float("nan"))
        
        del fitter
        return res