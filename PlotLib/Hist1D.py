import numpy as np
from numpy import array as arr
from matplotlib import pyplot as plt
# import uncertainties as unc
# from uncertainties import unumpy as unp
from uncertainties import ufloat as uf
from uncertainties.unumpy import uarray as uarr
import ROOT

def curve_fit_wrapper(hist, bins, mask, fitfunc, sigma=None, **kwargs):
    from scipy.optimize import curve_fit
    
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

def normalize(hist, bins):
    """
    Normalize a histogram to the integral of the PDF. (ie. the area under it is 1)

    Parameters
    - hist: histogram to normalize
    - bins: bin edges of the histogram

    Returns:
    - The normalized histogram.
    """
    return arr(hist) / np.sum(hist) / np.diff(bins)

def truncate(hist, bins, interval=0.95):
    """
    Truncate a 1D histogram to a certain interval. Cuts off the tails of the histogram., does not change the bin content of bins on the edge.
    
    Parameters
    - hist: values of the histogram
    - bins: bin edges (len(bins) = len(hist)+1)
    
    Returns
    - hist: truncated values
    """
    hist = hist / np.sum(hist)
    bin_centers = (bins[1:]+bins[:-1])/2
    Mean = np.sum(hist_norm*bin_centers)
    Stop = (1-interval)/2

    S = 0
    for i in range(len(hist)):
        S += hist_norm[i]
        if S > Stop:
            i_low = i
            break
    S = 1
    for i in range(len(hist)-1, -1, -1):
        S -= hist_norm[i]
        if S < 1-Stop:
            i_up = i
            break
    
    # need to include bin i_up, thus return up to i_up+1
    return hist[i_low:i_up+1], bins[i_low:i_up+2]

def rebin(hist, bins, factor):
    """
    Combine N bins of a 1D histogram. Rightmost bins are lost if the number of bins is not divisible by factor.
    
    Parameters
    - hist: values of the histogram
    - bins: bin edges (len(bins) = len(hist)+1)
    - factor: how many bins to combine to a single new bin
    
    Returns
    - new_hist: new values of the histogram
    - new_bins: new bin edges (len(new_bins) = len(new_hist)+1)
    """
    # factor = how many bins to combine
    
    if len(hist) % factor != 0:
        print("WARNING: Number of bins not divisible by factor, losing " + str(len(hist) % factor) + " bin(s)")
    binN = len(hist)
    new_binN = binN//factor
    new_bins = np.zeros(new_binN+1)
    new_hist = np.zeros(new_binN)
    
    new_bins = bins[::factor]
    for i in range(new_binN):
        new_hist[i] = np.sum(hist[i*factor:(i+1)*factor])
    return new_hist, new_bins

class fitter():
    def __init__(self, hist, bins, func, range_fit=None, NofPointsFFT=1000, fit_options=None):
        """
        
        Parameters:
        hist : np.array - 1D histogram data
        bins : np.array - bin edges
        func : str - one of ["LanGau"]
        range_fit : 2-length list - range in which to fit the function
        fit_options : str - ROOT fit method string
            L - Log likelihood method (if hist represents counts) (default is chi-square method)
            WL - Log likelihood method (if hist represents counts and is filled with weights)
            
            Q - Quiet mode (minimum printing)
            E - better error estimates using the Minos technique
        """
        # Setup fitting range
        if range_fit is None:
            self.range_fit = [bins[0]-1*(bins[-1]-bins[0]), bins[-1]+1*(bins[-1]-bins[0])] # extend range by 10% on both sides to make function behave nicely at the edges
        elif len(range_fit) != 2:
            raise ValueError("fitLanGau(): range_fit must be a list of length 2")
        else:
            self.range_fit = range_fit
            
        self.func_str = func
        if "S" not in fit_options:
            fit_options += "S"
        if "N" not in fit_options:
            fit_options += "N"
            print("[WARN] hist1D_fitter:  Adding \"N\" to fit_options to make drawing work as intended.")
        if "R" not in fit_options:
            fit_options += "R"
            print("[WARN] hist1D_fitter:  Adding \"R\" to fit_options to only fit in the given function range (Convolution does not work outside this range).")
        self.fit_options = fit_options
        
        # Setup root histogram
        self.TH1D = ROOT.TH1F("rHist","rHist",len(hist),bins[0],bins[-1])
        for i in range(len(hist)):
            self.TH1D.SetBinContent(i+1,hist[i])
        
        # Setup function
        if self.func_str == "LanGau":
            # Setup Landau and Gauss functions
            fLandau = ROOT.TF1("landau", "[0]*TMath::Landau(x, [1], [2])", self.range_fit[0], self.range_fit[1]) 
                # multiplying by factor [0] is possible because
                # Integral([0]*f1(x-y)*f2(y)dy) = A*Integral(f1(x-y)*f2(y)).
            fGauss = ROOT.TF1("gauss", "TMath::Gaus(x, [0], [1], true)", self.range_fit[0], self.range_fit[1])
            
            # Convolute
            rConv = ROOT.TF1Convolution(fLandau, fGauss)
            rConv.SetNofPointsFFT(NofPointsFFT)
            
            # Setup fit function
            self.TF1 = ROOT.TF1("FitFunc", rConv, self.range_fit[0], self.range_fit[1], rConv.GetNpar())
            
            # Estimate initial parameters, specific to a LanGau fit
            h_integral = self.TH1D.Integral(0, self.TH1D.GetNbinsX(), "width")
            h_width = self.range_fit[1] - self.range_fit[0]
        
            self.TF1.SetParName(0,"Scale")
            self.TF1.SetParameter(0, h_integral)
            self.TF1.SetParLimits(0, 0, 100000*h_integral)

            self.TF1.SetParName(1,"MPV")
            self.TF1.SetParameter(1,
                self.TH1D.GetBinCenter(self.TH1D.GetMaximumBin()))
            self.TF1.SetParLimits(1,
                self.TH1D.GetBinLowEdge(1),
                self.TH1D.GetBinLowEdge(self.TH1D.GetNbinsX()) + self.TH1D.GetBinWidth(self.TH1D.GetNbinsX()))

            self.TF1.SetParName(2,"Width")
            self.TF1.SetParameter(2, 0.03*h_width)
            self.TF1.SetParLimits(2,0.005*h_width,0.4*h_width)

            self.TF1.FixParameter(3,0)

            self.TF1.SetParName(4,"Sigma")
            self.TF1.SetParameter(4, 0.03*h_width)
            self.TF1.SetParLimits(4, 0.005*h_width, 0.4*h_width)
        else:
            raise ValueError("hist1D_fitter(): unknown function string", self.func_str)
        
    def setPar(self, parID, value):
        self.TF1.SetParameter(parID, value)
    
    def setParLimits(self, parID, low, high):
        self.TF1.SetParLimits(parID, low, high)
        
    def fixPar(self, parID, value):
        self.TF1.FixParameter(parID, value)
        
    def fit(self, fit_options=None):
        if fit_options is not None:
            print("[WARN] hist1D_fitter:  Fit options are:", self.fit_options, ". Overwriting with ", fit_options)
            self.fitRes = self.TH1D.Fit(self.TF1, fit_options)
        self.fitRes = self.TH1D.Fit(self.TF1, self.fit_options)
        if self.fitRes.Status() != 0:
            print("[WARN] hist1D_fitter: Fit did not succeed. Output:")
            self.fitRes.Print()
        
    def func(self, x):
        eval = np.vectorize(self.TF1.Eval)
        return eval(x)
    
    def integralFunc(self):
        x0 = self.TH1D.GetBinLowEdge(0)
        x1 = self.TH1D.GetBinLowEdge(self.TH1D.GetNbinsX()) + self.TH1D.GetBinWidth(self.TH1D.GetNbinsX())
        return self.TF1.Integral(x0, x1)
    
    def integralHist(self, option="width"):
        return self.TH1D.Integral(0, self.TH1D.GetNbinsX(), option)

    def getPar(self, parID):
        return self.TF1.GetParameter(parID), self.TF1.GetParError(parID)
    
    def getParUf(self, parID):
        return uf(self.TF1.GetParameter(parID), self.TF1.GetParError(parID))
    
    def getPars(self):
        return arr([self.TF1.GetParameter(i) for i in range(self.TF1.GetNpar())])
    
    def getParsUarr(self):
        return uarr(
            arr([self.TF1.GetParameter(i) for i in range(self.TF1.GetNpar())]), 
            arr([self.TF1.GetParError(i) for i in range(self.TF1.GetNpar())]))
    
    def getMaxX(self):
        return self.TF1.GetMaximumX(self.range_fit[0], self.range_fit[1])
    def getMax(self):
        return self.TF1.GetMaximum(self.range_fit[0], self.range_fit[1])
    
    def getStatus(self):
        return self.fitRes.Status()
    
    def draw(self, filename):
        self.TF1.SetNpx(1000)
        self.TH1D.GetListOfFunctions().Add(self.TF1)
        rCanvas = ROOT.TCanvas("rCanvas","rCanvas",800,600)
        self.TH1D.Draw()
        rCanvas.SaveAs(filename)
        
    def print(self):
        self.fitRes.Print()