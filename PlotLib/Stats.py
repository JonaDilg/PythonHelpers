import numpy as np
import ROOT
from numdifftools import Hessian
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import erf
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from uncertainties import ufloat as uf


# -- Helpers --

def getMean(entries, N=1000, truncateInterval=None):
    """
    Get the mean of a 1-d array of measurements (ie. drawn from the distribution)
    """
    if truncateInterval is not None:
        entries = truncateEntries(entries.copy(), truncateInterval)
    Mean = np.mean(entries)
    Unc = getUnc_Bootstrap(entries, np.mean, N=N)
    return uf(Mean, Unc)

def getStdDev(entries, N=1000, truncateInterval=None):
    """
    Get the standard deviation of a 1-d array of measurements (ie. drawn from the distribution)
    """
    if truncateInterval is not None:
        entries = truncateEntries(entries.copy(), truncateInterval)
    Std = np.std(entries)
    Unc = getUnc_Bootstrap(entries, np.std, N=1000)
    return uf(Std, Unc)

def getMedian(entries, N=1000):
    """
    Get the median of a 1-d array of measurements (ie. drawn from the distribution)
    """
    Median = np.median(entries)
    Unc = getUnc_Bootstrap(entries, np.median, N=N)
    return uf(Median, Unc)

def getUnc_Bootstrap(entries, func, N=1000, *args):
    """
    Bootstrap resampling method to estimate the uncertainty of a function.

    Parameters
    - entries: 1-d array of measurements (ie. drawn from the distribution)
    - func: function to evaluate on the resampled data
    - n: number of bootstrap samples
    - args: additional arguments to pass to the function

    Returns: mean, std
    - mean: mean of the function evaluated on the bootstrap samples
    - std: standard deviation of the function evaluated on the bootstrap samples
    """
    vals = np.zeros(N)
    for i in range(N):
        vals[i] = func(np.random.choice(entries, size=len(entries), replace=True), *args)
    return np.std(vals)

def truncateEntries(entries, interval):
    """
    Truncate entries to a given interval, retain the central set.

    Parameters
    - entries: 1-d array of measurements (ie. drawn from the distribution)
    - interval: fraction of entries to keep (0.0 < interval < 1.0)

    Returns: truncated entries
    """
    if interval is None or interval <= 0.0 or interval >= 1.0:
        raise ValueError("Truncate interval must be between 0.0 and 1.0")
    # entries = entries.sort()
    entries.sort()
    N = len(entries)
    return entries[int(N*(1-interval)/2):int(N*(1+interval)/2)]

def truncateEntriesToLower(entries, keep):
    """
    Truncate entries to a given interval, retain the lower set.

    Parameters
    - entries: 1-d array of measurements (ie. drawn from the distribution)
    - interval: fraction of entries to keep (0.0 < interval < 1.0)

    Returns: truncated entries
    """
    if keep is None or keep <= 0.0 or keep >= 1.0:
        raise ValueError("Truncate interval must be between 0.0 and 1.0")
    # entries = entries.sort()
    entries.sort()
    N = len(entries)
    return entries[:int(N*keep)]

def truncateEntriesToUpper(entries, keep):
    """
    Truncate entries to a given interval, retain the upper set.

    Parameters
    - entries: 1-d array of measurements (ie. drawn from the distribution)
    - interval: fraction of entries to keep (0.0 < interval < 1.0)

    Returns: truncated entries
    """
    if keep is None or keep <= 0.0 or keep >= 1.0:
        raise ValueError("Truncate interval must be between 0.0 and 1.0")
    entries.sort()
    N = len(entries)
    return entries[int(N*keep):]

# -- Fitting --

def LikelihoodFit(data, PDF, par0, bounds=None, maxiter=10000, est_unc=True):
    """
    Fit a function to data using maximum likelihood estimation.

    Parameters
    - data: 1-d array of measurements (ie. drawn from the distribution)
    - PDF: probability density function to fit to data
    - p0: initial guess for parameters
    - bounds (optional): bounds for parameters [ (min1, max1), (min2, max2), ... ]. Replace min/max with None for no bound.
    - maxiter (optional): maximum number of iterations for minimization
    - est_unc (optional): estimate parameter uncertainties using numerical Hessian matrix estimation. returns unc=[0,0,...] if False.

    Returns: par, unc
    - par: best-fit parameters
    - unc: estimated uncertainties on parameters
    """
    # negative log likelihood
    def NLL(params, data):
        return -np.sum(np.log(PDF(data, params)))

    res = minimize(NLL, par0, args=(data), bounds=bounds, options={'maxiter': maxiter},)
    if not res.success:
        print("Stats.LikelihoodFit(): fit failed")
        print(res.message)
    elif est_unc:
        Hesse_func = Hessian(lambda params: NLL(params, data))
        Hesse = Hesse_func(res.x)
        unc = np.sqrt(np.diag( np.linalg.inv(Hesse)))
        return res.x, unc
    return res.x, np.zeros_like(res.x)

def curve_fit_wrapper(x, y, fitfunc, mask=None, sigma=None, p0=None, bounds=[-np.inf,np.inf], return_pcov=False, **kwargs):
    
    # fit function needs to be defined as f(x, *p)
    # p0 is the initial guess for the fit parameters
    # mask is the mask to apply to the data
    
    if len(x) == 0:
        raise ValueError("curve_fit_wrapper(): x")
    if len(x) != len(y):
        raise ValueError("curve_fit_wrapper(): x and y must have the same length")
    
    if sigma is None:
        sigma = np.sqrt(abs(x))
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    
    x = x[mask]
    y = y[mask]
    sigma = sigma[mask]
    
    popt, pcov = curve_fit(fitfunc, x, y, sigma=sigma, p0=p0, bounds=bounds, check_finite=True, absolute_sigma=True, **kwargs)
    
    perr = np.sqrt(np.diag(pcov))
    dx = fitfunc(x, *popt) - y
    chi2 = np.sum(dx**2 / sigma**2)
    
    ndeg = len(y) - len(popt)
        
    if return_pcov:
        return popt, perr, pcov, chi2, ndeg
    return popt, perr, chi2, ndeg


# -- PDFs --

def pdfStep(x, par):
    """
    PDF of a step function.

    Parameters
    - x: point at which to evaluate the PDF
    - par: [x0, height] of the step function
    """
    x0, x1 = par
    height = 1 / (x1 - x0) # integral = 1 
    if x < x0 or x > x1:
        return 0
    return height
pdfStep = np.vectorize(pdfStep, excluded=[1])

def pdfStepGauss(x, par):
    """
    PDF of a step function convoluted with a Gaussian.

    Parameters
    - x: point at which to evaluate the PDF
    - par: [x0, x1, sigma] pos. of the step function and Gaussian
    """
    x0, x1, sigma = par
    # height = 1 / (x1 - x0) # integral = 1
    
    term1 = erf((x - x0) / (np.sqrt(2) * sigma))
    term2 = erf((x - x1) / (np.sqrt(2) * sigma))
    return 0.5 * (term1 - term2)
    
    term1 = erf((x1) / (np.sqrt(2) * sigma))
    term2 = erf((x0) / (np.sqrt(2) * sigma))
    return 0.5 / (x1-x0) * (term1 - term2)    
pdfStepGauss = np.vectorize(pdfStepGauss, excluded=[1])

def pdfNormal(x, sigma):
    """
    PDF of a normal distribution. (Gaussian centered at 0)

    Parameters
    - x: point at which to evaluate the PDF
    - sigma: standard deviation of the normal distribution
    """
    func = np.vectorize(ROOT.TMath.Gaus, excluded=[1,2,3])
    return func(x, 0, sigma, True)

def pdfGauss(x, par):
    """
    PDF of a Gaussian.

    Parameters
    - x: point at which to evaluate the PDF
    - par: [mean, sigma] of the Gaussian
    """
    mu, sigma = par
    func = np.vectorize(ROOT.TMath.Gaus, excluded=[1,2,3])
    return func(x, mu, sigma, True)

def pdfGumbel(x, par):
    """
    PDF of a Gumbel distribution.
            fgumbel = pdfGumbel(x, mu, beta)
    """
    z = (x - par[0]) / par[1]
    return np.exp(-z - np.exp(-z)) / par[0]

def pdfLandau(x, par):
    """
    PDF of a Landau distribution.
            fland = pdfLandau(xx, mu, eta) / eta
    Note: ROOT implementation is ever so slightly slower than the numeric one, but does not matter. I assume it's more accurate.
    """
    mpv, eta = par
    func = np.vectorize(ROOT.TMath.Landau, excluded=[1,2,3])
    return func(x, mpv, eta, True)

def is_number_(val):
    return isinstance(val, (float, np.floating, int, np.integer))


p1 = [0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253]
q1 = [1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063]
p2 = [0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211]
q2 = [1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714]
p3 = [0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101]
q3 = [1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675]
p4 = [0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186]
q4 = [1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511]
p5 = [1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910]
q5 = [1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357]
p6 = [1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109]
q6 = [1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939]
a1 = [0.04166666667, -0.01996527778, 0.02709538966]
a2 = [-1.845568670, -4.284640743]

def pdfLandau_manual(xs, par): # same algorithm is used in GSL
    x0, xi = par
    
    if (xi <= 0):
        return 0.
    vs = (xs - x0) / xi
    # double u, ue, us, denlan;

    if is_number_(vs):
        vs = np.array([vs])
    res = np.zeros_like(vs)	
 
    for i,v in enumerate(vs):
        if (v < -5.5):
            u = np.exp(v + 1.0)
            if (u < 1e-10):
                return 0.0
            ue = np.exp(-1 / u)
            us = np.sqrt(u)
            denlan = 0.3989422803 * (ue / us) * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u)
        elif (v < -1):
            u = np.exp(-v - 1)
            denlan = np.exp(-u) * np.sqrt(u) * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) / (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v)
        elif (v < 1):
            denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * v) / (q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * v) * v) * v) * v)
        elif (v < 5):
            denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * v) / (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * v) * v) * v) * v)
        elif (v < 12):
            u = 1 / v
            denlan = u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) / (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u)
        elif (v < 50):
            u = 1 / v
            denlan = u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) / (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u)
        elif (v < 300):
            u = 1 / v
            denlan = u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) / (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u)
        else:
            u = 1. / (v - v * np.log(v) / (v + 1))
            denlan = u * u * (1 + (a2[0] + a2[1] * u) * u)
        res[i] = denlan / xi

    if len(res) == 1:
        return res[0]
    return res

def _LanGau_ConvolutionIntegral(x_i, sc, sigma, mu, eta, nConvSteps):
    # Range of convolution integral
    xlow = x_i - sc * sigma
    xupp = x_i + sc * sigma

    step = (xupp - xlow) / nConvSteps

    # Discrete linear convolution of Landau and Gaussian
    sum = 0
    for i in range(1,int(nConvSteps/2)+1):
        xx = xlow + (i - 0.5) * step
        sum += pdfLandau(xx, [mu, eta]) / eta * pdfGauss(x_i, [xx, sigma])

        xx = xupp - (i - 0.5) * step
        sum += pdfLandau(xx, [mu, eta]) / eta * pdfGauss(x_i, [xx, sigma])
    return (step * sum) / np.sqrt(2 * np.pi) / sigma

def pdfLanGau(x, par):
    """
    PDF of convolution of Landau and Gaussian functions.
    (taken from https://github.com/SiLab-Bonn/pylandau/ on 2025-01-24. This implementation is a bit faster than doing the convolution via stupid numerical integration using scipy.integrate.quad)

    Parameters:
    - x: Point at which to evaluate the convolved PDF.
    - par: [mpv, eta, sigma] of the Landau and Gaussian functions.
        - mpv: Most Probable value of Landau.
        - eta: Width (scale) parameter of Landau.
        - sigma: Width of Gaussian.
    """
    mu, eta, sigma = par

    # Control constants
    nConvSteps = 100;       # number of convolution steps. Dont use to many.
    sc = 5;                 # convolution extends to +-sc Gaussian sigmas

    # Convolution steps have to be increased if sigma > eta * 5 to get stable solution that does not oscillate, addresses #1
    if (sigma > 5 * eta):
        nConvSteps *= int(sigma / eta / 5.)

    if is_number_(x):
        x = np.array([x])

    # new implementation using multiprocessing
    # ConvolutionIntegral_partial = partial(_LanGau_ConvolutionIntegral, sc=sc, sigma=sigma, mu=mu, eta=eta, nConvSteps=nConvSteps)
    # with ProcessPoolExecutor() as executor:
    #     x_res = list(executor.map(ConvolutionIntegral_partial, x))
    # if len(x_res) == 1:
    #     return x_res[0]
    # return np.array(x_res)
    
    # old implementation without multiprocessing. For some reason this is MUCH (~ 10x) faster than the above.
    for i_x in range(len(x)):
        # Range of convolution integral
        xlow = x[i_x] - sc * sigma
        xupp = x[i_x] + sc * sigma

        step = (xupp - xlow) / nConvSteps

        # Discrete linear convolution of Landau and Gaussian
        sum = 0
        for i in range(1,int(nConvSteps/2)+1):
            xx = xlow + (i - 0.5) * step
            sum += pdfLandau(xx, [mu, eta]) / eta * pdfGauss(x[i_x], [xx, sigma])

            xx = xupp - (i - 0.5) * step
            sum += pdfLandau(xx, [mu, eta]) / eta * pdfGauss(x[i_x], [xx, sigma])
        x[i_x] = (step * sum) / np.sqrt(2 * np.pi) / sigma
    if len(x) == 1:
        return x[0]
    return x

def convoluteGauss(x, sigma, par, pdf):
    """
    Convolute a function with a Gaussian.

    Parameters
    - x: point at which to evaluate the convolution
    - sigma: standard deviation of the Gaussian
    - par: tuple of parameters for the pdf
    - pdf: function to convolute with the Gaussian, pdf(x, par)
    """
    # nConvSteps = 1000
    # nSigmaRange = 8
    
    # x_kernel = np.linspace(x - nSigmaRange * sigma, x + nSigmaRange * sigma, nConvSteps)
    
    # # Evaluate original PDF and Gaussian kernel
    # y_pdf = pdf(x_kernel, par)
    # y_gauss = pdfNormal(x - x_kernel, sigma)
    
    # # Compute the convolution via numerical integration
    # return np.trapezoid(y_pdf * y_gauss, x_kernel)
    
        # Gaussian kernel
    def gaussian_kernel(xp):
        return norm.pdf(x - xp, scale=sigma)
    
    # Integrand: product of the function and the Gaussian
    def integrand(xp):
        return pdf(xp, par) * gaussian_kernel(xp)
    
    # Perform convolution integral over a wide enough range
    result, _ = quad(integrand, x - 5*sigma, x + 5*sigma, limit=100)
    # result, _ = quad(integrand, x - 10*sigma, x + 10*sigma, epsabs=1e-10, epsrel=1e-10, limit=500)

    return result