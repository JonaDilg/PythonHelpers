import numpy as np
import ROOT
from numdifftools import Hessian
from scipy.optimize import minimize
# from scipy.integrate import quad

# -- Helpers --



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

# -- PDFs --

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

