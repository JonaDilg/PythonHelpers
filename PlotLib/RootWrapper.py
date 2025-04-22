import numpy as np
from numpy import array as arr

import ROOT
import uncertainties as unc
from uncertainties import unumpy as unp
from uncertainties.unumpy import uarray as uarr
from uncertainties.unumpy import nominal_values as val
from uncertainties.unumpy import std_devs as dev
from uncertainties import ufloat as uf

import PlotLib.Hist2D as Hist2D

def getObject(FilePath, DirName, ObjName):
    """
    Get a ROOT object from a file.
    
    Parameters
    ----------
    FilePath : str
        Path to the ROOT file.
    DirName : str
        Name of the directory in the ROOT file.
    ObjName : str
        Name of the object to retrieve.
    
    Returns
    -------
    obj : ROOT.TObject
        The requested ROOT object.
    """
    
    rFile = ROOT.TFile(FilePath)
    if not rFile:
        raise KeyError("File not found at " + FilePath)
        
    rDir = rFile.Get(DirName)
    if not rDir:
        rFile.Close()
        raise KeyError("Directory" , DirName, "not found in .root file", FilePath)
    
    rObj = rDir.Get(ObjName)
    if not rObj:
        rFile.Close()
        raise KeyError("Object" , ObjName, "not found in directory", DirName, "in .root file", FilePath)
    
    # change ownership of the object to the current process (so it doesn't get deleted when the file is closed)
    rObjClone = rObj.Clone()
    rObjClone.SetDirectory(0)
    
    rFile.Close()
    return rObjClone

def importTH1(rTH1, rebin=None, scaleX=1, scaleY=1):
    if not rTH1:
        raise ValueError("TH1 is None")
      
    if rebin is not None:
        rTH1.Rebin(rebin)
    
    binN = rTH1.GetNbinsX()
    bins = np.zeros(binN+1)
    hist = uarr(np.zeros(binN),np.zeros(binN))
    
    for i in range(binN):
        hist[i] = uf(rTH1.GetBinContent(i+1), rTH1.GetBinError(i+1))
        bins[i] = rTH1.GetBinLowEdge(i+1)
    bins[-1] = rTH1.GetBinLowEdge(i+1) + rTH1.GetBinWidth(i+1)
    
    bins = bins * scaleX # convert to h
    hist = hist * scaleY # convert to um
    
    print(f"import.TH1(): final binWidth = {rTH1.GetBinWidth(1) * scaleX:.2e}")
    
    return hist, bins

def importTProfile_asXY(rTProf, rebin=None, scaleX=None, scaleY=None):
    if not rTProf:
        raise ValueError("TProfile is None")
    
    if rebin is not None:
        rTProf.Rebin(rebin)
        print(f"BinWidth after rebinning = {rTProf.GetBinWidth(1)}")
        
    N = rTProf.GetNbinsX()
    x = np.zeros(N)
    y = uarr(np.zeros(N), np.zeros(N))
    
    for i in range(N):
        x[i] = rTProf.GetBinCenter(i+1)
        y[i] = uf(rTProf.GetBinContent(i+1), rTProf.GetBinError(i+1))
        
    x = x[y!=0]
    y = y[y!=0]
    
    if scaleX is not None:
        x = x * scaleX
    if scaleY is not None:
        y = y * scaleY
    
    return x,y
        
def importTProfile2D_asHist(TProf, scaleX=None, scaleY=None):
    if not TProf:
        raise ValueError("TProfile2D is None")
    
    binN = [ TProf.GetNbinsX(), TProf.GetNbinsY() ]
    binRange = [[TProf.GetXaxis().GetBinLowEdge(1), TProf.GetXaxis().GetBinUpEdge(binN[0])], [TProf.GetYaxis().GetBinLowEdge(1), TProf.GetYaxis().GetBinUpEdge(binN[1])]]
    hist = Hist2D.Plot_2D(binN, binRange)
    
    for col in range(binN[0]+2):
        for row in range(binN[1]+2):
            binC = TProf.GetBinContent(col, row)
            binE = TProf.GetBinError(col, row)
            if binC == 0.:
                continue
            hist.val[col, row] = binC
            hist.unc[col, row] = binE
            
    if scaleX is not None:
        hist.binsX = hist.binsX * scaleX
    if scaleY is not None:
        hist.binsY = hist.binsY * scaleY
    
    return hist