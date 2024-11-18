from numpy import array as arr

def load_settings(runID):
    return settings_dict[runID]

charge_calibration = {
    2.32: arr([10100.13164889, 10363.74574137, 10670.19824758, 10031.88867229]),
    3.3: arr([10331.78239925, 10637.14243425, 10998.04551077, 10440.91481794]),
    4.57: arr([10484.04549243, 11056.13741112, 11343.78887434, 10718.16068878]),
    5.53: arr([10683.1111317 , 11370.82939689, 11738.21428846, 11063.603456  ]),
    6.78: arr([11048.71202011, 11472.03945169, 12507.79790097, 10893.02044626]),
    7.73: arr([11188.15599342, 11722.52276237, 12830.12738093, 11246.9124095 ]),
    8.96: arr([11342.9575507 , 12104.42856278, 13140.32899006, 11501.39547286]),
    9.91: arr([11475.03528696, 12401.21084898, 13477.9489228 , 11842.22663386])}

M_AnalysisWF = {"TriggerID":0, "Timestamp":1, "PixCol":2, "PixRow":3, "PixInCluster":4, "Charge":5, 
         "Baseline":6, "Amplitude":7, "NoiseRMS":8, "Risetime":9, "Falltime":10, "ToA":11, "ToT":12, "ClstSize":13, "ClstCol":14, "ClstRow":15, "ClstCharge":16, "TrkX":17, "TrkY":18, "TrkResX":19, "TrkResY":20}
M_WFDigitizer = {"TriggerID":0, "PixCol":1, "PixRow":2, "BaselineEst":3, "AmplitudeEst":4, "NoiseRMS":5, "Baseline":6, "Amplitude":7, "Risetime":8, "ToA":9, "ToT":10, "chi2red":11, "Timestamp":12, "fitp_0":13, "fitp_1":14}
                 
                 
settings_dict = {}
settings_dict[184] = {}
settings_dict[184]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/184_analysis.csv"
# settings_dict[184]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/scans/184_thr220.csv"
settings_dict[184]["data_type"] = "TB Sep 24" 
settings_dict[184]["setting"] = "Setting 2"
settings_dict[184]["sample"] = "#10, 4um, w/o hole"
settings_dict[184]["title"] = "Run 184"
settings_dict[184]["krum_bias_trim"] = arr([4.57,6.78,5.53,5.53])
settings_dict[184]["i_krum"] = "2nA via Carboard"
settings_dict[184]["v_dummypix"] = 340
settings_dict[184]["bias_v"] = -3.6
settings_dict[184]["nEvts"] = 4.26E6
settings_dict[184]["purity"] = 0.01562
settings_dict[184]["runID"] = 184
settings_dict[184]["threshold_1%"] = arr([17.51,15.86,15.94,16.73])
settings_dict[184]["M"] = M_AnalysisWF



settings_dict[190] = {}
settings_dict[190]["filepath"] =  "/home/jona/DESY/analysis_TB/output/csv/190_analysis.csv"
settings_dict[190]["data_type"] = "TB Sep 24" 
settings_dict[190]["setting"] = "Setting 3"
settings_dict[190]["sample"] = "#10, 4um, w/o hole"
settings_dict[190]["title"] = "Run 190"
settings_dict[190]["krum_bias_trim"] = arr([2.32,3.30,3.30,3.30])
settings_dict[190]["i_krum"] = "2nA via Carboard"
settings_dict[190]["v_dummypix"] = 340
settings_dict[190]["bias_v"] = -3.6
settings_dict[190]["corrected"] = True
settings_dict[190]["threshold"] = 0.025
settings_dict[190]["nEvts"] = 6.55E6
settings_dict[190]["purity"] = 0.01566
settings_dict[190]["runID"] = 190
settings_dict[190]["threshold_1%"] = arr([19.177, 16.791, 18.199, 18.134])
settings_dict[190]["M"] = M_AnalysisWF

settings_dict[192] = {}
settings_dict[192]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/192_analysis.csv"
settings_dict[192]["data_type"] = "TB Sep 24" 
settings_dict[192]["setting"] = "Setting 4"
settings_dict[192]["sample"] = "#10, 4um, w/o hole"
settings_dict[192]["title"] = "Run 192"
settings_dict[192]["krum_bias_trim"] = arr([6.78,9.91,9.91,9.91])
settings_dict[192]["i_krum"] = "2nA via Carboard"
settings_dict[192]["v_dummypix"] = 340
settings_dict[192]["bias_v"] = -3.6
settings_dict[192]["corrected"] = True
settings_dict[192]["threshold"] = 0.045
settings_dict[192]["nEvts"] = 11.73E6
settings_dict[192]["purity"] = 0.0010064
settings_dict[192]["runID"] = 192
settings_dict[192]["threshold_1%"] = arr([31.431,33.881, 30.963, 33.365])
settings_dict[192]["M"] = M_AnalysisWF

# Fe55 runs

for run in range(633,641):
    settings_dict[run] = {}
    settings_dict[run]["filepath"] = "/home/jona/DESY/analysis_Fe55/output/csv/"+str(run)+".csv"
    settings_dict[run]["sample"] = "#10 (4um)"
    settings_dict[run]["data_type"] = "Fe55 Sep 24"
    settings_dict[run]["title"] = "Run "+str(run)
    settings_dict[run]["i_krum"] = "2nA via Carboard"
    settings_dict[run]["v_dummypix"] = 340 # mV
    settings_dict[run]["v_krummref"] = 400 # mV
    settings_dict[run]["bias_v"] = -3.6
    settings_dict[run]["threshold"] = 0.015
    settings_dict[run]["runID"] = run
    settings_dict[run]["M"] = M_WFDigitizer
settings_dict[633]["krum_bias_trim"] = 5.53
settings_dict[633]["nEvts"] = 261119
settings_dict[634]["krum_bias_trim"] = 9.91
settings_dict[634]["nEvts"] = 264191
settings_dict[635]["krum_bias_trim"] = 3.30
settings_dict[635]["nEvts"] = 58975
settings_dict[636]["krum_bias_trim"] = 7.73
settings_dict[636]["nEvts"] = 103423
settings_dict[637]["krum_bias_trim"] = 2.32
settings_dict[637]["nEvts"] = 158719
settings_dict[638]["krum_bias_trim"] = 6.78
settings_dict[638]["nEvts"] = 100351
settings_dict[639]["krum_bias_trim"] = 4.57
settings_dict[639]["nEvts"] = 114687
settings_dict[640]["krum_bias_trim"] = 8.96
settings_dict[640]["nEvts"] = 163839
