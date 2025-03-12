from numpy import array as arr

def load_settings(runID):
    return settings_dict[runID]

charge_calibration = {
    0: arr([10100.13164889, 10363.74574137, 10670.19824758, 10031.88867229]),
    1: arr([10331.78239925, 10637.14243425, 10998.04551077, 10440.91481794]),
    2: arr([10484.04549243, 11056.13741112, 11343.78887434, 10718.16068878]),
    3: arr([10683.1111317 , 11370.82939689, 11738.21428846, 11063.603456  ]),
    4: arr([11048.71202011, 11472.03945169, 12507.79790097, 10893.02044626]),
    5: arr([11188.15599342, 11722.52276237, 12830.12738093, 11246.9124095 ]),
    6: arr([11342.9575507 , 12104.42856278, 13140.32899006, 11501.39547286]),
    7: arr([11475.03528696, 12401.21084898, 13477.9489228 , 11842.22663386])}

M_AnalysisWF = {"TriggerID":0, "Timestamp":1, "PixCol":2, "PixRow":3, "PixInCluster":4, "Charge":5, 
         "Baseline":6, "Amplitude":7, "NoiseRMS":8, "Risetime":9, "Falltime":10, "ToA":11, "ToT":12, "ClstSize":13, "ClstCol":14, "ClstRow":15, "ClstCharge":16, "TrkX":17, "TrkY":18, "TrkCol":19, "TrkRow":20, "TrkResX":21, "TrkResY":22}
M_WFDigitizer = {"TriggerID":0, "EventStartTime":1, "PixCol":2, "PixRow":3, 
    "BaselineEst":4, "AmplitudeEst":5, "ChargeEst":6,"NoiseRMS":7,
    "Baseline":8, "Amplitude":9, "Charge":10, "Risetime":11, "ToA":12, "ToT":13, "Timestamp":14} # new & corrected. in use from 2025-03-11
# M_WFDigitizer = {"TriggerID":0, "PixCol":1, "PixRow":2, "BaselineEst":3, "AmplitudeEst":4, "NoiseRMS":5, "Baseline":6, "Amplitude":7, "Risetime":8, "ToA":9, "ToT":10, "chi2red":11, "Timestamp":12, "fitp_0":13, "fitp_1":14}
                 

# DEFAULTS     
campaigns = ["Fe55 Sep 24", "TB Sep 24"]
sample = r"#10, 4$\,$um ngap"
i_krum = r"2$\,$nA, Carboard"
                      
settings_dict = {}

runID = 184
settings_dict[runID] = {}
settings_dict[runID]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/184_analysis.csv"
# settings_dict[184]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/scans/184_thr220.csv"
settings_dict[runID]["data_type"] = campaigns[1]
settings_dict[runID]["setting"] = "Setting 2"
settings_dict[runID]["sample"] = sample
settings_dict[runID]["title"] = "Run 184"
settings_dict[runID]["krum_trim_nominal"] = arr([4.57,6.78,5.53,5.53])
settings_dict[runID]["krum_trim"] = arr([2,4,3,3])
settings_dict[runID]["i_krum"] = i_krum
settings_dict[runID]["v_dummypix"] = 340
settings_dict[runID]["bias_v"] = -3.6
settings_dict[runID]["nEvts"] = 4.26E6
settings_dict[runID]["purity"] = 0.01562
settings_dict[runID]["runID"] = runID
settings_dict[runID]["threshold"] = 220
# settings_dict[runID]["threshold_1%"] = arr([17.51,15.86,15.94,16.73])
settings_dict[runID]["M"] = M_AnalysisWF


runID = 190
settings_dict[runID] = {}
settings_dict[runID]["filepath"] =  "/home/jona/DESY/analysis_TB/output/csv/190_analysis.csv"
settings_dict[runID]["data_type"] = campaigns[1]
settings_dict[runID]["setting"] = "Setting 3"
settings_dict[runID]["sample"] = sample
settings_dict[runID]["title"] = "Run 190"
settings_dict[runID]["krum_trim_nominal"] = arr([2.32,3.30,3.30,3.30])
settings_dict[runID]["krum_trim"] = arr([0,1,1,1])
settings_dict[runID]["i_krum"] = i_krum
settings_dict[runID]["v_dummypix"] = 340
settings_dict[runID]["bias_v"] = -3.6
settings_dict[runID]["corrected"] = True
settings_dict[runID]["nEvts"] = 6.55E6
settings_dict[runID]["purity"] = 0.01566
settings_dict[runID]["runID"] = runID
settings_dict[runID]["threshold"] = 220
# settings_dict[runID]["threshold_1%"] = arr([19.177, 16.791, 18.199, 18.134])
settings_dict[runID]["M"] = M_AnalysisWF

runID = 192
settings_dict[runID] = {}
settings_dict[runID]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/192_analysis.csv"
settings_dict[runID]["data_type"] = campaigns[1]
settings_dict[runID]["setting"] = "Setting 4"
settings_dict[runID]["sample"] = sample
settings_dict[runID]["title"] = "Run 192"
settings_dict[runID]["krum_trim_nominal"] = arr([6.78,9.91,9.91,9.91])
settings_dict[runID]["krum_trim"] = arr([4,7,7,7])
settings_dict[runID]["i_krum"] = i_krum
settings_dict[runID]["v_dummypix"] = 340
settings_dict[runID]["bias_v"] = -3.6
settings_dict[runID]["corrected"] = True
settings_dict[runID]["nEvts"] = 11.73E6
settings_dict[runID]["purity"] = 0.0010064
settings_dict[runID]["runID"] = runID
settings_dict[runID]["threshold"] = 600
# settings_dict[runID]["threshold_1%"] = arr([31.431,33.881, 30.963, 33.365])
settings_dict[runID]["M"] = M_AnalysisWF

# Fe55 runs
for runID in range(633,641):
    settings_dict[runID] = {}
    settings_dict[runID]["filepath"] = "/home/jona/DESY/analysis_Fe55/output/csv/chargeCal_"+str(runID)+".csv"
    settings_dict[runID]["sample"] = sample
    settings_dict[runID]["data_type"] = campaigns[0]
    settings_dict[runID]["title"] = "Run "+str(runID)
    settings_dict[runID]["i_krum"] = i_krum
    settings_dict[runID]["v_dummypix"] = 340 # mV
    settings_dict[runID]["v_krummref"] = 400 # mV
    settings_dict[runID]["bias_v"] = -3.6
    settings_dict[runID]["threshold"] = 0.015
    settings_dict[runID]["runID"] = runID
    settings_dict[runID]["M"] = M_WFDigitizer
runID = 633
settings_dict[runID]["krum_trim_nominal"] = 5.53
settings_dict[runID]["krum_trim"] = 3
settings_dict[runID]["nEvts"] = 261119
runID = 634
settings_dict[runID]["krum_trim_nominal"] = 9.91
settings_dict[runID]["krum_trim"] = 7
settings_dict[runID]["nEvts"] = 264191
runID = 635
settings_dict[runID]["krum_trim_nominal"] = 3.30
settings_dict[runID]["krum_trim"] = 1
settings_dict[runID]["nEvts"] = 58975
runID = 636
settings_dict[runID]["krum_trim_nominal"] = 7.73
settings_dict[runID]["krum_trim"] = 5
settings_dict[runID]["nEvts"] = 103423
runID = 637
settings_dict[runID]["krum_trim_nominal"] = 2.32
settings_dict[runID]["krum_trim"] = 0
settings_dict[runID]["nEvts"] = 158719
runID = 638
settings_dict[runID]["krum_trim_nominal"] = 6.78
settings_dict[runID]["krum_trim"] = 4
settings_dict[runID]["nEvts"] = 100351
runID = 639
settings_dict[runID]["krum_trim_nominal"] = 4.57
settings_dict[runID]["krum_trim"] = 2
settings_dict[runID]["nEvts"] = 114687
runID = 640
settings_dict[runID]["krum_trim_nominal"] = 8.96
settings_dict[runID]["krum_trim"] = 6
settings_dict[runID]["nEvts"] = 163839


