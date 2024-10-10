from numpy import array as arr

def load_settings(runID):
    return settings_dict[runID]

settings_dict = {}
settings_dict[184] = {}
settings_dict[184]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/184_full.csv"
settings_dict[184]["data_type"] = "TB Sep 24" 
settings_dict[184]["setting"] = "Setting 2"
settings_dict[184]["sample"] = "#10, 4um, w/o hole"
settings_dict[184]["title"] = "Run 184"
settings_dict[184]["krum_bias_trim"] = arr([4.57,6.78,5.53,5.53])
settings_dict[184]["i_krum"] = "2nA via Carboard"
settings_dict[184]["v_dummypix"] = 340
settings_dict[184]["bias_v"] = -3.6
settings_dict[184]["corrected"] = True
settings_dict[184]["threshold"] = 0.022
settings_dict[184]["nEvts"] = 4.26E6
settings_dict[184]["purity"] = 0.01562
settings_dict[184]["runID"] = 184

settings_dict[190] = {}
settings_dict[190]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/190_full.csv"
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

settings_dict[192] = {}
settings_dict[192]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/192_full.csv"
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