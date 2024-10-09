from numpy import array as arr

def load_settings(runID):
    return settings_dict[runID]

settings_dict = {}
settings_dict[184] = {}
settings_dict[184]["filepath"] = "/home/jona/DESY/analysis_TB/output/csv/184_hits.csv"
settings_dict[184]["setting"] = "Setting 2"
settings_dict[184]["sampe"] = "#10, 4um, w/o hole"
settings_dict[184]["title"] = "Run 184"
settings_dict[184]["krum_bias_trim"] = arr([4.57,6.78,5.53,5.53])
settings_dict[184]["i_krum"] = "2nA via Carboard"
settings_dict[184]["v_dummypix"] = 350
settings_dict[184]["bias_v"] = -3.6
settings_dict[184]["corrected"] = True
settings_dict[184]["threshold"] = 0.022
settings_dict[184]["nEvts"] = 4.26E6
settings_dict[184]["purity"] = 0.01562
settings_dict[184]["runID"] = 184