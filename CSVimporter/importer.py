import numpy as np
from numpy import array as arr

from CSVimporter.runSettings import load_settings

# usage: data[runID] = load_run(runID)
def load_run(runID, MaxEvts=None):
    
    run_dict = load_settings(runID)
    
    load_data(run_dict, MaxEvts)
    
    return run_dict

# usage: load_data(run_dict, runID, n=None)
# adds data to run_dict[runID]["data"]
def load_data(run_dict, n=4*2E6):
    dataRaw = np.genfromtxt(run_dict["filepath"], delimiter=',', skip_header=1, max_rows=n)

    # remove non-complete events from the back
    counter = 0
    while len(np.unique(dataRaw[-4:,0])) != 1:
        if counter > 10:
            print("[run importer] Your data does not contain a line for each pixel per event. Exiting.")
            break
        print("[run importer] Last 4 entries do not have the same event number. Cutting last entry until they do.")
        dataRaw = dataRaw[:-1]
        counter += 1

    # calculate number of events and entries
    nEvents = int(dataRaw[:,0].size / 4)
    nEntries = dataRaw[0].size

    # reshape data to [nEvents, 4, nEntries]
    # (ie. [event, pixel, entry])
    run_dict["data"] = np.zeros([nEvents,4,nEntries])
    for i in range(4):
        run_dict["data"][:,i,:] = dataRaw[i::4]
        
    print("[run importer] Loaded run", run_dict["runID"], "with", nEvents, "events and", nEntries, "entries per event per pixel.")
    
def dict_to_arr(data, runIDs, key):
    a = data[runIDs[0]][key]
    for i in range(1,len(runIDs)):
        a = np.append(a, data[runIDs[i]][key])
    return a