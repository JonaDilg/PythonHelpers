import numpy as np
from numpy import array as arr

from CSVimporter.runSettings import load_settings

# usage: 
# runs = {runID:{}}
# runs[runID] = load_run(runID)
def load_run(runID, MaxEvts=None, filepath=None):
    
    run = load_settings(runID) # this is a dict
    if filepath is not None:
        run["filepath"] = filepath
    
    load_data(run, run["filepath"], MaxEvts) # add events as array into run 
    # access data via run["data"][event,pixel,entryID]
    # entryIDs are encoded in the dict run["M"] (from csv header line)
    return run

# usage:
# runs = load_scan(runID, "thr", [30,60,90], n=None)
# filename must be scans/<runID>_<name><entry>.csv
def load_scan(runID, name, entries, n=None):
    runs = {}
    for i in range(len(entries)):
        # print(f"Loading run {runID} with {n} events for {name} = {entries[i]}")
        runs[entries[i]] = load_settings(runID)
        filepath = "/home/jona/DESY/analysis_TB/output/csv/scans/184_"+name+str(entries[i])+".csv"
        load_data(runs[entries[i]], filepath, n)
    return runs

# usage: load_data(run, runID, n=None)
# adds data to run_dict[runID]["data"]
def load_data(run, filepath,  n=4*2E6):
    dataRaw = np.genfromtxt(filepath, delimiter=',', skip_header=1, max_rows=n)

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

    # reshape data to [nEvents, 4, nEntries] (ie. [event, pixel, entry])
    run["data"] = np.zeros([nEvents,4,nEntries])
    for i in range(4):
        run["data"][:,i,:] = dataRaw[i::4]
        
    # get the csv header (first line)
    headerStr = np.loadtxt(filepath, delimiter=",", max_rows=1, dtype=str)
    run["M"] = {}
    for i in range(len(headerStr)):
        run["M"][str(headerStr[i])] = i

    print("[run importer] Loaded run", run["runID"], "with", nEvents, "events and", nEntries, "entries per event per pixel.")
    
    # the run dict we edited here is a pointer to the one we used before, no need to return anything
    
def dict_to_arr(data, runIDs, key):
    a = data[runIDs[0]][key]
    for i in range(1,len(runIDs)):
        a = np.append(a, data[runIDs[i]][key])
    return a