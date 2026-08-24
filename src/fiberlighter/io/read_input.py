from fiberlighter.recording import Recording

import numpy as np
import pandas as pd

#check for interleaved data format and common data format
def read_csv_file(path_to_file, header_rows = (0,1), time_column=0, gcamp_column = (1,3,5,7,9), iso_column = (2,4,6,8,10), event_column = (13, 14, 15, 16, 17, 18),  iso = True, interpolate = True, animals_total = 5)-> list[Recording]:
    """Read a fiber photometry CSV.

    Returns {"animal{i}": {"iso": {"time", "data"}, "gcamp": {...}}}.

    TODO
    ----
    - add event column extraction
    - test 
    - write tests in the tests folder
    - derive columns from header instead of hardcoding
    - parse animal names from header, fall back to animal{i}
    - detect iso vs gcamp from header names instead of assuming order
    - sniff header rows instead of hardcoding (0, 1)
    - validate len(gcamp_column) >= animals_total with a clear error
    - write error guards and messages for missing stuff etc.
    - dataclass instead of dict for better type safety and IDE support???
    - test with other data formats and possibly with nwb data and reader
    """
    recordings  = []
    data = pd.read_csv(path_to_file, header=list(header_rows))
    print("The header is ", data.columns)
        
    for animal in range(animals_total):
        recording  = Recording 
        if iso:
            iso_bool = data.iloc[:, iso_column[animal]].notna().to_numpy() # boolean array
            time_iso = data.iloc[iso_bool, time_column].to_numpy(float)
            data_iso = data.iloc[iso_bool, iso_column[animal]].to_numpy(float)
        gcamp_bool =  data.iloc[:, gcamp_column[animal]].notna().to_numpy() # boolean array
        time_gcamp = data.iloc[gcamp_bool, time_column].to_numpy(float)
        data_gcamp = data.iloc[gcamp_bool, gcamp_column[animal]].to_numpy(float)
        if interpolate:
            data_iso = np.interp(time_gcamp, time_iso, data_iso)
        # #THISIS WORK IN PROGRESS DO NOT USE ON OTHER DATA
        # if event_column is not None:
        #     event_bool = (data.iloc[:, event_column] == event_value).notna().to_numpy()
        #     print(event_bool)
        #     # events = data.iloc[event_bool, time_column].to_numpy(float)
        
        recordings.append(Recording(iso=data_iso, gcamp=data_gcamp, time=time_gcamp))
    return recordings