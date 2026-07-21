def read_data_1_animal(path_to_file, animal_num, animals_total):
    data = pd.read_csv(path_to_file,skiprows=1)
    print(data.columns)  # See what columns exist

    signal_gcamp = data[data.columns[animal_num*2]].iloc[1::2].values
    signal_405 = data[data.columns[animal_num*2-1]].iloc[::2].values
    time_gcamp = data[data.columns[0]].iloc[1::2].values
    time_405 = data[data.columns[0]].iloc[::2].values

    event_times = data[data[data.columns[animals_total*2+1]] == 1][data.columns[0]].values[animal_num-1]
    print(event_times)

    return signal_gcamp, signal_405, time_gcamp, time_405, event_times

def read_data_all_animals(path_to_file, animals_total):
    data = pd.read_csv(path_to_file,skiprows=1)
    print(data.columns)  # See what columns exist
    signal_gcamp = np.zeros((animals_total, data[data.columns[2]].iloc[1::2].values.size));
    signal_405 = np.zeros((animals_total, data[data.columns[1]].iloc[::2].values.size));
    time_gcamp = data[data.columns[0]].iloc[1::2].values
    time_405 = data[data.columns[0]].iloc[::2].values
    for i in range(1, animals_total+1):
        signal_gcamp[i-1] = data[data.columns[i*2]].iloc[1::2].values
        signal_405[i-1] = data[data.columns[i*2-1]].iloc[::2].values
    event_times = data[data[data.columns[animals_total*2+1]] == 1][data.columns[0]].values
    return signal_gcamp, signal_405, time_gcamp, time_405, event_times