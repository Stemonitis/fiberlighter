from fiberlighter.io.read_input import read_csv_file


recordings = read_csv_file("src/data/DATA2.csv")

print(recordings[0].iso_work)   # Recording object
recordings[1]

recordings[0].filtering.lowpass_filter()