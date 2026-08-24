from fiberlighter.io.read_input import read_csv_file
from fiberlighter.core_algorithms import bleach_correction, filtering, motion_correction


recordings = read_csv_file("src/data/DATA2.csv")

print(recordings[0].iso_work)   # Recording object
recordings[1]