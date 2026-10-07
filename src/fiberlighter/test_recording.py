from fiberlighter.io.read_input import read_csv_file
import numpy as np
import matplotlib.pyplot as plt
print(plt.get_backend())


# Channel assignment for this recording: EXC 1 = iso, EXC 2 = GCaMP.
agrpsal = read_csv_file("src/data/20260106-AgRP-SAL-3Hz_0000.csv", gcamp_column = (2,4,6,8,10,12,14,16,18), iso_column = (1,3,5,7,9,11,13,15,17), event_column = (19,), animals_total = 9)

# print(len(agrpsal))   # Recording object
# fig, ax = plt.subplots(
#     len(agrpsal),
#     1,
#     figsize=(12, 3 * len(agrpsal)),
#     sharex=True
# )
# if len(agrpsal) == 1:
#     ax = [ax]
# for record, ax in zip(agrpsal, ax):
#     plt.sca(ax)
#     record.visualization.basic_plot(ax)

# plt.tight_layout()
# plt.show()
# agrpsal[4].diagnostics.frequency_analysis()
# agrpsal[4].diagnostics.frequency_analysis_whole(heatmap=True)
# agrpsal[4].diagnostics.frequency_analysis_timeframed()

# agrpsal[4].diagnostics.coherence_whole(window_seconds=120)
# agrpsal[4].diagnostics.coherence_timeframed(window_seconds=120, time_window_seconds=600)
# agrpsal[4].diagnostics.compare_oscillations(channel="gcamp")
# agrpsal[4].diagnostics.bosc_analysis(channel="gcamp")
# agrpsal[4].diagnostics.byb(channel="gcamp")

# agrpsal[4].normalization.deltaF_over_Fo().visualization.basic_plot()
# agrpsal[4].visualization.basic_plot()
# agrpsal[4].bleach_correction.highpass_filter(cutoff=0.0002).visualization.basic_plot()
# agrpsal[4].bleach_correction.highpass_filter().visualization.basic_plot()
# agrpsal[4].bleach_correction.double_exponential(plot_fit=True).visualization.basic_plot()
recording = agrpsal[4].bleach_correction.double_exponential(plot_fit=True).visualization.plot_raw_with_baseline()
# recording.visualization.plot_raw_with_baseline()


# agrpex4 = read_csv_file("src/data/20260107-AgRP-EX4_0000.csv")


# # fig, ax = plt.subplots(
# #     len(agrpex4),
# #     1,
# #     figsize=(12, 3 * len(agrpex4)),
# #     sharex=True
# # )
# # if len(agrpex4) == 1:
# #     ax = [ax]
# # for record, ax in zip(agrpex4, ax):
# #     plt.sca(ax)
# #     record.visualization.basic_plot(ax)

# # plt.tight_layout()
# # plt.show()

# agrpex4[4].visualization.basic_plot()
# plt.show()

# pagdcz = read_csv_file("src/data/20260220-PAG-DCZ.csv")
# # fig, ax = plt.subplots(
# #     len(pagdcz),
# #     1,
# #     figsize=(12, 3 * len(pagdcz)),
# #     sharex=True
# # )
# # if len(pagdcz) == 1:
# #     ax = [ax]
# # for record, ax in zip(pagdcz, ax):
# #     plt.sca(ax)
# #     record.visualization.basic_plot(ax)

# # plt.tight_layout()
# # plt.show()

# pagdcz[4].visualization.basic_plot()
# plt.show()

# pagsal = read_csv_file("src/data/20260220-PAG-SAL.csv")

# # fig, ax = plt.subplots(
# #     len(pagsal),
# #     1,
# #     figsize=(12, 3 * len(pagsal)),
# #     sharex=True
# # )
# # if len(pagsal) == 1:
# #     ax = [ax]
# # for record, ax in zip(pagsal, ax):
# #     plt.sca(ax)
# #     record.visualization.basic_plot(ax)

# # plt.tight_layout()
# # plt.show()
# pagsal[4].visualization.basic_plot()
# plt.show()


# data1 = read_csv_file("src/data/DATA1-TH.csv")

# # fig, ax = plt.subplots(
# #     len(data1),
# #     1,
# #     figsize=(12, 3 * len(data1)),
# #     sharex=True
# # )
# # if len(data1) == 1:
# #     ax = [ax]
# # for record, ax in zip(data1, ax):
# #     plt.sca(ax)
# #     record.bleach_correction.double_exponential().motion_correction.robust_fit().visualization.plot_gcamp(ax)

# # plt.tight_layout()
# # plt.show()

# data1[0].visualization.basic_plot()
# plt.show()


# data2 = read_csv_file("src/data/DATA2.csv")

# # fig, ax = plt.subplots(
# #     len(data2),
# #     1,
# #     figsize=(12, 3 * len(data2)),
# #     sharex=True
# # )
# # if len(data2) == 1:
# #     ax = [ax]
# # for record, ax in zip(data2, ax):
# #     plt.sca(ax)
# #     record.visualization.basic_plot(ax)

# # plt.tight_layout()
# # plt.show()
# data2[4].visualization.basic_plot()
# plt.show()











# agrpsal = read_csv_file("src/data/20260106-AgRP-SAL-3Hz_0000.csv", gcamp_column = (1,3,5,7,9), iso_column = (2,4,6,8,10))
# agrpsal[4].visualization.basic_plot()
# plt.show()
# agrpsal[4].bleach_correction.double_exponential()
# agrpsal[4].visualization.basic_plot()
# plt.show()
# print(np.corrcoef(agrpsal[4].iso_work, agrpsal[4].gcamp_work)[0, 1])
# agrpsal[4].motion_correction.robust_fit()
# agrpsal[4].visualization.basic_plot()
# plt.show()

# print(np.median(agrpsal[4].iso), np.median(agrpsal[4].gcamp))
# agrpsal[4].noise_correction.bandpass_filter(0.1, 1)
# print(np.corrcoef(agrpsal[4].iso_work, agrpsal[4].gcamp_work)[0, 1])
# agrpsal[4].visualization.basic_plot()
# plt.show()



#Despike 
# agrpsal[4].noise_correction.hampel_filter().bleach_correction.double_exponential().motion_correction.robust_fit()


# agrpsal[3].visualization.basic_plot()
# plt.show()







# data1 = read_csv_file("src/data/DATA1-TH.csv")
# data1[0].visualization.basic_plot()
# plt.show()
# data1[0].bleach_correction.double_exponential()
# data1[0].visualization.basic_plot()
# plt.show()
# print(np.corrcoef(data1[0].iso_work, data1[0].gcamp_work)[0, 1])
# data1[0].motion_correction.robust_fit()
# data1[0].visualization.basic_plot()
# plt.show()
