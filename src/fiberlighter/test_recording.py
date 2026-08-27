from fiberlighter.io.read_input import read_csv_file
import matplotlib.pyplot as plt


agrpsal = read_csv_file("src/data/20260106-AgRP-SAL-3Hz_0000.csv")

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

agrpsal[4].visualization.basic_plot()
plt.show()

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

# pagdcz[5].visualization.basic_plot()
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
# pagsal[5].visualization.basic_plot()
# plt.show()


# data1 = read_csv_file("src/data/DATA1-TH.csv")

# fig, ax = plt.subplots(
#     len(data1),
#     1,
#     figsize=(12, 3 * len(data1)),
#     sharex=True
# )
# # if len(data1) == 1:
# #     ax = [ax]
# # for record, ax in zip(data1, ax):
# #     plt.sca(ax)
# #     record.bleach_correction.double_exponential().motion_correction.robust_fit().visualization.plot_gcamp(ax)

# # plt.tight_layout()
# # plt.show()

# data1[4].visualization.basic_plot()
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
# data2[5].visualization.basic_plot()
# plt.show()