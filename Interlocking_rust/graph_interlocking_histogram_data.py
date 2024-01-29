import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.integrate as integrate

dt = 8.17059e-8
frames = 2000
file_dir_1 = ["curl1",
              "curl2",
              "curl3",
              "curl4",
              "curl5"]
file_dir_2 = ["0", "1", "2", "3", "4", "5", "6", "7"]

vf_30_results = []
vf_40_results = []
vf_45_results = []

cutoffs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for cutoff in cutoffs:
    for i in range(len(file_dir_1)):
        for j in range(len(file_dir_2)):

            file = open("../collision_check/" +
                        file_dir_1[i] + file_dir_2[j] + "array.txt", "r")
            line = file.readline()
            line = line[:-2]
            string_values = line.split(", ")

            a = []
            for k in range(len(string_values)):
                a.append(float(string_values[k]))

            a = np.array(a)
            a = a * frames * dt * 100

            # x_first = np.linspace(1 * frames * dt * 100, 20 * frames * dt * 100, max_value)
            # p = np.polyfit(x_first, np.log(values[:20]), 1)

            # # Convert the polynomial back into an exponential
            # a = np.exp(p[1])
            # b = p[0]
            # x_fitted = np.linspace(np.min(x_first), np.max(x_first), 100)
            # y_fitted = a * np.exp(b * x_fitted)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            # print(a)
            counts, bins, bars = plt.hist(a, bins=np.arange(
                frames * dt * 100, 200 * frames * dt * 100, 2 * frames * dt * 100), ec="white", color="blue")

            # print("bins")
            # print(bins)
            # print("bars")
            # print(counts)

            test_count = 0
            for bin in bins:
                if bin >= cutoff:
                    break
                else:
                    test_count += 1

            summ = 0.0
            for k in range(test_count, len(counts)):
                summ += counts[k] * bins[k]

            result = summ

            center_bins = []
            for k in range(0, len(bins)-1):
                center_bins.append((bins[k] + bins[k+1])/2)

            count = 0
            for k in range(0, len(counts)):
                if counts[k] > 1.0:
                    count += 1
                else:
                    count -= 1
                    break
            if j > 3:
                p = np.polyfit(center_bins[:count], np.log(counts[:count]), 1)
                aa = np.exp(p[1])
                bb = p[0]
                x_fitted = np.linspace(
                    np.min(center_bins[:count]), np.max(center_bins[:count]), 100)
                y_fitted = aa * np.exp(bb * x_fitted)
                # plt.plot(x_fitted, y_fitted)
                # plt.plot(center_bins,counts)

                #
                # result = integrate.quad(
                #     lambda x: aa * np.exp(bb * x) * x, 1.25, 4)

                # print(file_dir_1[i] + file_dir_2[j])

                if j == 4:
                    # result[0]  use this instead of integration
                    vf_30_results.append(result)
                if j == 5:
                    vf_40_results.append(result)
                if j == 6:
                    vf_45_results.append(result)

                # print(len(center_bins))
                # print(len(counts))

            plt.ylim([1, 40000])
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.yscale("log")
            plt.xlabel("Interlocking Duration ($\.γ$t)", fontsize=34)
            plt.ylabel("Number of Interlocks", fontsize=34)

            plt.savefig(
                file_dir_1[i][:-1] + "_" + file_dir_2[j][:-1] + ".pdf",
                bbox_inches="tight",
            )
    print(cutoff)
    print("vf_30=" + str(vf_30_results))
    print("vf_40=" + str(vf_40_results))
    print("vf_45=" + str(vf_45_results))
    vf_30_results.clear()
    vf_40_results.clear()
    vf_45_results.clear()

    print(frames * dt * 1800)
