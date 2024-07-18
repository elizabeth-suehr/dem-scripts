import matplotlib.pyplot as plt
import numpy as np
filename_start = "liggghts_s_curl5_2024-03-13/"

folders = ["curl5_5/","curl5_6/","curl5_7/"]
radius = 1.39183e-04


for folder in folders:


    overlaps = []
    for i in range(92,1000):
        file = open(filename_start + folder + "fc" + str(i*1000) + ".vtk")

        for line in file.readlines():

            strvalues = line.split()
            if len(strvalues) == 10:
                overlap = float(strvalues[9])/radius 
                overlaps.append(overlap)

    counts, bins, bars = plt.hist(overlaps,bins=np.arange(
                            0.0, 1.0, 0.02),
                              ec="black", color="blue")
    plt.ylim([1, 200000])
    plt.xlim([0.0,1.0])
    plt.xticks([0.0,0.2,0.4,0.6,0.8, 1.0],fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("Overlap Percentage", fontsize=34)
    plt.ylabel("Number of Collisions", fontsize=34)
    plt.savefig("OverlapCheck/histogram_{0}.pdf".format(folder[:-1]),
        bbox_inches="tight",
    )




