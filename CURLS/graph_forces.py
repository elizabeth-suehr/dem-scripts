import matplotlib.pyplot as plt
import numpy as np
import math


filename_start = "liggghts_s_curl5_2024-03-13/"

folders = ["curl5_7/"]
radius = 1.39183e-04
center = [0.004318, 0.004318, 0.004318/2]
length = 0.008636 
width = 0.008636 
height = 0.004318



for folder in folders:
   
    for i in range(523,5000):
        x1 = []
        y1 = []
        z1 = []
        x2 = []
        y2 = []
        z2 = []
        force_magnitude = []
    
        file = open(filename_start + folder + "fc" + str(i*10000) + ".vtk")

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(90, -90)
        fig.set_facecolor('black')
        ax.set_facecolor('black') 
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        ax.grid(False)


        for line in file.readlines():

            strvalues = line.split()
            if len(strvalues) == 10:
                if abs(float(strvalues[0])-float(strvalues[3])) >= 0.004318 or abs(float(strvalues[1])-float(strvalues[4])) >= 0.004318 or abs(float(strvalues[2])-float(strvalues[5])) >= 0.003318:
                    continue
                x1.append(float(strvalues[0]))
                y1.append(float(strvalues[1]))
                z1.append(float(strvalues[2]))
                x2.append(float(strvalues[3]))
                y2.append(float(strvalues[4]))
                z2.append(float(strvalues[5]))
                force_magnitude.append(math.sqrt(float(strvalues[6])**2 + float(strvalues[7])**2 + float(strvalues[8])**2))
                
                
        max = np.max(force_magnitude)
        colors = plt.cm.jet(np.linspace(0,1,len(force_magnitude)))
        linecolors = np.array(force_magnitude)
        force_magnitude = force_magnitude/max



        for j in range(len(x1)):
            ax.plot3D([x1[j],x2[j]],[y1[j],y2[j]],[z1[j],z2[j]],color=plt.cm.plasma(force_magnitude[j]))
        plt.savefig("liggghts_s_curl5_2024-03-13/curl5_7_force_video/curl5_7_forces_{:04d}.png".format(i),
            bbox_inches="tight",
        )
        fig.clf()
        plt.clf()

        


    # plt.ylim([1, 200000])
    # plt.xlim([0.0,1.0])
    # plt.xticks([0.0,0.2,0.4,0.6,0.8, 1.0],fontsize=25)
    # plt.yticks(fontsize=25)
    # plt.xlabel("Overlap Percentage", fontsize=34)
    # plt.ylabel("Number of Collisions", fontsize=34)
    # 




