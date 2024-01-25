from ast import Num
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET



from numpy.lib.scimath import sqrt


start_count = [61000000, 61000000, 61000000, 61000000, 64250000, 22750000, 18500000]

filename_col = "./collision_check/curl_3/vf_30/collision.txt" #"../Collisions/Collision_data/curl_3/vf_5/collision.data"
file = open(filename_col)

reset = False
count = -1
colliding_aggolmerates = []

for line in file.readlines():


    if line.find("file") == -1:
        stringvalues = line.split()
        colliding_aggolmerates.append(int(stringvalues[0]))
        colliding_aggolmerates.append(int(stringvalues[1]))
    elif count != -1:
        filename = "vtp_files/curl_3/vf_30/" + str(start_count[4] + count * 2000) + "bal.vtp"
        new_filename = "colored_particles/curl_3/vf_30/" + str(count * 2000) + "new.vtp"

        mytree = ET.parse(filename)
        myroot = mytree.getroot()

        elementTest = (
            myroot.find("PolyData").find("Piece").find("PointData").findall("DataArray")
        )

        for element in elementTest:
            if element.get("Name") == "Agglomerates":
                text = element.text

                text_values = text.split()
                # print(int(len(text_values)/5))
                # text_values = text_values[1:-1]
                Values = []
                for j in range(len(text_values)):
                    Values.append(int(text_values[j]))

                new_values = ""
                for i in range(int(len(Values) / 5)):
                    if i in colliding_aggolmerates:
                        print("collision found!")
                        for j in range(5):
                            new_values += "" + str(0) + " "
                    else:
                        for j in range(5):
                            new_values += "" + str(1) + " "

                element.text = str(new_values)

        mytree.write(new_filename)

        colliding_aggolmerates = []
        count += 1
    else:
        count += 1

    

    
    
