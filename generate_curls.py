import math
import numpy as np
# import matplotlib.pyplot as plt


# quadractic method
def quadratic_method(radius_distance, radius, names):
    NUM_SPHERES = 5  # must be odd
    if NUM_SPHERES % 2 == 0:
        print("NUM_SPHERES must be odd")
        exit()

    end_seperation_length = np.array(radius_distance) * radius

    # calculate the center of each sphere
    center = np.zeros((NUM_SPHERES, 3))
    center[0] = np.array([0, 0, 0])

    count = 0

    for (sep, name) in zip(end_seperation_length, names):

        if (NUM_SPHERES - 1) * 2.0 * radius == sep:
            print("Straight rod particle generated as curl_" +
                  str(count))
            filename = "curl" + str(count)
            file = open(filename, "w")
            for i in range(NUM_SPHERES):
                stringtest = "{:.5e}".format(center[i][0] + radius * 2.0 * i) + " " + "{:.5e}".format(center[i][1]) + " " + \
                    "{:.5e}".format(center[i][2]) + " " + \
                    "{:.5e}".format(radius) + "\n"
                file.write(stringtest)
            file.close()
            count += 1
            continue

        print("Starting to generate curls with sep: ", sep, end=" ... ")

        inital_distance = radius * NUM_SPHERES//2 * 2
        inital_C = inital_distance/(sep)  # intial is larger than the target

        sign = -1
        change_C = 2.0

        last_residual = inital_distance
        count_iter = 0

        while True:
            count_iter += 1
            for i in range(1, NUM_SPHERES//2+1):
                delta = radius * 1e-4

                x = (center[i-1][0] + delta)
                center[i] = np.array([x, inital_C * x**2, 0])
                distance = np.linalg.norm(center[i] - center[i-1])

                while distance < radius * 2:
                    x += delta
                    center[i] = np.array([x, inital_C * x**2, 0])
                    distance = np.linalg.norm(center[i] - center[i-1])

            for i in range(NUM_SPHERES//2+1, NUM_SPHERES):
                center[i] = center[i - NUM_SPHERES//2]
                center[i][0] = -center[i][0]
            current_distance = center[NUM_SPHERES //
                                      2][0] - center[NUM_SPHERES - 1][0]

            if sep - current_distance > 0:
                inital_C += 1.0 * sign * change_C
                if sign == -1:
                    sign = 1
                    change_C = change_C * 0.6
            elif sep - current_distance < 0:
                inital_C += 1.0 * sign * change_C
                if sign == 1:
                    sign = -1
                    change_C = change_C * 0.6
            if abs(current_distance - sep) < 1e-7 * radius:
                break
            else:
                current_residual = abs(current_distance - sep)
                if current_residual > last_residual:
                    sign = -sign
                    change_C = change_C * 0.6

                if abs(current_residual - last_residual) < 1e-6 * radius:
                    change_C = change_C * 2.0
                last_residual = current_residual
                # print("current_distance: ", current_distance - sep)

        # # plot the spheres
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(NUM_SPHERES):
        #     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        #     x = center[i][0] + RADIUS * np.cos(u)*np.sin(v)
        #     y = center[i][1] + RADIUS * np.sin(u)*np.sin(v)
        #     z = center[i][2] + RADIUS * np.cos(v)
        #     ax.plot_wireframe(x, y, z, color="r")

        # plt.show()

        filename = "curl" + name

        file = open(filename, "w")

        for i in range(NUM_SPHERES):
            stringtest = "{:.5e}".format(center[i][0]) + " " + "{:.5e}".format(center[i][1]) + " " + \
                "{:.5e}".format(center[i][2]) + " " + \
                "{:.5e}".format(radius) + "\n"
            file.write(stringtest)

        file.close()
        count += 1
        print("Done\n")
