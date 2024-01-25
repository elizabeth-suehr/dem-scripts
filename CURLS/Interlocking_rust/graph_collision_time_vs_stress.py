import sys
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage


class HandlerLineImage(HandlerBase):
    def __init__(self, path, space=15, offset=10):
        self.space = space
        self.offset = offset
        self.image_data = plt.imread(path)
        super(HandlerLineImage, self).__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):

        width = width * 1.4
        height = height * 1.4

        l = matplotlib.lines.Line2D(
            [
                xdescent + self.offset,
                xdescent + (width - self.space) / 3.0 + self.offset,
            ],
            [ydescent + height / 2.0, ydescent + height / 2.0],
        )
        l.update_from(orig_handle)
        l.set_clip_on(False)
        l.set_transform(trans)

        bb = Bbox.from_bounds(
            xdescent + (width + self.space) / 3.0 + self.offset,
            ydescent,
            height * self.image_data.shape[1] / self.image_data.shape[0],
            height,
        )

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)
        return [l, image]


curl_image_0 = HandlerLineImage("../pictures/curl_0_thumbnail.png")
curl_image_1 = HandlerLineImage("../pictures/curl_1_thumbnail.png")
curl_image_2 = HandlerLineImage("../pictures/curl_2_thumbnail.png")
curl_image_3 = HandlerLineImage("../pictures/curl_3_thumbnail.png")
curl_image_4 = HandlerLineImage("../pictures/curl_4_thumbnail.png")
curl_image_5 = HandlerLineImage("../pictures/curl_5_thumbnail.png")


directory_1 = ["curl_0/", "curl_1/", "curl_2/", "curl_3/", "curl_4/", "curl_5/"]
directory_2 = [ "vf_10/", "vf_20/","vf_30/", "vf_40/", "vf_45/"]  # "vf_2.5/", "vf_5/",
particle_count = [
       
        1660 // 5,
        3320 // 5,
        4980 // 5,
        6640 // 5,
        1494,
    ] #  83, 830 // 5,

curl_1_xy = [
    # 0.1877111833403819,
    # 0.10246297389885886,
    0.0845014129956796,
    0.17160279990536267,
    0.38518694624007566,
    1.2601669132666753,
    3.1926567012548874,
]
curl_2_xy = [
    # 0.19404816608413739,
    # 0.11154353414401684,
    0.08455231440520286,
    0.1618638938686997,
    0.41478986014567826,
    1.5625909355739045,
    4.503070534977373,
]
curl_3_xy = [
    # 0.20577983854854612,
    # 0.11278232919171939,
    0.08364202357280028,
    0.15229831622603163,
    0.4087975686937332,
    1.619378180875923,
    4.7389222144158545,
]
curl_4_xy = [
    # 0.23245459959343445,
    # 0.1264980330800776,
    0.08778560226359522,
    0.1511810110374861,
    0.3836904126260188,
    1.4772057308271571,
    4.164594750944124,
]


volume_fraction = [ 0.025, 0.05, 0.10, 0.20,0.30, 0.4, 0.45]  # 

collision_counts = np.zeros((6, 5, 1800))
collision_averages = np.zeros((6, 5))
collision_percentages = np.zeros((6, 5))
binding_average = np.zeros((6, 5))


for i in range(len(directory_1)):

    for j in range(len(directory_2)):
        k = 0
        if i == 0 or i == 5:
            collision_counts[i][j][k] = 0
            continue

        filename = (
            "collision_check/"+
             directory_1[i] + directory_2[j] + "collision.txt"
        )

        per_particle_collision_count = np.zeros((particle_count[j], particle_count[j]))
        list_of_particles = []
        with open(filename) as f:
            count = 0
            for line in f:
                stringvalues = line.split()
                if stringvalues[0] == "file":

                    collision_counts[i][j][k] = count
                    k = int(stringvalues[1])
                    count = 0
                    list_of_particles = []
                elif float(stringvalues[0]) >= 0 and float(stringvalues[1]) >= 0:
                    if not int(stringvalues[0]) in list_of_particles:
                        count += 1
                        list_of_particles.append(int(stringvalues[0]))
                    if not int(stringvalues[1]) in list_of_particles:
                        count += 1
                        list_of_particles.append(int(stringvalues[1]))

                    value_one = int(stringvalues[0])
                    value_two = int(stringvalues[1])
                   
                    if value_one < value_two:
                        per_particle_collision_count[value_one][value_two] += 1
                    else:
                        per_particle_collision_count[value_two][value_one] += 1
        
        per_particle_total_time = np.zeros((particle_count[j]))
        per_particle_redudant_count = np.zeros((particle_count[j]))

        for k in range(particle_count[j]):
            for l in range(particle_count[j]):
                if per_particle_collision_count[k][l] > 0:
                    per_particle_total_time[k] += per_particle_collision_count[k][l]
                    per_particle_total_time[l] += per_particle_collision_count[k][l]
                    per_particle_redudant_count[k] += 1
                    per_particle_redudant_count[l] += 1

        # print(per_particle_total_time)
        # print(per_particle_redudant_count)
        # binding_collision_time =  np.zeros((particle_count[j]))
        
        # for k in range(particle_count[j]):
        #     if per_particle_redudant_count[k] > 0:
        #         binding_collision_time[k] = per_particle_total_time[k] / per_particle_redudant_count[k]
                
        
        # binding_average[i][j] = np.mean(binding_collision_time)
        # total = sum(per_particle_total_time)
        # test_ave = total / 1800
        # collision_averages[i][j] = test_ave





for i in range(len(directory_1)):
    for j in range(len(directory_2)):
        collision_averages[i][j] = sum(collision_counts[i][j][:]) / len(
            collision_counts[i][j]
        )

        # print(collision_counts[i][j])
        collision_percentages[i][j] = (
            sum(collision_counts[i][j][:])
            / len(collision_counts[i][j][:])
            / particle_count[j]
            * 100
        )


print(binding_average)



matplotlib.rcParams.update({"font.size": 18})

plt.figure(1)
marks = ["o", "v", "x", "s", "*", "D", "+"]
labelnames = ["Curl 1", "Curl 2", "Curl 3", "Curl 4"]
curls = [object] * 4
(curls[0],) = plt.plot(
    collision_averages[1][:], curl_1_xy, "--", marker=marks[1], color="C1"
)
(curls[1],) = plt.plot(
    collision_averages[2][:], curl_2_xy, "--", marker=marks[2], color="C2"
)
(curls[2],) = plt.plot(
    collision_averages[3][:], curl_3_xy, "--", marker=marks[3], color="C3"
)
(curls[3],) = plt.plot(
    collision_averages[4][:], curl_4_xy, "--", marker=marks[4], color="C4"
)

plt.xlabel("Normalized Interlocking Time")
plt.ylabel("Shear Stress $(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
plt.yscale("log")
plt.yscale("log")
plt.legend(
    curls,
    labelnames,
    handler_map={
        curls[0]: curl_image_1,
        curls[1]: curl_image_2,
        curls[2]: curl_image_3,
        curls[3]: curl_image_4,
    },
    frameon=False,
    handlelength=5.0,
    labelspacing=0.0,
    fontsize=14,
    borderpad=0.15,
    loc=2,
    handletextpad=0.2,
    borderaxespad=0.15,
)


# print(collision_averages[1][:])
# print(collision_averages[2][:])
# print(collision_averages[3][:])
# print(collision_averages[4][:])
plt.savefig("interlocking_time_vs_stress_xy.pdf", bbox_inches="tight")




# plt.figure(2)
# marks = ["o", "v", "x", "s", "*", "D", "+"]
# labelnames = ["Curl 1", "Curl 2", "Curl 3", "Curl 4"]
# curls = [object] * 4
# (curls[0],) = plt.plot(
#     binding_average[1][:], curl_1_xy, "--", marker=marks[1], color="C1"
# )
# (curls[1],) = plt.plot(
#     binding_average[2][:], curl_2_xy, "--", marker=marks[2], color="C2"
# )
# (curls[2],) = plt.plot(
#     binding_average[3][:], curl_3_xy, "--", marker=marks[3], color="C3"
# )
# (curls[3],) = plt.plot(
#     binding_average[4][:], curl_4_xy, "--", marker=marks[4], color="C4"
# )

# plt.xlabel("Redundant Interlocking Time")
# plt.ylabel("Shear Stress $(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
# plt.yscale("log")
# plt.yscale("log")
# plt.legend(
#     curls,
#     labelnames,
#     handler_map={
#         curls[0]: curl_image_1,
#         curls[1]: curl_image_2,
#         curls[2]: curl_image_3,
#         curls[3]: curl_image_4,
#     },
#     frameon=False,
#     handlelength=5.0,
#     labelspacing=0.0,
#     fontsize=14,
#     borderpad=0.15,
#     loc=2,
#     handletextpad=0.2,
#     borderaxespad=0.15,
# )

# plt.savefig("binding_interlocking_time.pdf", bbox_inches="tight")

