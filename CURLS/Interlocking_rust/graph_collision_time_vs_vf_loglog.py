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


directory_1 = ["curl_1/", "curl_2/", "curl_3/", "curl_4/"]
directory_2 = ["vf_2.5/", "vf_5/", "vf_10/", "vf_20/", "vf_30/","vf_40/", "vf_45/"] #  "vf_40/",
particle_count = [83, 166, 332, 664, 996, 1328, 1494]

volume_fraction = [0.025, 0.05, 0.10, 0.20, 0.30, 0.4, 0.45]

collision_counts = np.empty((6, 7, 1800))
collision_averages = np.empty((6, 7))
collision_percentages = np.empty((6, 7))

for i in range(len(directory_1)):

    for j in range(len(directory_2)):
        k = 0

        filename = (
            "collision_check/"+
             directory_1[i] + directory_2[j] + "collision.txt"
        )
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
        


for i in range(len(directory_1)):
    for j in range(len(directory_2)):
        collision_averages[i][j] = sum(collision_counts[i][j][:]) / len(
            collision_counts[i][j]
        )
        collision_percentages[i][j] = (
            sum(collision_counts[i][j][:])
            / len(collision_counts[i][j][:])
            / particle_count[j]
            * 100
        )


matplotlib.rcParams.update({"font.size": 18})

plt.figure(1)
marks = [ "v", "x", "s", "*", "D"]
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']
labelnames = ["Curl 1", "Curl 2", "Curl 3", "Curl 4"]
curls = [object] * 4
for i in range(len(directory_1)):
    (curls[i],) = plt.plot(
        volume_fraction, collision_averages[i][:], "--", marker=marks[i], color=colors[i]
    )


# x = np.linspace(0.025, 0.3, 100)
# y = 2500.0 * x**2.25
# plt.plot(x, y)


#plt.yscale("log")
#plt.xscale("log")
plt.xlabel("Solid volume fraction (ν)")
plt.ylabel("Normalized Interlocking Time")
plt.legend(
    curls,
    labelnames,
    handler_map={
        curls[0]: curl_image_1,
        curls[1]: curl_image_2,
        curls[2]: curl_image_3,
        curls[3]: curl_image_4,
        # curls[4]: curl_image_4,
        # curls[5]: curl_image_5,
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


plt.savefig("interlocking_time_vs_vf.pdf", bbox_inches="tight")


plt.figure(2)
marks = ["o", "v", "x", "s", "*", "D", "+"]
labelnames = ["Curl 1", "Curl 2", "Curl 3", "Curl 4"]
curls = [object] * 4
for i in range(len(directory_1)):
    (curls[i],) = plt.plot(
        volume_fraction, collision_percentages[i][:], "--", marker=marks[i], color=colors[i]
    )


plt.xlabel("Solid volume fraction (ν)")
plt.ylabel("Interlocked Percentage")
plt.legend(
    curls,
    labelnames,
    handler_map={
        curls[0]: curl_image_1,
        curls[1]: curl_image_2,
        curls[2]: curl_image_3,
        curls[3]: curl_image_4,
        # curls[4]: curl_image_4,
        # curls[5]: curl_image_5,
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


plt.savefig("percentage_interlocked_vs_vf_log_log.pdf", bbox_inches="tight")
# for i in range(len(directory_1)):
#     plt.plot(volume_fraction, stress_average_xy[i][:]/collision_averages[i][:], label=directory_1[i])

# plt.xlabel("Volume Fraction")
# plt.ylabel("Average stress/ average interlocking")
# plt.legend()
# plt.show()
