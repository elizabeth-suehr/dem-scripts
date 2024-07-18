import matplotlib.pyplot as plt
import os

def high_vf_box_whisker_compare(self, use_fortran=True, use_liggghts=True, general_folder_name="", series_name="", high_volume_fractions=[7], labelnames=["Curl 0", "Curl 1",
                                                                                                                                                              "Curl 2", "Curl 3", "Curl 4", "Curl 5", "Curl 6"]):
        if general_folder_name == "":
            general_folder_name = "high_volume_box_" + \
                self.simulations[0].root_folder_name
        if series_name == "":
            series_name = "high_volume_box_" + \
                self.simulations[0].particletemplate.particle.file_shape_name

        try:
            os.makedirs(general_folder_name)
        except OSError:
            # print(OSError)
            print("General folder name for graph already exists")

        # Now that all the data is loaded, we can graph the stress vs volume fraction in on nice
        plt.figure(1)
        fig, ax = plt.subplots()

        markers = ["o", "v", "x", "s", "*", "D", "+"]
        colors = ["#e05252",
                  "#e1893f", "#D6B11F", "#91b851", "#52a0e0",  "#1F73B8", "#7768ae"]
        colors = ['Red', 'Orange', 'Pink',
                  'Green', 'LightBlue', 'Blue', "Purple"]

        plots = []
        data = []
        for (i, simulation) in enumerate(self.simulations):
            plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("Curl #")
            if use_liggghts:

                for j in range(min(high_volume_fractions), max(high_volume_fractions)+1):
                    n_length = int(len(simulation.l_normal_stress_vs_time[j]) *
                                   simulation.stress_vs_time_cutoff_range[j])
                    data.append(
                        simulation.l_normal_stress_vs_time[j][n_length:])

        my_dict = dict(zip(labelnames, data))
        ax.boxplot(my_dict.values())
        ax.set_xticklabels(my_dict.keys(), fontsize='small')
        ax.set_yscale('log')

        plt.savefig(general_folder_name +
                    "/normal_box_{}.pdf".format(series_name), dpi=250)
        plt.clf()

        plt.figure(1)
        fig, ax = plt.subplots()
        data = []
        for (i, simulation) in enumerate(self.simulations):
            plt.ylabel("$(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")

            if use_liggghts:
                for j in range(min(high_volume_fractions), max(high_volume_fractions)+1):
                    s_length = int(len(simulation.l_shear_stress_vs_time[j]) *
                                   simulation.stress_vs_time_cutoff_range[j])
                    data.append(
                        simulation.l_shear_stress_vs_time[j][s_length:])

        my_dict = dict(zip(labelnames, data))
        ax.boxplot(my_dict.values())
        ax.set_xticklabels(my_dict.keys())
        ax.set_yscale('log')

        plt.savefig(general_folder_name + "/shear_box_{}.pdf".format(
            series_name), dpi=250)
        plt.clf()




Movement= [8.73551377662823e-05, 0.00013145826189718982, 0.0001373785533476233, 0.0001722214202453617, 0.00015193227433745493, 0.0001229220671106221, 8.083287718513078e-05]
Stress= [30.723755932321676, 9481.292618203433, 19746.43490927875, 23531.313195592356, 3443.4974723526093, 655.0004058358264, 22.20843159052716]