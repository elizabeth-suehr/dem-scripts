
import math
import sys


def load_lebc():
    sys.path.append("..")  # Adds higher directory to python modules path.
    import generation_validation as lebc
    return lebc


lebc = load_lebc()
# All the rods have the same equivalent volume diameter


def rod_series_simulation_specific(i):
    aspect_ratio = i
    filename = "rod" + str(aspect_ratio)

    particle = lebc.Particle()
    # particle.create_multisphere(filename, density, monticarlo_count, is_particle_already_centered?)
    particle.load_or_create_multisphere(
        filename, 2500.0, 600_000_000, True)
    # particle.create_multisphere(filename, 2500.0, 600_000_000, True) #this line does same as above but forces the data to be rewritten

    print(particle)  # for debugging purposes

    ####################################################################################

    # Generate Particle Template

    youngsmod = 8.7e9
    poissonratio = 0.30
    frictioncoefficient = 0.0
    restitutioncoefficient = 0.95
    cohesion = 0.0
    yieldstress = 1.9306e30

    particle_templete = lebc.ParticleTemplate(
        particle, youngsmod, poissonratio, frictioncoefficient, restitutioncoefficient, cohesion, yieldstress)

    ####################################################################################

    # Generate Simulation

    simulation = lebc.ShearSimulation(particle_templete)
    simulation.auto_setup()

    if aspect_ratio >= 3:
        simulation.relaxationtime = 0.05

    simulation.hasdate_in_foldername = False
    simulation.is_sbatch_high_priority = False
    simulation.sbatch_time = "20-00:00:00"
    simulation.use_liggghts_for_filling = True

    return simulation


# only test rod 2, 3, 4, 5, 6
def make_and_gen():
    equiv_diameter = 4.76E-04
    for i in range(2, 7):

        ####################################################################################
        # Define Particle "rod" (not specific to test.py or lebc.py)
        aspect_ratio = i

        sphere_volume = (4.0 / 3.0 * math.pi *
                         (equiv_diameter / 2.0)**3) / aspect_ratio
        sphere_radius = (3.0 / 4.0 * sphere_volume / math.pi)**(1.0 / 3.0)

        filename = "rod" + str(i)
        file = open(filename, "w")

        if aspect_ratio == 2:
            file.write(str(sphere_radius) + " " + str(0.0) + " " + str(0.0) +
                       " " + str(sphere_radius) + "\n")
            file.write(str(-sphere_radius) + " " + str(0.0) + " " + str(0.0) +
                       " " + str(sphere_radius) + "\n")
        elif aspect_ratio % 2 == 0:
            for i in range(1, aspect_ratio // 2 + 1):
                x = sphere_radius * i * 2.0 - sphere_radius
                y = 0.0
                z = 0.0
                file.write(str(x) + " " + str(y) + " " + str(z) +
                           " " + str(sphere_radius) + "\n")
                file.write(str(-x) + " " + str(y) + " " + str(z) +
                           " " + str(sphere_radius) + "\n")
        else:
            x = 0.0
            y = 0.0
            z = 0.0
            file.write(str(x) + " " + str(y) + " " + str(z) +
                       " " + str(sphere_radius) + "\n")
            for i in range(1, aspect_ratio // 2 + 1):
                x = sphere_radius * i * 2.0
                y = 0.0
                z = 0.0
                file.write(str(x) + " " + str(y) + " " + str(z) +
                           " " + str(sphere_radius) + "\n")
                file.write(str(-x) + " " + str(y) + " " + str(z) +
                           " " + str(sphere_radius) + "\n")
        file.close()

        ####################################################################################

        # Load Particle data into lebc.Particle

        simulation = rod_series_simulation_specific(aspect_ratio)

        # generate_fortran_files("extra_string_if_needed", include date in filename, is sbatch files high priority?)

        # simulation.generate_fortran_files()

        simulation.generate_liggghts_files([4, 4, 2], random_orientation=False)


def rod_series_liggghts_init_to_fortran():
    for i in range(6, 7):
        simulation = rod_series_simulation_specific(i)

        simulation.generate_fortran_read_from_liggghts_files()


def rod_series_validate_liggghts():
    my_list = [2, 3, 4, 5, 6]
    for i in my_list:
        aspect_ratio = i
        simulation = rod_series_simulation_specific(aspect_ratio)

        simulation.liggghts_graph_stress_vs_volume_fraction(
            already_loaded=False)  # if true it won't try and reload the data
        # if false it will reload data and plot all stress vs time graphs


def rod_series_validate_fortran():

    my_list = [2, 4]
    for i in my_list:
        aspect_ratio = i
        simulation = rod_series_simulation_specific(aspect_ratio)

        simulation.fortran_graph_stress_vs_volume_fraction(
            already_loaded=False)


def rod_series_validate_all():
    my_list = [2, 3, 4, 5, 6]

    all_simuations = []
    for i in my_list:
        aspect_ratio = i
        simulation = rod_series_simulation_specific(aspect_ratio)

        simulation.graph_liggghts_vs_fortran()

        all_simuations.append(simulation)

    all = lebc.SimulationCompare(all_simuations)
    all.graph_compare(use_fortran=False, use_liggghts=True,
                      general_folder_name="Rod_Series", series_name="rods")
    all.print_lowest_volumefraction_stress()


def rod6_random_angle_simulation():
    filename = "rod6"

    particle = lebc.Particle()
    # particle.create_multisphere(filename, density, monticarlo_count, is_particle_already_centered?)
    particle.load_or_create_multisphere(filename, 2500.0, 600_000_000, True)
    # particle.create_multisphere(filename, 2500.0, 600_000_000, True) #this line does same as above but forces the data to be rewritten

    ####################################################################################

    # Generate Particle Template

    youngsmod = 8.7e9
    poissonratio = 0.30
    frictioncoefficient = 0.0
    restitutioncoefficient = 0.95
    cohesion = 0.0
    yieldstress = 1.9306e30

    particle_templete = lebc.ParticleTemplate(
        particle, youngsmod, poissonratio, frictioncoefficient, restitutioncoefficient, cohesion, yieldstress)

    ####################################################################################

    # Generate Simulation

    simulation = lebc.ShearSimulation(particle_templete)
    simulation.auto_setup()

    simulation.volume_fractions = [0.4, 0.45, 0.5]
    simulation.reset_particle_count()
    simulation.cycle_count = [200e6, 200e6, 200e6]
    simulation.stress_print_count = [30000, 30000, 30000]
    simulation.body_position_print_count = 10000
    simulation.save_count = 10000000

    # generate_fortran_files("extra_string_if_needed", include date in filename, is sbatch files high priority?)
    simulation.hasdate_in_foldername = False
    simulation.is_sbatch_high_priority = False
    simulation.sbatch_time = "20-00:00:00"
    simulation.extra = "Random_Orientation_test"
    simulation.use_liggghts_for_filling = True

    return simulation


def random_angle_liggghts_and_fortran():
    simulation = rod6_random_angle_simulation()

    simulation.generate_liggghts_files([4, 4, 2], random_orientation=True)

    # simulation.generate_fortran_files()


def create_read_files_from_liggghts():
    simulation = rod6_random_angle_simulation()
    simulation.generate_fortran_read_from_liggghts_files()


def random_angle_validate_liggghts():
    simulation = rod6_random_angle_simulation()

    simulation.liggghts_graph_stress_vs_volume_fraction(
        already_loaded=False)  # if true it won't try and reload the data
    # if false it will reload data and plot all stress vs time graphs


def random_angle_validate_fortran():
    simulation = rod6_random_angle_simulation()

    simulation.fortran_graph_stress_vs_volume_fraction(
        already_loaded=False)


# make_and_gen()
# rod_series_liggghts_init_to_fortran()
# rod_series_validate_liggghts()
# rod_series_validate_fortran()
rod_series_validate_all()

# Run This first, then after liggghts had mode files
# random_angle_liggghts_and_fortran()
# Run this second
# create_read_files_from_liggghts()
# Run this third
# random_angle_validate_liggghts()
# Run this fourth
# random_angle_validate_fortran()
