
import math
import sys


def load_lebc():
    sys.path.append("..")  # Adds higher directory to python modules path.
    import generation_validation as lebc
    return lebc


lebc = load_lebc()


FILENAMES = ["da15_dc15_h20","da20_dc20_h20","da30_dc30_h20","da36_dc36_h20","da45_dc45_h20", "fib21", "fib33", "fib55", "fib100"]

def asperities_series_simulation_specific(i):
    filename = FILENAMES[i]

    particle = lebc.Particle()
    # particle.create_multisphere(filename, density, monticarlo_count, is_particle_already_centered?)
    particle.load_or_create_multisphere(
        filename, 2500.0, 600_000_000, is_centered=False, needs_rotated=False, is_point_mass=True, delta_cutoff=1e-8)
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
    simulation.scale_domain = 1.0
    simulation.setup_asperities()
    simulation.body_position_print_count = [10000,10000]
    simulation.relaxationtime = [0.005,0.005]
    simulation.cycle_count = [50e6, 50e6]
    simulation.extra = "2025-03-29"
    simulation.hasdate_in_foldername = False
    simulation.is_sbatch_high_priority = False
    simulation.sbatch_time = "7-00:00:00"
    simulation.use_liggghts_for_filling = True

    return simulation



# only test rod 2, 3, 4, 5, 6
def make_and_gen():
    for i in range(0, 9):

        ####################################################################################

        # Load Particle data into lebc.Particle

        simulation = asperities_series_simulation_specific(i)

        # generate_fortran_files("extra_string_if_needed", include date in filename, is sbatch files high priority?)

        simulation.generate_fortran_files()

        simulation.generate_liggghts_files([4, 4, 2], random_orientation=False)


def asperities_series_liggghts_init_to_fortran():
    for i in range(0, 9):
        simulation = asperities_series_simulation_specific(i)

        simulation.generate_fortran_read_from_liggghts_files()


def asperities_series_validate_liggghts():
    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in my_list:
        simulation = asperities_series_simulation_specific(i)

        simulation.liggghts_graph_stress_vs_volume_fraction(
            already_loaded=False)  # if true it won't try and reload the data
        # if false it will reload data and plot all stress vs time graphs


def asperities_series_validate_fortran():

    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in my_list:
        aspect_ratio = i
        simulation = asperities_series_simulation_specific(aspect_ratio)

        simulation.fortran_graph_stress_vs_volume_fraction(
            already_loaded=False)


def asperities_series_validate_all():
    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    all_simuations = []
    for i in my_list:
        aspect_ratio = i
        simulation = asperities_series_simulation_specific(aspect_ratio)

        #simulation.graph_liggghts_vs_fortran()

        all_simuations.append(simulation)

    all = lebc.SimulationCompare(all_simuations)
    all.stress_vs_vf_graph_compare(use_fortran=False, use_liggghts=True,
                                   general_folder_name="Asperities_Series", series_name="asperities")
    #all.print_lowest_volumefraction_stress()


# def create_read_files_from_liggghts():
#     simulation.generate_fortran_read_from_liggghts_files()


# def random_angle_validate_liggghts():

#     simulation.liggghts_graph_stress_vs_volume_fraction(
#         already_loaded=False)  # if true it won't try and reload the data
#     # if false it will reload data and plot all stress vs time graphs


# def random_angle_validate_fortran():
#     simulation.fortran_graph_stress_vs_volume_fraction(
#         already_loaded=False)


make_and_gen()
# asperities_series_liggghts_init_to_fortran()
# asperities_series_validate_liggghts()
# asperities_series_validate_fortran()
#asperities_series_validate_all()

# Run This first, then after liggghts had mode files
# random_angle_liggghts_and_fortran()
# Run this second
# create_read_files_from_liggghts()
# Run this third
# random_angle_validate_liggghts()
# Run this fourth
# random_angle_validate_fortran()
