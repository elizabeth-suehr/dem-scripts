
import math
import sys
import generate_curls as curls


def load_lebc():
    sys.path.append("..")  # Adds higher directory to python modules path.
    import generation_validation as lebc
    return lebc


lebc = load_lebc()
# Curl Shape generator and fortran and liggghts file generator and validator
# All the curls have the same equivalent volume diameter, and have 5 spheres in the curl

# Curl 1,2,3,4,5,6 have seperation 2.5D, 2D, 1.5D, 1D, 0.5D, and 0D respectively


def curl_series_simulation_specific(i):
    curl_number = i
    filename = "curl" + str(curl_number)

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
    simulation.auto_setup()

    simulation.relaxationtime = 0.05

    simulation.hasdate_in_foldername = False
    simulation.is_sbatch_high_priority = False
    simulation.sbatch_time = "20-00:00:00"
    simulation.use_liggghts_for_filling = True

    return simulation


def make_and_gen(remake_base_particle_shapes):
    equiv_diameter = 4.76E-04

    aspect_ratio = 5

    sphere_volume = (4.0 / 3.0 * math.pi *
                     (equiv_diameter / 2.0)**3) / aspect_ratio
    sphere_radius = (3.0 / 4.0 * sphere_volume / math.pi)**(1.0 / 3.0)

    sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]

    if remake_base_particle_shapes:
        # This saves curl particle shapes called curl1, curl2, curl3, etc..
        curls.quadratic_method(sphere_radius_distance, sphere_radius)

    for i in range(0, len(sphere_radius_distance)):
        ####################################################################################
        # Load Particle data into lebc.Particle
        simulation = curl_series_simulation_specific(i)
        # generate_fortran_files("extra_string_if_needed", include date in filename, is sbatch files high priority?)
        # simulation.generate_fortran_files()
        simulation.auto_setup()
        simulation.generate_liggghts_files([4, 4, 2], random_orientation=False)


def curl_series_liggghts_init_to_fortran():
    sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]
    specific_runs = [0, 1, 2, 3, 4, 5, 6]
    for i in specific_runs:
        simulation = curl_series_simulation_specific(i)

        simulation.generate_fortran_read_from_liggghts_files()


def curl_series_validate_liggghts():
    sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]
    specific_runs = [0, 1, 2, 3, 4, 5, 6]
    for i in specific_runs:
        simulation = curl_series_simulation_specific(i)
        simulation.liggghts_graph_stress_vs_volume_fraction(
            already_loaded=False)  # if true it won't try and reload the data
        # if false it will reload data and plot all stress vs time graphs


def curl_series_validate_fortran():
    sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]
    specific_runs = [0, 1, 2, 3, 4, 5, 6]
    for i in specific_runs:
        simulation = curl_series_simulation_specific(i)
        simulation.fortran_graph_stress_vs_volume_fraction(
            already_loaded=False)


def curl_series_validate_all():
    sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]
    specific_runs = [0, 1, 2, 3, 4, 5, 6]
    for i in specific_runs:
        simulation = curl_series_simulation_specific(i)
        simulation.graph_liggghts_vs_fortran()


make_and_gen(remake_base_particle_shapes=False)
# curl_series_liggghts_init_to_fortran()
# curl_series_validate_liggghts()
# curl_series_validate_fortran()
# curl_series_validate_all()
