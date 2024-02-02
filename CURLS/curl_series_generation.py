
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
    filename = "curl" + curl_number

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
    simulation.scale_domain = 1.2
    simulation.auto_setup()
    simulation.cycle_count = 2 * simulation.cycle_count
    simulation.extra = "2024-01-31"
    simulation.lock_symmetry = ["false","false","false","false","false","false","true","true"]
    simulation.hasdate_in_foldername = False
    simulation.is_sbatch_high_priority = True
    simulation.sbatch_time = "1-00:00:00"
    simulation.use_liggghts_for_filling = True

    return simulation


def rod_series_simulation_specific(i):
    aspect_ratio = i
    filename = "rod" + str(aspect_ratio)

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
    simulation.scale_domain = 1.2
    simulation.auto_setup()
    simulation.cycle_count = 2 * simulation.cycle_count
    #simulation.extra = "2024-01-29"
    simulation.hasdate_in_foldername = True
    simulation.is_sbatch_high_priority = True
    simulation.sbatch_time = "1-00:00:00"
    simulation.use_liggghts_for_filling = True

    return simulation


def make_and_gen(remake_base_particle_shapes, make_vtk_files):
    equiv_diameter = 4.76E-04

    aspect_ratio = 5

    sphere_volume = (4.0 / 3.0 * math.pi *
                     (equiv_diameter / 2.0)**3) / aspect_ratio
    sphere_radius = (3.0 / 4.0 * sphere_volume / math.pi)**(1.0 / 3.0)

    sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]

    if remake_base_particle_shapes:
        # This saves curl particle shapes called curl1, curl2, curl3, etc..
        curls.quadratic_method(sphere_radius_distance,
                               sphere_radius, ["0", "1", "2", "3", "4", "5", "6"])

    for i in range(0, len(sphere_radius_distance)):
        ####################################################################################
        # Load Particle data into lebc.Particle
        simulation = curl_series_simulation_specific(str(i))
        # generate_fortran_files("extra_string_if_needed", include date in filename, is sbatch files high priority?)
        # simulation.generate_fortran_files()

        simulation.generate_liggghts_files([4, 4, 2], random_orientation=False)
        if make_vtk_files:
            simulation.particletemplate.particle.legacy_vtk_printout()
    # Include Rod 3
    simulation_rod = rod_series_simulation_specific(3)
    simulation_rod.generate_liggghts_files([4, 4, 2], random_orientation=False)
    if make_vtk_files:
        simulation_rod.particletemplate.particle.legacy_vtk_printout()

    # Include half curl particle
    if remake_base_particle_shapes:
        # This saves curl particle shapes called curl1, curl2, curl3, etc..
        curls.quadratic_method(
            [7.75, 7.5, 7.25], sphere_radius, ["0.25", "0.5", "0.75"])
     ####################################################################################
    # Load Particle data into lebc.Particle
    simulation7 = curl_series_simulation_specific("0.25")
    simulation8 = curl_series_simulation_specific("0.5")
    simulation9 = curl_series_simulation_specific("0.75")
    # generate_fortran_files("extra_string_if_needed", include date in filename, is sbatch files high priority?)
    # simulation.generate_fortran_files()

    simulation7.generate_liggghts_files([4, 4, 2], random_orientation=False)
    simulation8.generate_liggghts_files([4, 4, 2], random_orientation=False)
    simulation9.generate_liggghts_files([4, 4, 2], random_orientation=False)
    if make_vtk_files:
        simulation7.particletemplate.particle.legacy_vtk_printout()
        simulation8.particletemplate.particle.legacy_vtk_printout()
        simulation9.particletemplate.particle.legacy_vtk_printout()


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
        simulation = curl_series_simulation_specific(str(i))
        simulation.liggghts_graph_stress_vs_volume_fraction(
            already_loaded=False)  # if true it won't try and reload the data
        # if false it will reload data and plot all stress vs time graphs


def curl_series_projected_area():
    sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]
    specific_runs = [0, 1, 2, 3, 4, 5, 6]

    all_simuations = []
    for i in specific_runs:
        simulation = curl_series_simulation_specific(str(i))
        simulation.load_vf_vs_stress(use_fortran=False, use_liggghts=True)
        all_simuations.append(simulation)

    simulation1 = rod_series_simulation_specific(str(3))
    all_simuations.append(simulation1)
    all = lebc.SimulationCompare(all_simuations)
    all.effective_projected_area(use_fortran=False, use_liggghts=True,
                                 general_folder_name="Curl_Projected_Areas", series_name="curls")


# def curl_series_validate_fortran():
#     sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]
#     specific_runs = [0, 1, 2, 3, 4, 5, 6]
#     for i in specific_runs:
#         simulation = curl_series_simulation_specific(i)
#         simulation.fortran_graph_stress_vs_volume_fraction(
#             already_loaded=False)


# def curl_series_validate_all():
#     sphere_radius_distance = [8, 7, 6, 5, 4, 3, 2]
#     specific_runs = [0, 1, 2, 3, 4, 5, 6]

#     all_simuations = []
#     for i in specific_runs:
#         simulation = curl_series_simulation_specific(i)
#         # simulation.liggghts_graph_stress_vs_volume_fraction(
#         #     already_loaded=False)
#         for j in range(len(simulation.volume_fractions)-2):
#             simulation.liggghts_graph_stress_vs_time_specific(i)
#         simulation.load_vf_vs_stress()

#         all_simuations.append(simulation)

#     all = lebc.SimulationCompare(all_simuations)
#     all.stress_vs_vf_graph_compare(use_fortran=False, use_liggghts=True,
#                                    general_folder_name="Curl_Stresses", series_name="curls")
#     # all.print_lowest_volumefraction_stress()


def curls_series_validate_all():
    my_list = [0, 1, 2, 3, 4, 5, 6]

    all_simuations = []
    for i in my_list:
        aspect_ratio = i
        simulation = curl_series_simulation_specific(str(aspect_ratio))

        simulation.load_vf_vs_stress(use_fortran=False, use_liggghts=True)

        all_simuations.append(simulation)
    simulation1 = rod_series_simulation_specific(str(3))
    all_simuations.append(simulation1)
    all = lebc.SimulationCompare(all_simuations)
    all.stress_vs_vf_graph_compare(use_fortran=False, use_liggghts=True,
                                   general_folder_name="Curl_Series", series_name="curls")
    all.high_vf_box_whisker_compare(use_fortran=False, use_liggghts=True,
                                    general_folder_name="HighVolume_Box", series_name="curls", high_volume_fractions=[7])
    # all.print_lowest_volumefraction_stress()


def curl_extra_highvolume_fraction_box():
    my_list = ["0", "0.25","0.5","0.75", "1", "2", "3", "4", "5", "6"]

    all_simuations = []
    for i in my_list:
        aspect_ratio = i
        simulation = curl_series_simulation_specific(aspect_ratio)

        simulation.liggghts_graph_stress_vs_time_specific(7,"HighVolume_Box")

        all_simuations.append(simulation)

    
    all = lebc.SimulationCompare(all_simuations)
    all.high_vf_box_whisker_compare(use_fortran=False, use_liggghts=True,
                                    general_folder_name="HighVolume_Box", series_name="curls", high_volume_fractions=[0],labelnames=["Curl 0", "Curl 0.25", "Curl 0.5", "Curl 0.75", "Curl 1",
                                                                                                                                                              "Curl 2", "Curl 3", "Curl 4", "Curl 5", "Curl 6"])
    # all.print_lowest_volumefraction_stress()

def rod3_series_validate_liggghts():
    specific_runs = [3]
    for i in specific_runs:
        simulation = rod_series_simulation_specific(str(i))
        simulation.liggghts_graph_stress_vs_volume_fraction(
            already_loaded=False)  # if true it won't try and reload the data
        # if false it will reload data and plot all stress vs time graphs
def curl0_7_graph_specfic():
    simulation = curl_series_simulation_specific(str(0))
    simulation.liggghts_graph_stress_vs_time_specific(7)

########################
#make_and_gen(remake_base_particle_shapes=False, make_vtk_files=True)
########################
# Call these together
#curl_series_validate_liggghts()
#curls_series_validate_all()
#######################
#rod3_series_validate_liggghts()
#######################
#curl_extra_highvolume_fraction_box()
########################
#curl_series_projected_area()
########################
curl0_7_graph_specfic()
