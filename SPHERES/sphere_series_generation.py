
import math
import sys


def load_lebc():
    sys.path.append("..")  # Adds higher directory to python modules path.
    import generation_validation as lebc
    return lebc


lebc = load_lebc()
# Curl Shape generator and fortran and liggghts file generator and validator
# All the curls have the same equivalent volume diameter, and have 5 spheres in the curl

# Curl 1,2,3,4,5,6 have seperation 2.5D, 2D, 1.5D, 1D, 0.5D, and 0D respectively


def sphere_series_simulation_specific():
    filename = "sphere"

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
    simulation.relaxationtime = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    simulation.cycle_count = [60e6, 40e6, 40e6, 30e6, 30e6, 30e6, 30e6, 30e6]
    # simulation.extra = "2024-02-07"
    simulation.lock_symmetry = ["false","false","false","false","false","false","false","false"]
    simulation.hasdate_in_foldername = False
    simulation.is_sbatch_high_priority = False
    simulation.sbatch_time = "2-00:00:00"
    simulation.use_liggghts_for_filling = False
    simulation.is_single_sphere = False

    return simulation


def make_and_gen(remake_base_particle_shapes, make_vtk_files):
   
    # Include Rod 3
    simulation_sphere = sphere_series_simulation_specific()
    simulation_sphere.generate_liggghts_files([4, 4, 2], random_orientation=False)
    if make_vtk_files:
        simulation_sphere.particletemplate.particle.legacy_vtk_printout()


def sphere_series_validate_liggghts():
    simulation = sphere_series_simulation_specific()
    simulation.liggghts_graph_stress_vs_volume_fraction(
        already_loaded=False)  # if true it won't try and reload the data
        # if false it will reload data and plot all stress vs time graphs


########################
make_and_gen(remake_base_particle_shapes=False, make_vtk_files=True)
########################
# sphere_series_validate_liggghts()

