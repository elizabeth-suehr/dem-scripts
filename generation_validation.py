import itertools
import math
import random
import numpy as np
from numba import jit
import os
from datetime import date
import fortranformat as ff
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
from matplotlib import rc, rcParams


###############################################################################
# Random math functions

# Taken from LIGGGHTS, guessing was in LAMMPS


def exyz_to_q(ex, ey, ez):
    # squares of quaternion components

    q0sq = 0.25 * (ex[0] + ey[1] + ez[2] + 1.0)
    q1sq = q0sq - 0.5 * (ey[1] + ez[2])
    q2sq = q0sq - 0.5 * (ex[0] + ez[2])
    q3sq = q0sq - 0.5 * (ex[0] + ey[1])

    # some component must be greater than 1/4 since they sum to 1
    # compute other components from it

    q = [0.0, 0.0, 0.0, 0.0]

    if q0sq >= 0.25:
        q[0] = math.sqrt(q0sq)
        q[1] = (ey[2] - ez[1]) / (4.0 * q[0])
        q[2] = (ez[0] - ex[2]) / (4.0 * q[0])
        q[3] = (ex[1] - ey[0]) / (4.0 * q[0])
    elif q1sq >= 0.25:
        q[1] = math.sqrt(q1sq)
        q[0] = (ey[2] - ez[1]) / (4.0 * q[1])
        q[2] = (ey[0] + ex[1]) / (4.0 * q[1])
        q[3] = (ex[2] + ez[0]) / (4.0 * q[1])
    elif q2sq >= 0.25:
        q[2] = math.sqrt(q2sq)
        q[0] = (ez[0] - ex[2]) / (4.0 * q[2])
        q[1] = (ey[0] + ex[1]) / (4.0 * q[2])
        q[3] = (ez[1] + ey[2]) / (4.0 * q[2])
    elif q3sq >= 0.25:
        q[3] = math.sqrt(q3sq)
        q[0] = (ex[1] - ey[0]) / (4.0 * q[3])
        q[1] = (ez[0] + ex[2]) / (4.0 * q[3])
        q[2] = (ez[1] + ey[2]) / (4.0 * q[3])

    quat = quat.normalized()
    return quat


def quaternion_mult(q, r):
    return [
        r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
        r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
        r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
        r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0],
    ]


def point_rotation_by_quaternion(point, quat):
    r = [0, point[0], point[1], point[2]]
    q = [quat[0], quat[1], quat[2], quat[3]]
    q_conj = [quat[0], -1 * quat[1], -1 * quat[2], -1 * quat[3]]
    return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]


@jit(nopython=True)
def calcualte_volume(x, y, z, r, aabb, repeat_count):
    length = [
        aabb[1][0] - aabb[0][0],
        aabb[1][1] - aabb[0][1],
        aabb[1][2] - aabb[0][2],
    ]
    center_of_mass = [0.0, 0.0, 0.0]
    positions_in_particle = 0
    for i in range(repeat_count):

        position = [
            aabb[0][0] + length[0] * random.random(),
            aabb[0][1] + length[1] * random.random(),
            aabb[0][2] + length[2] * random.random(),
        ]

        for j in range(len(r)):
            distance = math.sqrt(
                (position[0] - x[j]) ** 2
                + (position[1] - y[j]) ** 2
                + (position[2] - z[j]) ** 2
            )
            if distance < r[j]:
                positions_in_particle += 1

                center_of_mass[0] += position[0]
                center_of_mass[1] += position[1]
                center_of_mass[2] += position[2]
                break

    center_of_mass[0] /= float(positions_in_particle)
    center_of_mass[1] /= float(positions_in_particle)
    center_of_mass[2] /= float(positions_in_particle)

    volume = (
        length[0]
        * length[1]
        * length[2]
        * float(positions_in_particle / repeat_count)
    )
    return volume, center_of_mass


@jit(nopython=True)
def calculate_inertia_tensor(x, y, z, r, aabb, mass, repeat_count):
    positions_in_particle = 0
    length = [
        aabb[1][0] - aabb[0][0],
        aabb[1][1] - aabb[0][1],
        aabb[1][2] - aabb[0][2],
    ]

    inertia_tensor = np.zeros((3, 3), dtype=np.float64)
    for i in range(repeat_count):

        position = [
            aabb[0][0] + length[0] * random.random(),
            aabb[0][1] + length[1] * random.random(),
            aabb[0][2] + length[2] * random.random(),
        ]

        for j in range(len(r)):
            distance = math.sqrt(
                (position[0] - x[j]) ** 2
                + (position[1] - y[j]) ** 2
                + (position[2] - z[j]) ** 2
            )
            if distance < r[j]:
                positions_in_particle += 1

                inertia_tensor[0][0] += mass * (
                    position[1] ** 2 + position[2] ** 2
                )
                inertia_tensor[1][1] += mass * (
                    position[0] ** 2 + position[2] ** 2
                )
                inertia_tensor[2][2] += mass * (
                    position[0] ** 2 + position[1] ** 2
                )
                inertia_tensor[0][1] -= mass * \
                    position[0] * position[1]
                inertia_tensor[1][0] -= mass * \
                    position[0] * position[1]
                inertia_tensor[1][2] -= mass * \
                    position[1] * position[2]
                inertia_tensor[2][1] -= mass * \
                    position[1] * position[2]
                inertia_tensor[2][0] -= mass * \
                    position[0] * position[2]
                inertia_tensor[0][2] -= mass * \
                    position[0] * position[2]
                break

    inertia_tensor[0][0] /= positions_in_particle
    inertia_tensor[1][1] /= positions_in_particle
    inertia_tensor[2][2] /= positions_in_particle
    inertia_tensor[0][1] /= positions_in_particle
    inertia_tensor[1][0] /= positions_in_particle
    inertia_tensor[1][2] /= positions_in_particle
    inertia_tensor[2][1] /= positions_in_particle
    inertia_tensor[2][0] /= positions_in_particle
    inertia_tensor[0][2] /= positions_in_particle

    return inertia_tensor
###############################################################################
# Particle Class and functions


class Particle(object):

    def __init__(self):

        self.file_shape_name = ""
        self.file_data_name = ""

        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)
        self.z = np.array([], dtype=np.float64)
        self.r = np.array([], dtype=np.float64)

        self.max_radius = 0
        self.min_radius = 1e10

        self.inertia_tensor = np.zeros((3, 3), dtype=np.float64)
        self.volume = 0
        self.equvi_diameter = 0
        self.mass = 0
        self.density = 2500.0
        self.aabb = np.zeros((2, 3), dtype=np.float64)

    def __str__(self):
        # returnstring = ""  # "Type: " + str(self.type)
        returnstring = "Volume: " + str(self.volume)
        returnstring += "\nDensity: " + str(self.density)
        returnstring += "\nMass: " + str(self.mass)
        returnstring += "\nequvi_diameter: " + str(self.equvi_diameter)

        returnstring += "\nInertia_tensor: \n " + str(self.inertia_tensor)
        returnstring += "\nPositions: \n"
        for i in range(len(self.r)):
            returnstring += (
                str(self.x[i])
                + " "
                + str(self.y[i])
                + " "
                + str(self.z[i])
                + " "
                + str(self.r[i])
                + "\n"
            )
        return returnstring

    def legacy_vtk_printout(self):
        # vtk DataFile Version 2.0
        try:
            file = open('legacy_'+self.file_data_name+'.vtk', "w")
        except:
            print("[Error] can not create " +
                  'legacy_'+self.file_data_name+'.vtk')
            return False

        with file:

            # Header for vtk file
            file.write('# vtk DataFile Version 2.0\n')
            file.write(
                'Unstructured grid legacy vtk file with point scalar data\nASCII\n\nDATASET UNSTRUCTURED_GRID\n')

            # str here it total number of data points
            file.write('POINTS ' + str(len(self.r)) + ' double\n')

            # Writes out Each Non Zero Sphere position
            for (x, y, z) in zip(self.x, self.y, self.z):
                file.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

            # Header for Sphere Radius
            file.write('\nPOINT_DATA ' + str(len(self.r)) +
                       '\nSCALARS radii double\nLOOKUP_TABLE default\n')

            # Writes out Each Non Zero Sphere Radius
            for r in self.r:
                file.write("{0}\n".format(r))

            # End of Script
            file.close()

    def load_or_create_multisphere(self, filename, density=2500.0, repeat_count=200_000_000, is_centered=False, needs_rotated=False, is_point_mass=False, delta_cutoff=1e-8):
        # self.type = ParticleType.MULTISPHERE
        if not self.load_multisphere(filename):
            self.create_multisphere(
                filename, density, repeat_count, is_centered, needs_rotated, is_point_mass, delta_cutoff)

    def load_multisphere(self, filename):
        self.file_data_name = filename + "_data"
        self.file_shape_name = filename

        try:
            file = open(self.file_data_name, "r")
        except:
            print("[Error] load_multisphere: file not found")
            return False

        with file:
            self.volume = float(file.readline().split()[1])
            self.density = float(file.readline().split()[1])
            self.mass = float(file.readline().split()[1])
            self.equvi_diameter = float(file.readline().split()[1])

            if file.readline().split()[0] == "Inertia_tensor:":
                for i in range(3):
                    line = file.readline().replace(
                        "[", "").replace("]", "").split()
                    for j in range(3):
                        self.inertia_tensor[i][j] = float(line[j])

            x = []
            y = []
            z = []
            r = []

            if file.readline().split()[0] == "Positions:":
                for line in file.readlines():
                    line = line.split()
                    x.append(float(line[0]))
                    y.append(float(line[1]))
                    z.append(float(line[2]))
                    r.append(float(line[3]))

            self.x = np.array(x, dtype=np.float64)
            self.y = np.array(y, dtype=np.float64)
            self.z = np.array(z, dtype=np.float64)
            self.r = np.array(r, dtype=np.float64)

            if len(self.x) == 0:
                print("[Error] load_multisphere: file is empty")
                return False

            self.calculate_aabb()
            self.set_min_max_radius()
            return True

    def create_multisphere(self, filename, density=2500.0, repeat_count=200_000_000, is_centered=False, needs_rotated=False, is_point_mass=False, delta_cutoff=1e-8):
        print("Creating multisphere " + filename)
        self.density = density
        self.file_data_name = filename + "_data"
        self.file_shape_name = filename
        try:
            file = open(filename, "r")
        except:
            print("[Error] create_multisphere: file not found")
            return False

        x = []
        y = []
        z = []
        r = []

        for line in file.readlines():
            line = line.split()
            x.append(float(line[0]))
            y.append(float(line[1]))
            z.append(float(line[2]))
            r.append(float(line[3]))

        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.z = np.array(z, dtype=np.float64)
        self.r = np.array(r, dtype=np.float64)

        if len(self.x) == 0:
            print("[Error] create_multisphere: file is empty")
            return False

        self.calculate_aabb()
        self.calcualte_volume_and_interia_tensor(
            repeat_count, self.density,  is_centered, needs_rotated, is_point_mass, delta_cutoff)
        if needs_rotated:
            self.rotate_to_principal_axes_of_inertia()

        self.set_min_max_radius()

        self.save_multisphere()
        return True

    def save_multisphere(self):
        if self.file_data_name == "":
            print("[Error] save_multisphere: file name not set")
            return False

        file = open(self.file_data_name, "w")

        file.write(str(self))

    def resize(self, size):
        for i in range(len(self.r)):
            self.x[i] *= size
            self.y[i] *= size
            self.z[i] *= size
            self.r[i] *= size
        self.set_min_max_radius()

    def set_min_max_radius(self):
        self.max_radius = 0
        self.min_radius = 1e10
        for i in range(len(self.r)):
            self.max_radius = max(self.max_radius, self.r[i])
            self.min_radius = min(self.min_radius, self.r[i])

    def subract_transform_sphere_positions(self, center):
        for i in range(len(self.r)):
            self.x[i] -= center[0]
            self.y[i] -= center[1]
            self.z[i] -= center[2]

    def add_sphere(self, x, y, z, radius):
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.r.append(radius)
        self.set_min_max_radius()

    # axial aligned bounding box, volume in which we generate random points for all our calculations

    def calculate_aabb(self):

        self.aabb[0][0] = self.x[0]
        self.aabb[0][1] = self.y[0]
        self.aabb[0][2] = self.z[0]
        self.aabb[1][0] = self.x[0]
        self.aabb[1][1] = self.y[0]
        self.aabb[1][2] = self.z[0]

        for i in range(len(self.r)):
            self.aabb[0][0] = min(self.aabb[0][0], self.x[i] - self.r[i])
            self.aabb[0][1] = min(self.aabb[0][1], self.y[i] - self.r[i])
            self.aabb[0][2] = min(self.aabb[0][2], self.z[i] - self.r[i])
            self.aabb[1][0] = max(
                self.aabb[1][0], self.x[i] + self.r[i])
            self.aabb[1][1] = max(
                self.aabb[1][1], self.y[i] + self.r[i])
            self.aabb[1][2] = max(
                self.aabb[1][2], self.z[i] + self.r[i])

    def calcualte_volume_and_interia_tensor(self, repeat_count, density,  is_centered=False, needs_rotated=False, is_point_mass=False, delta_cutoff=1e-8):

        volume, center_of_mass = calcualte_volume(
            self.x, self.y, self.z, self.r, self.aabb, repeat_count)

        self.volume = volume
        if not is_centered:
            if is_point_mass:
                center_of_mass = np.array(
                    [np.mean(self.x), np.mean(self.y), np.mean(self.z)])
                if abs(center_of_mass[0]) < delta_cutoff:
                    center_of_mass[0] = 0.0
                if abs(center_of_mass[1]) < delta_cutoff:
                    center_of_mass[1] = 0.0
                if abs(center_of_mass[2]) < delta_cutoff:
                    center_of_mass[2] = 0.0
            for (x, y, z) in zip(self.x, self.y, self.z):
                if abs(x) < delta_cutoff:
                    x = 0.0
                if abs(y) < delta_cutoff:
                    y = 0.0
                if abs(z) < delta_cutoff:
                    z = 0.0

            self.subract_transform_sphere_positions(center_of_mass)

        self.calculate_aabb()

        self.equvi_diameter = 2.0 * \
            (3.0 / 4.0 * self.volume / math.pi)**(1.0 / 3.0)

        self.mass = self.volume * density

        self.inertia_tensor = calculate_inertia_tensor(
            self.x, self.y, self.z, self.r, self.aabb, self.mass, repeat_count)

    def rotate_to_principal_axes_of_inertia(self):
        # calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.inertia_tensor)

        # sort eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # calculate rotation matrix
        rotation_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                rotation_matrix[i][j] = eigenvectors[j][i]

        ez = np.cross(rotation_matrix[0], rotation_matrix[1])
        dot = np.dot(ez, rotation_matrix[2])
        if dot < 0.0:
            rotation_matrix[2] = rotation_matrix[2] * -1
        # rotate particle
        for i in range(len(self.r)):
            position = [self.x[i], self.y[i], self.z[i]]
            position = np.matmul(rotation_matrix, position)
            self.x[i] = position[0]
            self.y[i] = position[1]
            self.z[i] = position[2]


class ParticleTemplate(object):
    particle: Particle

    def __init__(self, particle, ym, pr, fc, rc, co, ys):
        self.particle = particle
        self.youngsmod = ym
        self.poissonratio = pr
        self.frictioncoefficient = fc
        self.restitutioncoefficient = rc
        self.cohesion = co
        self.yieldstress = ys


###############################################################################
# Shear Simulation Class and functions

class ShearSimulation(object):
    particletemplate: ParticleTemplate

    def __init__(self, particletemplate):

        self.extra = ""
        self.hasdate_in_foldername = False
        self.is_sbatch_high_priority = False
        self.sbatch_time = "5-00:00:00"
        self.use_liggghts_for_filling = True

        self.root_folder_name = ""
        self.volume_fractions = []
        self.particle_count = []
        self.particletemplate = particletemplate

        self.scale_domain = 1.0
        self.domain = np.zeros((2, 3), dtype=np.float64)
        self.domain_volume = 0.0
        self.shearstrainrate = 0.0
        self.gravity = np.zeros(3, dtype=np.float64)
        self.relaxationtime = []
        self.delta_time = 0.0
        self.cycle_count = []
        self.cycle_delay = []
        self.stress_print_count = []
        self.save_count = 0
        self.body_position_print_count = []
        self.lock_symmetry = []
        self.is_single_sphere = False

        # Validation Variables
        self.liggghts_loaded_volume_fraction = []
        self.liggghts_kinetic_normal_stress = []
        self.liggghts_kinetic_shear_stress = []
        self.liggghts_collisional_normal_stress = []
        self.liggghts_collisional_shear_stress = []

        self.liggghts_middle50quartile_last_third_normal = []
        self.liggghts_middle50quartile_last_third_shear = []

        self.liggghts_time = []
        self.fortran_loaded_volume_fraction = []
        self.fortran_kinetic_normal_stress = []
        self.fortran_kinetic_shear_stress = []
        self.fortran_collisional_normal_stress = []
        self.fortran_collisional_shear_stress = []
        self.fortran_time = []

        self.l_normal_stress_ave = []
        self.l_normal_stress_vs_time = []
        self.l_shear_stress_ave = []
        self.l_shear_stress_vs_time = []
        self.l_volume_fractions = []

        self.f_normal_stress_ave = []
        self.f_shear_stress_ave = []
        self.f_volume_fractions = []
        self.stress_vs_time_cutoff_range = []

        self.quant_range = [0.0, 1.0]


    def __str__(self):
        return "No finished yet"

    def auto_delta_time(self):
        nu = self.particletemplate.poissonratio
        E = self.particletemplate.youngsmod
        dens = self.particletemplate.particle.density
        min_radius = self.particletemplate.particle.min_radius

        alphap = 0.1631 * nu + 0.876605
        shear_mod = E / (2. * (1. + nu))
        # using min radius
        dt = np.pi * min_radius * \
            np.sqrt(np.min(dens) / shear_mod) / alphap

        self.delta_time = dt

    def auto_setup(self):
        self.root_folder_name = "s_" + self.particletemplate.particle.file_shape_name

        self.shearstrainrate = 100.0
        self.volume_fractions = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]

        self.relaxationtime = [0.025,0.025,0.025,0.025,0.0125,0.0125,0.0125,0.0125]

        self.cycle_count = [60e6, 40e6, 40e6, 30e6, 30e6, 30e6, 30e6, 30e6]
        self.cycle_delay = [5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6]
        self.stress_print_count = [100000, 100000,
                                   50000, 50000, 100000, 60000, 60000, 60000]
        self.stress_vs_time_cutoff_range = [0.8, 0.7, 0.6, 0.5, 0.7, 0.7, 0.7, 0.7]
        self.body_position_print_count = [5000,5000,5000,5000,10000,10000,10000,10000]
        self.lock_symmetry = ["false","false","false","false","false","false","false","false"]
        self.save_count = 5000

        # Uses equvialent volume diameter to size the domain
        e_v_d = self.particletemplate.particle.equvi_diameter * self.scale_domain

        self.domain[0][0] = 0.0
        self.domain[0][1] = 0.0
        self.domain[0][2] = 0.0
        self.domain[1][0] = e_v_d * 15.12
        self.domain[1][1] = e_v_d * 15.12
        self.domain[1][2] = e_v_d * 15.12 * 0.5

        self.domain_volume = (
            self.domain[1][0] - self.domain[0][0]) * (self.domain[1][1] - self.domain[0][1]) * (self.domain[1][2] - self.domain[0][2])
        self.reset_particle_count()

        self.auto_delta_time()

    def reset_particle_count(self):
        self.particle_count.clear()
        for vf in self.volume_fractions:
            self.particle_count.append(
                int(self.domain_volume * vf / self.particletemplate.particle.volume))

    def generate_fortran_files(self):
        root_name = "fortran_" + self.root_folder_name

        if self.hasdate_in_foldername:
            root_name += "_" + str(date.today())
        if self.extra != "":
            root_name += "_" + self.extra

        try:
            os.makedirs(root_name)
        except OSError:
            # print(OSError)
            print("generate_fortran_files: root folder already exists")

        for i in range(len(self.volume_fractions)):
            path = root_name + "/vf_" + str(self.volume_fractions[i])
            try:
                os.makedirs(path)
                os.makedirs(path + "/vtp_files")
                os.makedirs(path + "/cpi_files")
            except OSError:
                # print(OSError)
                print("generate_fortran_files: folder already exists")

            self.generate_fortran_init_file(
                path, i)
            self.generate_fortran_sbatch_file(
                path, i)
        if not self.use_liggghts_for_filling:
            if np.max(np.array(self.volume_fractions)) > 0.21:

                path = root_name + "/filling"
                try:
                    os.makedirs(path)
                    os.makedirs(path + "/vtp_files")
                    os.makedirs(path + "/cpi_files")
                except OSError:
                    # print(OSError)
                    print("generate_fortran_files: folder already exists")
                self.generate_fortran_filling_file(
                    path, np.max(np.array(self.volume_fractions)))
                self.generate_fortran_sbatch_file(
                    path, i, self.is_sbatch_high_priority, self.sbatch_time)

        file_low = open(root_name + "/start_low.sh", "w")
        file_low.write("#!/bin/bash\n")
        file_high = open(root_name + "/start_high.sh", "w")
        file_high.write("#!/bin/bash\n")
        for i in range(len(self.volume_fractions)):
            if self.volume_fractions[i] <= 0.21:
                file_low.write(
                    "cd vf_{}/\n".format(self.volume_fractions[i]))
                file_low.write("sbatch start\n")
                file_low.write("cd ..\n")
            else:
                file_high.write(
                    "cd vf_{}/\n".format(self.volume_fractions[i]))
                file_high.write("sbatch start\n".format(
                    str(self.volume_fractions[i])))
                file_high.write("cd ..\n")
        file_low.close()
        file_high.close()

    def generate_fortran_init_file(self, path, i):
        volume_fraction = self.volume_fractions[i]
        particle_count = self.particle_count[i]
        cyc_count = self.cycle_count[i]
        stress_count = self.stress_print_count[i]

        file = open(path + "/initialize", "w")
        if volume_fraction > 0.21 and not self.use_liggghts_for_filling:
            file.write("restart VF{}\n".format(int(volume_fraction * 100)))

        else:
            # TODO automate additional start settings (need to look at documents)
            file.write("start {:.6f} {:.6f} {:.6f} 500 150000 15000 1000 2 5 log\n\n".format(
                self.domain[1][0], self.domain[1][1], self.domain[1][2]))

            radii = self.particletemplate.particle.r

            unique_radii = np.unique(radii)

            for j, radius in enumerate(unique_radii):
                file.write("dia {:.4e} {}\n".format(radius * 2.0, j + 1))
                file.write("ymd {:.4e} {}\n".format(
                    self.particletemplate.youngsmod, j + 1))
                file.write("prat {:.4e} {}\n".format(
                    self.particletemplate.poissonratio, j + 1))
                file.write("dens {:.4e} {}\n".format(
                    self.particletemplate.particle.density, j + 1))
                file.write("fric {:.4e} {}\n".format(
                    self.particletemplate.frictioncoefficient, j + 1))
                file.write("yie {:.4e} {}\n".format(
                    self.particletemplate.yieldstress, j + 1))
                file.write("\n")

        # Restitution Coefficient to silly fortran damping command
        e = self.particletemplate.restitutioncoefficient
        beta = -math.log(e) / math.sqrt(math.log(e)**2 + math.pi ** 2)
        file.write("damp {:.4f} {:.4f} 1 0 1\n".format(
            (1 - e), (1 - e) / (2 * math.pi * beta)))

        file.write("grav {:.4f} {:.4f} {:.4f}\n".format(
            self.gravity[0], self.gravity[1], self.gravity[2]))

        file.write("leb {:.3f}\n".format(self.shearstrainrate))

        file.write("frac {:.3f}\n".format(self.relaxationtime[i]))
        file.write("\n")

        self.write_fortran_pridata(path)

        if volume_fraction <= 0.21 and not self.use_liggghts_for_filling:
            file.write("agglom cub 1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                self.domain[0][0], self.domain[1][0], self.domain[0][1], self.domain[1][1], self.domain[0][2], self.domain[1][2]))

            cvs_string = "cvs 1 1 " + \
                str(len(self.particletemplate.particle.r))
            cvs_string += "\n"
            file.write(cvs_string)

            file.write("rgp {} 1 1".format(particle_count))
            file.write("\n")
        elif self.use_liggghts_for_filling:
            cvs_string = "cvs 1 1 " + \
                str(len(self.particletemplate.particle.r))
            cvs_string += "\n"
            file.write(cvs_string)
            file.write("read  1  1 " + str(self.particle_count[i]) + " cap\n")

        file.write("pri cpi\nvtp bal\n")
        file.write("\n")

        if stress_count > self.body_position_print_count[i]:
            count = 0
            while count < cyc_count:
                file.write("ast on\n")
                inter_count = 0
                while inter_count < stress_count:
                    file.write("cyc {}\n".format(
                        self.body_position_print_count[i]))
                    count += self.body_position_print_count[i]
                    inter_count += self.body_position_print_count[i]
                    file.write("pri cpi\n")
                    # file.write("pri cpi read\nvtp bal\n")
                file.write("pri ast\nast off\n")

                if count % self.save_count == 0:
                    file.write("save SHEAR" +
                               str(int(count // self.save_count)) + "\n")
                file.write("\n")
        elif stress_count < self.body_position_print_count[i]:
            count = 0
            while count < cyc_count:
                inter_count = 0
                while inter_count < self.body_position_print_count[i]:
                    file.write("ast on\n")
                    file.write("cyc {}\n".format(stress_count))
                    count += stress_count
                    inter_count += stress_count
                    file.write("pri ast\nast off\n")
                file.write("pri cpi read\nvtp bal\n")
                if count % self.save_count == 0:
                    file.write("save SHEAR" +
                               str(int(count // self.save_count)) + "\n")
                file.write("\n")
        else:
            count = 0
            while count < cyc_count:
                file.write("ast on\n")
                file.write("cyc {}\n".format(stress_count))
                count += stress_count
                file.write("pri cpi read\nvtp bal\n")
                file.write("pri ast\nast off\n")

                if count % self.save_count == 0:
                    file.write("save SHEAR" +
                               str(int(count // self.save_count)) + "\n")
                file.write("\n")

    def write_fortran_pridata(self, path):
        # for x in self.particletemplate.particle.x:
        #     cvs_string += " {:.5e}".format(x)
        radii = self.particletemplate.particle.r
        u, indices = np.unique(radii, return_inverse=True)

        #    1   5.569668e-04   0.000000e+00   0.000000e+00
        #    1   2.784834e-04   0.000000e+00   0.000000e+00
        #    1   0.000000e+00   0.000000e+00   0.000000e+00
        #    1  -2.784834e-04   0.000000e+00   0.000000e+00
        #    1  -5.569668e-04   0.000000e+00   0.000000e+00
        pridata_string = ""
        for k, ind in enumerate(indices):
            pridata_string += "       " + str(ind + 1) + "  "
            if (self.particletemplate.particle.x[k] >= 0):
                pridata_string += " "
            pridata_string += "{:.6e}".format(
                self.particletemplate.particle.x[k]) + "  "
            if (self.particletemplate.particle.y[k] >= 0):
                pridata_string += " "
            pridata_string += "{:.6e}".format(
                self.particletemplate.particle.y[k]) + "  "
            if (self.particletemplate.particle.z[k] >= 0):
                pridata_string += " "
            pridata_string += "{:.6e}".format(
                self.particletemplate.particle.z[k])
            if k != len(indices) - 1:
                pridata_string += "\n"

        pridata = open(path + "/pridata.txt", "w")
        pridata.write(pridata_string)
        pridata.close()

    def generate_fortran_filling_file(self, path):

        self.write_fortran_pridata(path)

        file = open(path + "/initialize", "w")

        file.write("start {:.6f} {:.6f} {:.6f} 500 150000 15000 1000 2 5 log\n\n".format(
            self.domain[1][0], self.domain[1][1], self.domain[1][2]))

        radii = self.particletemplate.particle.r

        unique_radii = np.unique(radii)

        for j, radius in enumerate(unique_radii):
            file.write("dia {:.4e} {}\n".format(radius * 2.0, j + 1))
            file.write("ymd {:.4e} {}\n".format(
                self.particletemplate.youngsmod, j + 1))
            file.write("prat {:.4e} {}\n".format(
                self.particletemplate.poissonratio, j + 1))
            file.write("dens {:.4e} {}\n".format(
                self.particletemplate.particle.density, j + 1))
            file.write("fric {:.4e} {}\n".format(
                self.particletemplate.frictioncoefficient, j + 1))
            file.write("yie {:.4e} {}\n".format(
                self.particletemplate.yieldstress, j + 1))
            file.write("\n")

        # Restitution Coefficient to silly fortran damping command
        pre_e = 0.5
        pre_beta = -math.log(pre_e) / \
            math.sqrt(math.log(pre_e)**2 + math.pi ** 2)
        file.write("damp {:.4f} {:.4f} 1 0 1\n".format(
            (1 - pre_e), (1 - pre_e) / (2 * math.pi * pre_beta)))

        e = self.particletemplate.restitutioncoefficient
        beta = -math.log(e) / math.sqrt(math.log(e)**2 + math.pi ** 2)

        file.write("grav {:.4f} {:.4f} {:.4f}\n".format(
            self.gravity[0], -20.0, self.gravity[2]))

        file.write("leb {:.3f}\n".format(0.0))
        ##TODO 
        ##file.write("frac {:.3f}\n".format(self.relaxationtime[i]))

        file.write("DWALL FPL(0.0 0.0 0.0 {:.6f} 0.0 0.0 {:.6f} 0.0 {:.6f} 0.0 0.0 {:.6f}) vel(0.0 0.0 0.0) mat(1)\n".format(
            self.domain[1][0], self.domain[1][1], self.domain[1][2], self.domain[1][2]))

        file.write("agglom cub 1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            self.domain[0][0], self.domain[1][0], self.domain[0][1], self.domain[1][1], self.domain[0][2], self.domain[1][2]))
        file.write("agglom cub 2 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            self.domain[0][0], self.domain[1][0], self.domain[1][1] / 2.0, self.domain[1][1], self.domain[0][2], self.domain[1][2]))
        file.write("agglom cub 3 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            self.domain[0][0], self.domain[1][0], self.domain[1][1] / 4.0, self.domain[1][1], self.domain[0][2], self.domain[1][2]))

        cvs_string = "cvs 1 1 " + \
            str(len(self.particletemplate.particle.r)) + "\n"
        file.write(cvs_string)

        total_added_particles = 0
        for p, vf in zip(self.particle_count, self.volume_fractions):
            if vf < 0.18:
                continue
            else:
                this_added_p = p - total_added_particles
                total_added_particles += this_added_p
                if vf < .29:
                    file.write("rgp {} 1 1\n".format(this_added_p))
                elif vf < .39:
                    file.write("rgp {} 1 2\n".format(this_added_p))
                else:
                    file.write("rgp {} 1 3\n".format(this_added_p))
                file.write("pri cpi read\nvtp bal\n")
                file.write("save VF" + str(int(vf * 100)) + "\n")
                file.write("damp {:.4f} {:.4f} 1 0 1\n".format(
                    (1 - e), (1 - e) / (2 * math.pi * beta)))
                file.write("cyc 100000\n")
                file.write("damp {:.4f} {:.4f} 1 0 1\n".format(
                    (1 - pre_e), (1 - pre_e) / (2 * math.pi * pre_beta)))

    def generate_fortran_sbatch_file(self, path, i):

        file = open(path + "/start", "w")

        sbatch_start = "#!/bin/bash\n#SBATCH --job-name="
        sbatch_partition = "\n#SBATCH --partition="
        if self.is_sbatch_high_priority:
            sbatch_partition += "high\n"
        else:
            sbatch_partition += "med\n"

        sbatch_start_mid = "#SBATCH --ntasks=1\n#SBATCH --mem=800mb\n#SBATCH --time=" + self.sbatch_time + \
            "\n#SBATCH --output=output_%j.log\n#SBATCH --error=error_%j.log\npwd; hostname; date\n"
        sbatch_start_end = "rm ./demcfd\nln -s ../../CODE/demcfd\n./demcfd < initialize\ndate\n"
        if self.volume_fractions[i] < 0.21 or self.use_liggghts_for_filling:

            file.write(sbatch_start + self.root_folder_name +
                       "_vf_" + str(self.volume_fractions[i]) + "\n\n")
            file.write(sbatch_partition)
            file.write(sbatch_start_mid)
            file.write(sbatch_start_end)
        else:
            file.write(sbatch_start + self.root_folder_name +
                       "_vf_" + str(self.volume_fractions[i]) + "\n\n")
            file.write(sbatch_start_mid)
            file.write("ln -s ../filling/VF" +
                       str(int(self.volume_fractions[i] * 100)) + "\n")
            file.write(sbatch_partition)
            file.write(sbatch_start_end)

    def generate_liggghts_files(self, mpi_settings, random_orientation=False):
        root_name = "liggghts_" + self.root_folder_name

        if not mpi_settings:
            mpi_settings=[4, 4, 2]
        
        if self.hasdate_in_foldername:
            root_name += "_" + str(date.today())
        if self.extra != "":
            root_name += "_" + self.extra

        try:
            os.makedirs(root_name)

        except OSError:
            # print(OSError)
            print("generate_ligghts_files: root folder already exists")

        try:
            for i in range(len(self.volume_fractions)):
                os.makedirs(root_name + "/" +
                            "cpi_" + self.root_folder_name + "_" + str(i))
        except OSError:
            print("generate_ligghts_files: sub cpi folder already exists")

        self.generate_ligghts_particle_data_file(root_name)

        for i in range(len(self.volume_fractions)):
            if self.is_single_sphere:
                self.generate_liggghts_init_file(
                    root_name, i, [2, 2, 1], random_orientation)
                self.generate_liggghts_sbatch_file(
                    root_name, i, [2, 2, 1])
            elif (self.volume_fractions[i] < 0.1 and not mpi_settings):
                self.generate_liggghts_init_file(
                    root_name, i, [3, 2, 1], random_orientation)
                self.generate_liggghts_sbatch_file(
                    root_name, i, [3, 2, 1])
            else:
                self.generate_liggghts_init_file(
                    root_name, i, mpi_settings, random_orientation)
                self.generate_liggghts_sbatch_file(
                    root_name, i, mpi_settings)

        file_low = open(root_name + "/start_low.sh", "w")
        file_low.write("#!/bin/bash\n")
        file_high = open(root_name + "/start_high.sh", "w")
        file_high.write("#!/bin/bash\n")
        for i in range(len(self.volume_fractions)):
            if self.volume_fractions[i] <= 0.21:
                file_low.write("sbatch sbatch_start_" + str(i) + "\n")
            else:
                file_high.write("sbatch sbatch_start_" + str(i) + "\n")

        file_low.close()
        file_high.close()

    def generate_ligghts_particle_data_file(self, root_name):
        file = open(root_name + "/particle_data", "w")

        x = self.particletemplate.particle.x
        y = self.particletemplate.particle.y
        z = self.particletemplate.particle.z
        r = self.particletemplate.particle.r
        for i in range(len(self.particletemplate.particle.r)):
            file.write("{:.6e} {:.6e} {:.6e} {:.6e}\n".format(
                x[i], y[i], z[i], r[i]))

        file.close()

    def generate_liggghts_init_file(self, root_name, i, mpi_settings, random_orientation=False):

        fout = open(root_name + '/in.' +
                    self.root_folder_name + '_' + str(i), 'w')
        fout.write('# LIGGGHTS shear simulation input file\n\n')

        fout.write('# General simulation options\n')

        fout.write('atom_style        granular\n')

        fout.write('boundary          p p p\n')
        fout.write('newton            off\n')
        fout.write('echo              both\n')
        fout.write('communicate       single vel yes\n')
        fout.write('units             si\n')
        fout.write(
            'log               log.{0}_{1}.liggghts\n'.format(self.root_folder_name, str(i)))
        fout.write('atom_modify       map array\n\n')
        fout.write('processors        {0} {1} {2}\n\n'.format(
            mpi_settings[0], mpi_settings[1], mpi_settings[2]))

        fout.write('# Set domain\n')

        fout.write('region            domain block {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} units box volume_limit 1e-16\n'.format(
            self.domain[0][0], self.domain[1][0], self.domain[0][1], self.domain[1][1], self.domain[0][2], self.domain[1][2]))

        fout.write('create_box        {} domain\n\n'.format(1))
        if not self.is_single_sphere:
            fout.write('neighbor          {0:.3e} bin\n'.format(
                self.particletemplate.particle.max_radius * 1.1))

        # # fout.write('neigh_modify      every 1 delay 0 check no one {0} page {1}\n\n'.format(npar_max,51*npar_max))
        # fout.write(
        #      'neigh_modify      every 1 delay 0 check no contact_distance_factor {0:.1e}\n\n'.format(npar_max))
        fout.write('neigh_modify      every 1 delay 0 check no\n')

        fout.write('# Set particle properties\n')
        fout.write('hard_particles    yes\n')

        fout.write(
            'fix               m1 all property/global youngsModulus peratomtype {0}\n'.format(self.particletemplate.youngsmod))
        fout.write(
            'fix               m2 all property/global poissonsRatio peratomtype {0}\n'.format(self.particletemplate.poissonratio))
        fout.write(
            'fix               m3 all property/global coefficientRestitution peratomtypepair {0} {1}\n'.format(1, self.particletemplate.restitutioncoefficient))
        fout.write(
            'fix               m4 all property/global coefficientFriction peratomtypepair {0} {1}\n'.format(1, self.particletemplate.frictioncoefficient))
        fout.write(
            'fix               m5 all property/global coefficientRollingFriction peratomtypepair {0} {1}\n\n'.format(1, 0.0))

        fout.write('# Set collision models and time step\n')
        fout.write('pair_style        gran model hertz tangential history\n')
        fout.write('pair_coeff        * *\n')

        fout.write('timestep          {0:.4e}\n\n'.format(1e-16))

        fout.write('# Set up particle insertion\n')

        fout.write('group             nve_group{0} region domain\n'.format(1))
        if not self.is_single_sphere:
            fout.write('fix               pts{0} nve_group{0} particletemplate/multisphere 123457 atom_type {0} volume_limit 1e-18 density constant {1:.3f} nspheres {2} ntry 10000000 spheres file {3} scale 1.0 type {4}\n'.format(
                1, self.particletemplate.particle.density, len(self.particletemplate.particle.r), "particle_data", 1))
        if self.is_single_sphere:
            fout.write('fix               pts{0} nve_group{0} particletemplate/sphere 123457 atom_type {0} volume_limit 1e-18 density constant {1:.3f} radius constant {2}\n'.format(
                1, self.particletemplate.particle.density, self.particletemplate.particle.r[0]))
        # fix               pts1 nve_group1 particletemplate/sphere 123457 atom_type 1 volume_limit 1e-18 density constant 2500.000 radius constant 0.000050

        fout.write(
            'fix               pdd{0} nve_group{0} particledistribution/discrete 15485867 1 pts{0} 1.0\n'.format(1))
        fout.write(
            'fix               ins{0} nve_group{0} insert/pack seed 32452867 distributiontemplate pdd{0} maxattempt {1} vel constant 0. 0. 0. &\n'.format(1, int(1e6)))

        if random_orientation:
            # fout.write(
            #     '                  omega constant 0. 0. 0. orientation random insert_every once overlapcheck yes all_in yes region domain ntry_mc 100000000 volumefraction_region {0}\n'.format(self.volume_fractions[i]))
            fout.write(
                '                  omega constant 0. 0. 0. orientation random insert_every once overlapcheck yes all_in yes particles_in_region {0} region domain ntry_mc 100000000 \n'.format(self.particle_count[i]))
        else:
            fout.write(
                '                  omega constant 0. 0. 0. insert_every once overlapcheck yes all_in yes particles_in_region {0} region domain ntry_mc 100000000 \n'.format(self.particle_count[i]))
        if self.is_single_sphere:
            fout.write('fix               integr1 nve_group{0} nve/sphere\n'.format(1))
        else:
            fout.write('fix               ms nve_group{0} multisphere\n'.format(1))

        fout.write('run               1\n\n')

        fout.write('# Run briefly to eliminate potential overlaps\n')
        fout.write('fix               limcheck all slowinit xmax {0:.2e} reset {1} threshold {2:.2e} start_dt {3:.3e} end_dt {4:.3e}\n'.format(
            1e-7, int(1000), 1e-7, 1e-15, self.delta_time * self.relaxationtime[i] * 2.0))  # should make this available for user to set
        fout.write('run               {0}\n\n'.format(int(200_000)))

        fout.write('unfix             limcheck\n')
        fout.write('fix               m3 all property/global coefficientRestitution peratomtypepair {0} {1}\n'.format(
            1, self.particletemplate.restitutioncoefficient))

        fout.write('timestep          {0:.4e}\n\n'.format(
            self.delta_time * self.relaxationtime[i]))

        # if self.volume_fractions[i] > 0.2:
        #     fout.write('fix               leboundary all lebc {0} {1} gtemp {2} ave_reset {3}\n'.format(
        #         self.shearstrainrate, "true", 1e-9, self.stress_print_count[i]))
        # else:
        fout.write('fix               leboundary all lebc {0} {1} gtemp {2} ave_reset {3} body_data cpi_{4}_{5} {6} lock_symmetry {7}\n'.format(
            self.shearstrainrate, "true", 1e-9, self.stress_print_count[i], self.root_folder_name, str(i), self.body_position_print_count[i], self.lock_symmetry[i]))

        # fout.write('run              {0}\n'.format(
        #     int(self.cycle_delay[i])))

        if self.body_position_print_count[i] > 0:
            if self.is_single_sphere:
                fout.write('dump               dmpvtk all custom/vtk {0} {1} id x y z vx vy vz\n'.format(
                    self.body_position_print_count[i], ('vtk_'+self.root_folder_name + '_' + str(i) + '/series_*.vtk')))
            else:
                fout.write('dump               dmpvtk all custom/vtk {0} {1} id id_multisphere x y z vx vy vz\n'.format(
                    self.body_position_print_count[i], ('vtk_'+self.root_folder_name + '_' + str(i) + '/series_*.vtk')))

        fout.write('run               {0}\n'.format(
            int(self.cycle_count[i])))

        fout.close()

    def generate_liggghts_sbatch_file(self,
                                      root_name, i, mpi_settings):

        file = open(root_name + "/sbatch_start_" + str(i), "w")

        mpi_cores = mpi_settings[0] * mpi_settings[1] * mpi_settings[2]

        mpi_tasks = mpi_cores

        mpi_nodes = 1

        if mpi_cores > 32:
            mpi_nodes = 2
            mpi_tasks = mpi_cores // 2 + mpi_cores % 2

        sbatch_start = "#!/bin/bash\n#SBATCH --job-name="
        # sbatch_start_mid = "#SBATCH --nodes={}\n#SBATCH --ntasks-per-node={}\n#SBATCH --cpus-per-task={}\n".format(
        #     mpi_nodes, mpi_cores, 1)
        sbatch_start_mid = "#SBATCH --ntasks-per-node={}\n".format(
             mpi_cores)

        sbatch_partition = "\n#SBATCH --partition="
        if self.is_sbatch_high_priority:
            sbatch_partition += "high\n"
        else:
            sbatch_partition += "med\n"

        sbatch_start_mid += "#SBATCH --mem=16gb\n#SBATCH --time=" + self.sbatch_time + \
            "\n#SBATCH --output=output_%j.log\n#SBATCH --error=error_%j.log\npwd; hostname; date\n"

        sbatch_start_end = "ln -s ~/Projects/LIGGGHTS_CFDRC/src/lmp_auto\nmpiexec -np " + \
            str(mpi_cores) + " ./lmp_auto < in." + \
            self.root_folder_name + '_' + str(i) + " \ndate\n"

        file.write(sbatch_start + root_name + '_' + str(i) + "\n\n")
        file.write(sbatch_partition)
        file.write(sbatch_start_mid)
        file.write("module use /home/suehr/module_files/\n")
        file.write("module load openmpi/4.1.5\n")
        file.write("module load vtk\n")
        file.write(sbatch_start_end)

    def generate_fortran_read_from_liggghts_files(self):

        for i in range(len(self.volume_fractions)):
            ligghts_root_name = "liggghts_" + self.root_folder_name

            if self.hasdate_in_foldername:
                ligghts_root_name += "_" + str(date.today())
            if self.extra != "":
                ligghts_root_name += "_" + self.extra

            fortran_root_name = "fortran_" + self.root_folder_name

            if self.hasdate_in_foldername:
                fortran_root_name += "_" + str(date.today())
            if self.extra != "":
                fortran_root_name += "_" + self.extra

            vf_read_file_path = ligghts_root_name + \
                "/xcm_quat" + str(self.particle_count[i]) + ".txt"

            file = open(vf_read_file_path, 'r')

            write_file = open(fortran_root_name + "/vf_" +
                              str(self.volume_fractions[i]) + "/read.dat", "w")

            header_line = ff.FortranRecordWriter('(1x,i8,7(2x,g15.8),1x,2i8)')

            for line in file.readlines():

                values = line.split()

                new_string = header_line.write([int(values[0]), float(values[1]), float(values[2]), float(
                    values[3]), float(values[4]), float(values[5]), float(values[6]), float(values[7]), 1, 1])

                write_file.write(new_string + "\n")

            file.close()
            write_file.close()

        pass

    ###########################################################################
    # Validation Functions

    # Radial Distribution Function from lebc.py Written by Dr. Gale

    def g0(self, alp):
        # ratio =  np.min(alp/ asmx,1.0-1e-4);
        # /* MPGCOMMENT 03-29-2016 ---> Sinclair and Jackson et al. (1989) */
        # r     =  (1.0 / ( 1.0 - np.cbrt(ratio) ) );
        # /* MPGCOMMENT 03-29-2016 ---> Made up */
        # r     =  7.0 / 10.0 * (1.0 / ( 1.0 - power(ratio,1./3.) ) );
        # r = 1.0/(1-alp) + 1.5*alp/np.power(1.0-alp,2) + 0.5*np.power(alp,2)/np.power(1.0-alp,3);
        F = (alp * alp - 0.8 * alp + 0.636 * (0.8 - 0.636)) / \
            (0.8 * 0.636 - 0.16 - 0.636**2)
        for i in range(0, len(F)):
            if (alp[i] < 0.4):
                F[i] = 1.0
        r = F * (2. - alp) / (2. * (1. - alp)**3) + \
            (1. - F) * 2. / (0.636 - alp)
        return r

    # Granular Kinetic Theory from lebc.py Written by Dr. Gale
    def monodisperse_compute(self):

        alp = np.linspace(0.001, max(np.max(self.volume_fractions), 0.62), 200)

        ds = self.particletemplate.particle.equvi_diameter
        rho = self.particletemplate.particle.density
        T = gtemp = 1.0
        e = self.particletemplate.restitutioncoefficient

        M_PI = np.pi
        eta = 0.5 * (1. + e)
        F = 1. / (eta * (2. - eta) * self.g0(alp)) * (1. + (8. / 5.) * eta * alp * self.g0(alp)) * (1. + (8. / 5.)
                                                                                                    * eta * alp * self.g0(alp) * (3. * eta - 2.)) + 768. * eta * alp * alp * self.g0(alp) / (25. * M_PI)
        gamma = (48. / np.sqrt(M_PI)) * eta * (1. - eta) * rho * \
            alp * alp * self.g0(alp) * np.power(gtemp, 1.5) / ds
        pxy = np.sqrt(-F * (5. / 96.) * rho *
                      np.sqrt(M_PI * gtemp) * ds * (-gamma))
        dudy = -gamma / pxy

        # /* /\* MPGCOMMENT 03-30-2016 ---> Gidaspower et al. (1994) *\/ */
        # // pressure
        Pcol = 2.0 * self.g0(alp) * rho * (alp * alp) * T * (1.0 + e)
        Pkin = rho * alp * T
        pressure = (Pcol + Pkin)
        # shear
        mucol = (4.0 / 5.0) * (alp * alp) * rho * ds * \
            self.g0(alp) * (1.0 + e) * np.sqrt(T / M_PI)
        mukin = (1.0 / 15.0 * np.sqrt(T * M_PI) * rho * ds * self.g0(alp) * (1.0 + e) * alp * alp +
                 1.0 / 6.0 * np.sqrt(T * M_PI) * rho * ds * alp +
                 10.0 / 96.0 * np.sqrt(T * M_PI) * rho * ds / (1.0 + e) / self.g0(alp))
        shear_visc = (mucol + mukin)

        gamma = 48. / np.sqrt(M_PI) * (eta - eta * eta) * rho * \
            alp**2 * self.g0(alp) * np.power(gtemp, 1.5) / \
            self.particletemplate.particle.equvi_diameter
        dudy = -gamma / pxy
        ndcoeff = self.particletemplate.particle.density * \
            self.particletemplate.particle.equvi_diameter**2 * \
            dudy * dudy  # *nd_diam**2*shearStr**2)
        pressure = pressure / ndcoeff
        shear_visc = pxy / ndcoeff
        # #bulk
        # bulk_visc = (4.0/3.0) * (alp*alp) * rho * ds * g0(alp,asmx) * (1.0+e) * np.sqrt(T/M_PI);
        # # thermal
        # kappa_dil = 75.0/384.0 * rho * ds * np.sqrt(M_PI*T);
        # term1 = 2.0/(1.0+e)/g0(alp,asmx)*np.power(1.0+6.0/5.0*(1.0+e)*g0(alp,asmx)*alp,2.0)*kappa_dil;
        # term2 = 2.0*alp*alp*rho*ds*g0(alp,asmx)*(1.0+e)*np.sqrt(T/M_PI);
        # therm_cond = term1+term2;

        # collisional dissipation
        AR_factor = 1.0
        Utrace = 0.0
        ndcoeff2 = ds / (T * np.sqrt(T))
        # coldissipation = (3.0*(1.0-e*e)*g0(alp,asmx)*rho*alp*alp*( 4.0/ds * np.sqrt(T/M_PI) * AR_factor -Utrace)*T)*ndcoeff2 ;
        # coldissipation = 12.0*(1.0-e*e)*g0(alp,asmx)*rho*alp*alp/np.sqrt(M_PI) ;
        coldissipation = 8.0 * (1.0 - e * e) * \
            self.g0(alp) * alp / np.sqrt(M_PI)
        return alp, pressure, shear_visc, coldissipation

    def clear_liggghts_data(self):
        self.liggghts_collisional_normal_stress.clear()
        self.liggghts_collisional_shear_stress.clear()
        self.liggghts_kinetic_normal_stress.clear()
        self.liggghts_kinetic_shear_stress.clear()
        self.liggghts_time.clear()

        self.liggghts_loaded_volume_fraction.clear()

    def liggghts_graph_stress_vs_time_specific(self, i, additional_save_path=""):
        volume_fraction = self.volume_fractions[i]

        ligghts_root_name = "liggghts_" + self.root_folder_name

        if self.hasdate_in_foldername:
            ligghts_root_name += "_" + str(date.today())
        if self.extra != "":
            ligghts_root_name += "_" + self.extra

        file_name = "log." + self.root_folder_name + "_" + str(i) + ".liggghts"

        try:
            file = open(ligghts_root_name + "/" + file_name, "r")
        except OSError:
            # print(OSError)
            print("Liggghts file " + file_name + " not found")
            return

        with file:
            lines = file.readlines()
            time = []

            kinetic_normal_stress = []
            collisional_normal_stress = []

            kinetic_shear_stress = []
            collisional_shear_stress = []

            time_count = 0
            for line in lines:
                if line.startswith("stress: "):
                    values = line.split()
                    time_count += self.stress_print_count[i]
                    time.append(time_count * self.delta_time *
                                self.relaxationtime[i])
                    kinetic_normal_stress.append(float(values[2]))
                    kinetic_shear_stress.append(float(values[3]))
                    collisional_normal_stress.append(float(values[5]))
                    collisional_shear_stress.append(float(values[6]))

            time = np.array(time)
            kinetic_normal_stress = np.array(kinetic_normal_stress)
            collisional_normal_stress = np.array(collisional_normal_stress)
            kinetic_shear_stress = np.array(kinetic_shear_stress)
            collisional_shear_stress = np.array(collisional_shear_stress)

            if len(time) == 0:
                print("Found no stress in Liggghts File: {} ".format(file_name))

            self.liggghts_kinetic_normal_stress.append(kinetic_normal_stress)
            self.liggghts_collisional_normal_stress.append(
                collisional_normal_stress)
            self.liggghts_kinetic_shear_stress.append(kinetic_shear_stress)
            self.liggghts_collisional_shear_stress.append(
                collisional_shear_stress)
            self.liggghts_time.append(np.array(time))
            self.liggghts_loaded_volume_fraction.append(volume_fraction)

            plt.figure(1)
            plt.ylabel("$(σ) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("Dimensionless Time ($\.γ$t)")

            n_length = int(len(kinetic_normal_stress) * self.stress_vs_time_cutoff_range[i]) * \
                self.stress_print_count[i] * \
                self.delta_time * self.relaxationtime[i] * self.shearstrainrate
            
            plt.semilogy([n_length, n_length], [(np.max(kinetic_normal_stress + collisional_normal_stress) / 100 / (self.particletemplate.particle.density * self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2)),
                         (np.max(kinetic_normal_stress + collisional_normal_stress) / (self.particletemplate.particle.density * self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2))])
            self.l_normal_stress_vs_time.append((kinetic_normal_stress + collisional_normal_stress) / (self.particletemplate.particle.density *
                                                                                                       self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2))
            plt.semilogy(time * self.shearstrainrate, (kinetic_normal_stress + collisional_normal_stress) / (self.particletemplate.particle.density *
                         self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), label="Normal " + self.particletemplate.particle.file_shape_name + " vf " + str(volume_fraction))
            self.l_shear_stress_vs_time.append(np.abs(kinetic_shear_stress + collisional_shear_stress) / (self.particletemplate.particle.density *
                                                                                                          self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2))
            plt.semilogy(time * self.shearstrainrate, np.abs(kinetic_shear_stress + collisional_shear_stress) / (self.particletemplate.particle.density *
                         self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), label="Shear " + self.particletemplate.particle.file_shape_name + " vf " + str(volume_fraction))

            plt.legend()

            plt.savefig(ligghts_root_name +
                        "/stress_vs_time_" + str(i) + ".pdf")
            if additional_save_path != "":
                plt.savefig(additional_save_path +
                            "/l_stress_vs_time_{}_{}.pdf".format(self.particletemplate.particle.file_shape_name, i))
            plt.clf()

    def ligghts_graph_stress_vs_time(self, additional_save_path=""):
        for i in range(len(self.volume_fractions)):
            self.liggghts_graph_stress_vs_time_specific(
                i, additional_save_path)

    def liggghts_graph_stress_vs_volume_fraction(self, already_loaded=True, additional_save_path=""):

        if not already_loaded:
            self.ligghts_graph_stress_vs_time()

        ligghts_root_name = "liggghts_" + self.root_folder_name

        if self.hasdate_in_foldername:
            ligghts_root_name += "_" + str(date.today())
        if self.extra != "":
            ligghts_root_name += "_" + self.extra

        volume_fractions = self.liggghts_loaded_volume_fraction

        normal_stress_ave = []
        shear_stress_ave = []

        for (i, (k_normal_stress, c_normal_stress, k_shear_stress, c_shear_stress)) in enumerate(zip(self.liggghts_kinetic_normal_stress, self.liggghts_collisional_normal_stress, self.liggghts_kinetic_shear_stress, self.liggghts_collisional_shear_stress)):

            normal_stresses = k_normal_stress + c_normal_stress
            shear_stresses = k_shear_stress + c_shear_stress

            n_length = int(len(normal_stresses) *
                           self.stress_vs_time_cutoff_range[i])
            s_length = int(len(shear_stresses) *
                           self.stress_vs_time_cutoff_range[i])

            if n_length < 10:
                print("Warning: averaging over <10 values")

            normal_stress_ave.append(
                np.mean(normal_stresses[n_length:]))
            shear_stress_ave.append(abs(
                np.mean(np.abs(shear_stresses[s_length:]))))

            low_normal, high_normal = np.quantile(
                normal_stresses[n_length:], self.quant_range)

            temp_normal = []
            for s in normal_stresses[n_length:]:
                if s > low_normal and s < high_normal:
                    temp_normal.append(s)
            self.liggghts_middle50quartile_last_third_normal.append(
                np.mean(temp_normal))

            low_shear, high_shear = np.quantile(
                shear_stresses[s_length:], self.quant_range)
            temp_stress = []
            for s in shear_stresses[s_length:]:
                if s > low_shear and s < high_shear:
                    temp_stress.append(s)
            self.liggghts_middle50quartile_last_third_shear.append(
                np.mean(temp_stress))

        normal_stress_ave = np.array(normal_stress_ave)
        shear_stress_ave = np.array(shear_stress_ave)
        volume_fractions = np.array(volume_fractions)

        self.l_normal_stress_ave = normal_stress_ave
        self.l_shear_stress_ave = shear_stress_ave
        self.l_volume_fractions = volume_fractions

        self.liggghts_middle50quartile_last_third_normal = np.array(
            self.liggghts_middle50quartile_last_third_normal)
        self.liggghts_middle50quartile_last_third_shear = np.array(
            self.liggghts_middle50quartile_last_third_shear)

        vfLunmono, pnLunmono, snLunmono, _coldissipation = self.monodisperse_compute()

        plt.figure(1)
        plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, pnLunmono, 'k-',
                     linewidth=3, label='Kinetic Theory')

        plt.semilogy(volume_fractions, normal_stress_ave / (self.particletemplate.particle.density *
                                                            self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'rs', linewidth=0.8, linestyle='dotted', label="Normal " + self.particletemplate.particle.file_shape_name)
        # self.add_literature_data_to_graph(True)
        plt.legend()
        plt.savefig(ligghts_root_name +
                    "/normal_stress_vs_vf_{}.pdf".format(self.particletemplate.particle.file_shape_name))
        plt.clf()

        plt.figure(1)
        plt.ylabel("$(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, snLunmono, 'k-',
                     linewidth=4, label='Kinetic Theory')
        plt.semilogy(volume_fractions, shear_stress_ave / (self.particletemplate.particle.density *
                                                           self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'rs', linewidth=0.8, linestyle='dotted',label="Shear " + self.particletemplate.particle.file_shape_name)
        # self.add_literature_data_to_graph(is_normal=False)
        plt.legend()
        plt.savefig(ligghts_root_name + "/shear_stress_vs_vf_{}.pdf".format(
            self.particletemplate.particle.file_shape_name))

        plt.clf()

        plt.figure(1)
        plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, pnLunmono, 'k-',
                     linewidth=3, label='Kinetic Theory')

        plt.semilogy(volume_fractions, self.liggghts_middle50quartile_last_third_normal / (self.particletemplate.particle.density *
                                                                                           self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2),'rs', linewidth=0.8, linestyle='dotted', label="Normal " + self.particletemplate.particle.file_shape_name)
        # self.add_literature_data_to_graph(True)
        plt.legend()
        plt.savefig(ligghts_root_name +
                    "/quartile_normal_stress_vs_vf_{}.pdf".format(self.particletemplate.particle.file_shape_name))
        plt.clf()

        plt.figure(1)
        plt.ylabel("$(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, snLunmono, 'k-',
                     linewidth=4, label='Kinetic Theory')
        plt.semilogy(volume_fractions, self.liggghts_middle50quartile_last_third_shear / (self.particletemplate.particle.density *
                                                                                          self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'rs', linewidth=0.8, linestyle='dotted', label="Shear " + self.particletemplate.particle.file_shape_name)
        # self.add_literature_data_to_graph(is_normal=False)
        plt.legend()
        plt.savefig(ligghts_root_name + "/quartile_shear_stress_vs_vf_{}.pdf".format(
            self.particletemplate.particle.file_shape_name))

        plt.clf()

    def clear_fortran_data(self):
        self.fortran_collisional_normal_stress.clear()
        self.fortran_collisional_shear_stress.clear()
        self.fortran_kinetic_normal_stress.clear()
        self.fortran_kinetic_shear_stress.clear()
        self.fortran_time.clear()
        self.fortran_loaded_volume_fraction.clear()

    def fortran_graph_stress_vs_time_specific(self, i, additional_save_path=""):
        volume_fraction = self.volume_fractions[i]

        fortran_root_name = "fortran_" + self.root_folder_name

        if self.hasdate_in_foldername:
            fortran_root_name += "_" + str(date.today())
        if self.extra != "":
            fortran_root_name += "_" + self.extra

        fortran_root_name += "/vf_" + str(volume_fraction)

        try:
            file = open(fortran_root_name + "/AveStress.dat", "r")
        except OSError:
            # print(OSError)
            print("Fortran AveStress.dat file not found in " + fortran_root_name)
            return

        with file:
            time = []
            kinetic_normal_stress = []
            collisional_normal_stress = []
            kinetic_shear_stress = []
            collisional_shear_stress = []

            line = file.readline()
            line = file.readline()
            while line:
                stringvalues = line.split()
                time.append(float(stringvalues[0]) * self.delta_time *
                            self.relaxationtime[i])
                kinetic_normal_stress.append(float(stringvalues[3]))
                kinetic_shear_stress.append(float(stringvalues[4]))
                collisional_normal_stress.append(float(stringvalues[6]))
                collisional_shear_stress.append(float(stringvalues[7]))
                # Stress_yy.append(float(stringvalues[9]))
                # Stress_xy.append(float(stringvalues[10]))
                line = file.readline()

            kinetic_normal_stress = np.array(kinetic_normal_stress)
            collisional_normal_stress = np.array(collisional_normal_stress)
            kinetic_shear_stress = np.array(kinetic_shear_stress)
            collisional_shear_stress = np.array(collisional_shear_stress)
            time = np.array(time)

            if len(time) == 0:
                print("Found no stress in Fortran Simulation: {} ".format(
                    fortran_root_name))
                return

            self.fortran_kinetic_normal_stress.append(kinetic_normal_stress)
            self.fortran_collisional_normal_stress.append(
                collisional_normal_stress)
            self.fortran_kinetic_shear_stress.append(kinetic_shear_stress)
            self.fortran_collisional_shear_stress.append(
                collisional_shear_stress)
            self.fortran_time.append(time)

            self.fortran_loaded_volume_fraction.append(volume_fraction)

            plt.figure(1)
            plt.ylabel("$(σ) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("Dimensionless Time ($\.γ$t)")
            n_length = int(len(kinetic_normal_stress) *
                           self.stress_vs_time_cutoff_range[i]) * \
                self.stress_print_count[i] * \
                self.delta_time * self.relaxationtime[i] * self.shearstrainrate

            plt.semilogy([n_length, n_length], [np.min((kinetic_normal_stress + collisional_normal_stress) / (self.particletemplate.particle.density *
                         self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2)), np.max((kinetic_normal_stress + collisional_normal_stress) / (self.particletemplate.particle.density *
                                                                                                                                                                     self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2))])

            plt.semilogy(time * self.shearstrainrate, (kinetic_normal_stress + collisional_normal_stress) / (self.particletemplate.particle.density *
                         self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), label="Normal " + self.particletemplate.particle.file_shape_name + " vf " + str(volume_fraction))
            plt.semilogy(time * self.shearstrainrate, np.abs(kinetic_shear_stress + collisional_shear_stress) / (self.particletemplate.particle.density *
                         self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), label="Shear " + self.particletemplate.particle.file_shape_name + " vf " + str(volume_fraction))

            plt.legend()

            plt.savefig(fortran_root_name +
                        "/stress_vs_time_" + str(volume_fraction) + ".pdf")
            if additional_save_path != "":
                plt.savefig(additional_save_path +
                            "/f_stress_vs_time_{}_{}.pdf".format(self.particletemplate.particle.file_shape_name, volume_fraction))
            plt.clf()

    def fortran_graph_stress_vs_time(self, additional_save_path=""):
        for i in range(len(self.volume_fractions)):
            self.fortran_graph_stress_vs_time_specific(i, additional_save_path)

    def fortran_graph_stress_vs_volume_fraction(self, already_loaded=True):

        if not already_loaded:
            self.fortran_graph_stress_vs_time()

        fortran_root_name = "fortran_" + self.root_folder_name

        if self.hasdate_in_foldername:
            fortran_root_name += "_" + str(date.today())
        if self.extra != "":
            fortran_root_name += "_" + self.extra

        # fortran_root_name += "/vf_" + str(volume_fraction)

        volume_fractions = self.fortran_loaded_volume_fraction

        normal_stress_ave = []
        shear_stress_ave = []

        for i, (k_normal_stress, c_normal_stress, k_shear_stress, c_shear_stress) in enumerate(zip(self.fortran_kinetic_normal_stress, self.fortran_collisional_normal_stress, self.fortran_kinetic_shear_stress, self.fortran_collisional_shear_stress)):

            normal_stresses = k_normal_stress + c_normal_stress
            shear_stresses = k_shear_stress + c_shear_stress
            n_length = int(len(normal_stresses) *
                           self.stress_vs_time_cutoff_range[i])
            s_length = int(len(shear_stresses) *
                           self.stress_vs_time_cutoff_range[i])

            if n_length < 10 or s_length < 10:
                print("Warning: averaging over <10 values")
                volume_fractions.pop(i - pop_count)
                pop_count += 1
                continue

            normal_stress_ave.append(
                np.mean(normal_stresses[n_length:]))
            shear_stress_ave.append(abs(
                np.mean(np.abs(shear_stresses[s_length:]))))

        normal_stress_ave = np.array(normal_stress_ave)
        shear_stress_ave = np.array(shear_stress_ave)
        volume_fractions = np.array(volume_fractions)

        vfLunmono, pnLunmono, snLunmono, _coldissipation = self.monodisperse_compute()

        plt.figure(1)
        plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, pnLunmono, 'k-',
                     linewidth=4, label='Kinetic Theory')
        plt.plot(volume_fractions, normal_stress_ave / (self.particletemplate.particle.density *
                                                        self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'rs', linewidth=0.8, linestyle='dotted', label="Normal " + self.particletemplate.particle.file_shape_name)

        # self.add_literature_data_to_graph(True)
        plt.savefig(fortran_root_name +
                    "/normal_stress_vs_vf_{}.pdf".format(self.particletemplate.particle.file_shape_name))

        plt.clf()

        plt.figure(1)
        plt.ylabel("$(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, snLunmono, 'k-',
                     linewidth=4, label='Kinetic Theory')
        plt.plot(volume_fractions, shear_stress_ave / (self.particletemplate.particle.density *
                                                       self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'rs', linewidth=0.8, linestyle='dotted', label="Shear " + self.particletemplate.particle.file_shape_name)
        # self.add_literature_data_to_graph(is_normal=False)
        plt.savefig(fortran_root_name + "/shear_stress_vs_vf_{}.pdf".format(
            self.particletemplate.particle.file_shape_name))

        plt.clf()

    def add_literature_data_to_graph(self, is_normal,  color='blue'):
        curl_vf = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]
        # stresses xy
        curl_0_xy = [0.18431267635898857, 0.10533966487104529, 0.09104493601081359,
                     0.1915798671443623, 0.32080276305435673, 0.6490373002283759, 1.0214936003757875]
        curl_1_xy = [0.18367692476011882, 0.09760702116503658, 0.08463205830986278,
                     0.1611901079542458, 0.41921213773960553, 1.2879637136782673, 3.146242914956905]
        curl_2_xy = [0.18311016735456978, 0.1066595457969883, 0.08208424718225502,
                     0.15766605794049596, 0.43870675890136124, 1.5780930315790995, 4.327870848966893]
        curl_3_xy = [0.18874054107830066, 0.11974615765895519, 0.08157479841902532,
                     0.14821534884967988, 0.442685000126121, 1.6505722502853597, 4.687382633867693]
        curl_4_xy = [0.2637930002340734, 0.12547916766519773, 0.08448164151450277,
                     0.15040874168152926, 0.39937850736643393, 1.4991871185171846, 4.281167837710025]
        curl_5_xy = [0.32328908868525696, 0.18259869014127486, 0.11727442091695756,
                     0.15494254128609156, 0.3961673212314838, 1.1544599047372266, 2.342318464760482]
        # stresses yy
        curl_0_yy = [0.5656900319112476, 0.36929181249218357, 0.37930530302455695,
                     0.7585219698293981, 1.428640715619119, 3.136822877318178, 5.0898130552298655]
        curl_1_yy = [0.5563586304922369, 0.35401643400143334, 0.34337368122020173,
                     0.6938718501732444, 1.648562469649389, 5.2481342812454965, 13.138746662865987]
        curl_2_yy = [0.5717188732159838, 0.3606425615712801, 0.3446365428676307,
                     0.7018431256406372, 1.8289130557414053, 6.374434848472287, 17.896000134764208]
        curl_3_yy = [0.569509910549374, 0.3677985373849852, 0.33521967994911417,
                     0.7069983845934169, 1.877948004122656, 6.567859501608791, 18.995213808533574]
        curl_4_yy = [0.7468499557518299, 0.4116384082230158, 0.35798385934003285,
                     0.696624311780879, 1.7939588565221556, 6.208763335527074, 17.00182810119522]
        curl_5_yy = [1.0101664681952056, 0.6001889445369429, 0.4451240765608879,
                     0.6469703725119114, 1.4315900744395864, 4.059044620535561, 8.331972972720303]

        rod2_vf = [0.02533,
                   0.05016,
                   0.09937,
                   0.20135,
                   0.29942,
                   0.39916,
                   0.51278]

        rod2_normal = [2.0985,
                       1.0669,
                       0.7719,
                       1.0152,
                       1.9158,
                       4.2535,
                       13.93]

        rod6_vf = [
            0.02639,
            0.05138,
            0.09918,
            0.19967,
            0.29766,
            0.40227,
            0.42095,
            0.44532,
            0.45671,
            0.46438]

        rod6_normal = [
            0.42828,
            0.34198,
            0.449,
            0.8323,
            1.2876,
            2.9912,
            3.7171,
            5.149,
            6.339,
            19.253
        ]

        rod4_vf = [
            0.02497,
            0.05071,
            0.09839,
            0.1997,
            0.30101,
            0.40075,
            0.47964,
            0.50017
        ]

        rod4_normal = [
            0.7564,
            0.5088,
            0.474,
            0.9109,
            1.7663,
            3.8864,
            14.428,
            35.295]

        rod2_vis = [
            0.6436,
            0.30878,
            0.20375,
            0.25643,
            0.48546,
            1.1089,
            3.6883
        ]

        rod4_vis = [
            0.24964,
            0.15551,
            0.14578,
            0.25643,
            0.48945,
            1.022,
            3.5974,
            8.01
        ]

        rod6_vis = [
            0.14683,
            0.11591,
            0.1411,
            0.2344,
            0.30731,
            0.6794,
            0.7936,
            1.1558,
            1.361,
            4.1325
        ]
        filestr = self.particletemplate.particle.file_shape_name
        if is_normal:
            if filestr == "rod2":
                plt.semilogy(rod2_vf, rod2_normal, '*',
                             color=color, label='Guo: rod2')
            if filestr == "rod4":
                plt.semilogy(rod4_vf, rod4_normal, '*',
                             color=color, label='Guo: rod4')
            if filestr == "rod6":
                plt.semilogy(rod6_vf, rod6_normal, '*',
                             color=color, label='Guo: rod6')
            if filestr == "curl0":
                plt.semilogy(rod4_vf, rod4_normal, '*',
                             color=color, label='Suehr: rod4')
                plt.semilogy(rod6_vf, rod6_normal, '*',
                             color=color, label='Guo: rod6')
            # if filestr == "curl_5":
            #     plt.semilogy(curl_vf, curl_5_yy, '*',
            #                  color=color, label='Suehr: curl_5')
            
            # if filestr == "rod3":
            #     plt.semilogy(rod2_vf, rod2_normal, '*',
            #                  color=color, label='Guo: rod2')
            #     plt.semilogy(rod4_vf, rod4_normal, 'v',
            #                  color=color, label='Guo: rod4')
            if filestr == "rod5":
                plt.semilogy(curl_vf, curl_0_yy, '*', color=color,
                             label='Suehr: curl_0 or rod5')
        else:
            if filestr == "rod2":
                plt.semilogy(rod2_vf, rod2_vis, '*',
                             color=color, label='Guo: rod2')
            if filestr == "rod4":
                plt.semilogy(rod4_vf, rod4_vis, '*',
                             color=color, label='Guo: rod4')
            if filestr == "rod6":
                plt.semilogy(rod6_vf, rod6_vis, '*',
                             color=color, label='Guo: rod6')
                
            # if filestr == "rod3":
            #     plt.semilogy(rod2_vf, rod2_vis, '*',
            #                  color=color, label='Guo: rod2')
            #     plt.semilogy(rod4_vf, rod4_vis, 'v',
            #                  color=color, label='Guo: rod4')

            if filestr == "curl0":
                plt.semilogy(rod4_vf, rod4_vis, '*',
                             color=color, label='Suehr: rod4')
                plt.semilogy(rod6_vf, rod6_vis, '*',
                             color=color, label='Guo: rod6')
            # if filestr == "curl_3":
            #     plt.semilogy(curl_vf, curl_3_xy, '*',
            #                  color=color, label='Suehr: curl_3')
            # if filestr == "curl_5":
            #     plt.semilogy(curl_vf, curl_3_xy, '*',
            #                  color=color, label='Suehr: curl_5')
            if filestr == "rod5":
                plt.semilogy(curl_vf, curl_0_xy, '*', color=color,
                             label='Suehr: curl_0 or rod5')

    def load_vf_vs_stress(self, use_fortran=True, use_liggghts=True, max_volumefraction=0):
        if use_fortran:
            # FORTRAN SECTION
            volume_fractions = self.fortran_loaded_volume_fraction

            normal_stress_ave = []
            shear_stress_ave = []

            pop_count = 0
            for i, (k_normal_stress, c_normal_stress, k_shear_stress, c_shear_stress) in enumerate(zip(self.fortran_kinetic_normal_stress, self.fortran_collisional_normal_stress, self.fortran_kinetic_shear_stress, self.fortran_collisional_shear_stress)):

                # if max_volumefraction != 0 and i >= max_volumefraction:
                #     continue

                normal_stresses = k_normal_stress + c_normal_stress
                shear_stresses = k_shear_stress + c_shear_stress
                n_length = int(len(normal_stresses) *
                               self.stress_vs_time_cutoff_range[i])
                s_length = int(len(shear_stresses) *
                               self.stress_vs_time_cutoff_range[i])

                if n_length < 10 or s_length < 10:
                    print("Warning: averaging over <10 values")
                    volume_fractions.pop(i - pop_count)
                    pop_count += 1
                    continue

                normal_stress_ave.append(
                    np.mean(normal_stresses[:n_length]))

                shear_stress_ave.append(abs(
                    np.mean(np.abs(shear_stresses[:s_length]))))

            self.f_normal_stress_ave = np.array(normal_stress_ave)
            self.f_shear_stress_ave = np.array(shear_stress_ave)
            self.f_volume_fractions = np.array(volume_fractions)

        if use_liggghts:
            # LIGGGHTS SECTION
            volume_fractions = self.liggghts_loaded_volume_fraction

            normal_stress_ave = []
            shear_stress_ave = []

            self.liggghts_middle50quartile_last_third_normal = []
            self.liggghts_middle50quartile_last_third_shear = []

            pop_count = 0
            for i, (k_normal_stress, c_normal_stress, k_shear_stress, c_shear_stress) in enumerate(zip(self.liggghts_kinetic_normal_stress, self.liggghts_collisional_normal_stress, self.liggghts_kinetic_shear_stress, self.liggghts_collisional_shear_stress)):

                # if max_volumefraction != 0 and i >= max_volumefraction:
                #     continue

                normal_stresses = k_normal_stress + c_normal_stress
                shear_stresses = k_shear_stress + c_shear_stress
                shear_stresses = np.abs(shear_stresses)

                n_length = int(len(normal_stresses) *
                               self.stress_vs_time_cutoff_range[i])
                s_length = int(len(shear_stresses) *
                               self.stress_vs_time_cutoff_range[i])

                if n_length < 10 or s_length < 10:
                    print("Warning: averaging over <10 values")
                    volume_fractions.pop(i - pop_count)
                    pop_count += 1
                    continue

                # Temp Fixes for Curl data, ignoring spikes in stress

                normal_stress_ave.append(
                    np.mean(normal_stresses[n_length:]))
                shear_stress_ave.append(abs(
                    np.mean(np.abs(shear_stresses[s_length:]))))

                low_normal, high_normal = np.quantile(
                    normal_stresses[n_length:], self.quant_range)

                temp_normal = []
                for s in normal_stresses[n_length:]:
                    if s > low_normal and s < high_normal:
                        temp_normal.append(s)
                self.liggghts_middle50quartile_last_third_normal.append(
                    np.mean(temp_normal))

                low_shear, high_shear = np.quantile(
                    shear_stresses[s_length:], self.quant_range)
                temp_stress = []
                for s in shear_stresses[s_length:]:
                    if s > low_shear and s < high_shear:
                        temp_stress.append(s)
                self.liggghts_middle50quartile_last_third_shear.append(
                    np.mean(temp_stress))

            self.l_normal_stress_ave = np.array(normal_stress_ave)
            self.l_shear_stress_ave = np.array(shear_stress_ave)
            self.l_volume_fractions = np.array(volume_fractions)

            self.liggghts_middle50quartile_last_third_normal = np.array(
                self.liggghts_middle50quartile_last_third_normal)
            self.liggghts_middle50quartile_last_third_shear = np.array(
                self.liggghts_middle50quartile_last_third_shear)

    def graph_liggghts_vs_fortran(self, general_folder_name=""):
        if general_folder_name == "":
            general_folder_name = self.root_folder_name

        try:
            os.makedirs(general_folder_name)
        except OSError:
            # print(OSError)
            print("General folder name for graph already exists")

        self.clear_fortran_data()
        self.clear_liggghts_data()
        self.fortran_graph_stress_vs_time(
            additional_save_path=general_folder_name)
        self.ligghts_graph_stress_vs_time(
            additional_save_path=general_folder_name)

        # FORTRAN SECTION
        volume_fractions = self.fortran_loaded_volume_fraction

        normal_stress_ave = []
        shear_stress_ave = []

        pop_count = 0
        for i, (k_normal_stress, c_normal_stress, k_shear_stress, c_shear_stress) in enumerate(zip(self.fortran_kinetic_normal_stress, self.fortran_collisional_normal_stress, self.fortran_kinetic_shear_stress, self.fortran_collisional_shear_stress)):

            normal_stresses = k_normal_stress + c_normal_stress
            shear_stresses = k_shear_stress + c_shear_stress
            n_length = int(len(normal_stresses) *
                           self.stress_vs_time_cutoff_range[i])
            s_length = int(len(shear_stresses) *
                           self.stress_vs_time_cutoff_range[i])

            if n_length < 10 or s_length < 10:
                print("Warning: averaging over <10 values")
                volume_fractions.pop(i - pop_count)
                pop_count += 1
                continue

            normal_stress_ave.append(
                np.mean(normal_stresses[n_length:]))
            shear_stress_ave.append(abs(
                np.mean(np.abs(shear_stresses[s_length:]))))

        self.f_normal_stress_ave = np.array(normal_stress_ave)
        self.f_shear_stress_ave = np.array(shear_stress_ave)
        self.f_volume_fractions = np.array(volume_fractions)

        # LIGGGHTS SECTION
        volume_fractions = self.liggghts_loaded_volume_fraction

        normal_stress_ave = []
        shear_stress_ave = []

        pop_count = 0
        for i, (k_normal_stress, c_normal_stress, k_shear_stress, c_shear_stress) in enumerate(zip(self.liggghts_kinetic_normal_stress, self.liggghts_collisional_normal_stress, self.liggghts_kinetic_shear_stress, self.liggghts_collisional_shear_stress)):

            normal_stresses = k_normal_stress + c_normal_stress
            shear_stresses = k_shear_stress + c_shear_stress
            n_length = int(len(normal_stresses) *
                           self.stress_vs_time_cutoff_range[i])
            s_length = int(len(shear_stresses) *
                           self.stress_vs_time_cutoff_range[i])

            if n_length < 10 or s_length < 10:
                print("Warning: averaging over <10 values")
                volume_fractions.pop(i - pop_count)
                pop_count += 1
                continue

            normal_stress_ave.append(
                np.mean(normal_stresses[n_length:]))
            shear_stress_ave.append(abs(
                np.mean(np.abs(shear_stresses[s_length:]))))

        self.l_normal_stress_ave = np.array(normal_stress_ave)
        self.l_shear_stress_ave = np.array(shear_stress_ave)
        self.l_volume_fractions = np.array(volume_fractions)

        vfLunmono, pnLunmono, snLunmono, _coldissipation = self.monodisperse_compute()

        plt.figure(1)
        plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, pnLunmono, 'k-',
                     linewidth=4, label='Kinetic Theory')
        plt.plot(self.f_volume_fractions, self.f_normal_stress_ave / (self.particletemplate.particle.density *
                                                                      self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'gs', linewidth=1, label="Fortran Normal " + self.particletemplate.particle.file_shape_name)
        plt.plot(self.l_volume_fractions, self.l_normal_stress_ave / (self.particletemplate.particle.density *
                                                                      self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'rs', linewidth=0.8, linestyle='dotted', label="LIGGGHTS Normal " + self.particletemplate.particle.file_shape_name)
        # self.add_literature_data_to_graph(is_normal=True)
        plt.legend()
        plt.savefig(general_folder_name +
                    "/normal_stress_vs_vf_{}.pdf".format(self.particletemplate.particle.file_shape_name))

        plt.clf()

        plt.figure(1)
        plt.ylabel("$(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
        plt.xlabel("Solid volume fraction (ν)")
        plt.semilogy(vfLunmono, snLunmono, 'k-',
                     linewidth=4, label='Kinetic Theory')
        plt.plot(self.f_volume_fractions, self.f_shear_stress_ave / (self.particletemplate.particle.density *
                                                                     self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'gs', linewidth=1, label="Fortran Shear " + self.particletemplate.particle.file_shape_name)
        plt.plot(self.l_volume_fractions, self.l_shear_stress_ave / (self.particletemplate.particle.density *
                                                                     self.shearstrainrate**2 * self.particletemplate.particle.equvi_diameter**2), 'rs', linewidth=0.8, linestyle='dotted', label="LIGGGHTS Shear " + self.particletemplate.particle.file_shape_name)
        # self.add_literature_data_to_graph(is_normal=False)
        plt.legend()
        plt.savefig(general_folder_name + "/shear_stress_vs_vf_{}.pdf".format(
            self.particletemplate.particle.file_shape_name))

        plt.clf()


# class HandlerLineImage(HandlerBase):
#     def __init__(self, path, space=15, offset=10):
#         self.space = space
#         self.offset = offset
#         self.image_data = plt.imread(path)
#         super(HandlerLineImage, self).__init__()

#     def create_artists(
#         self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
#     ):

#         width = width * 1.4
#         height = height * 1.4

#         l = matplotlib.lines.Line2D(
#             [
#                 xdescent + self.offset,
#                 xdescent + (width - self.space) / 3.0 + self.offset,
#             ],
#             [ydescent + height / 2.0, ydescent + height / 2.0],
#         )
#         l.update_from(orig_handle)
#         l.set_clip_on(False)
#         l.set_transform(trans)

#         bb = Bbox.from_bounds(
#             xdescent + (width + self.space) / 3.0 + self.offset,
#             ydescent,
#             height * self.image_data.shape[1] / self.image_data.shape[0],
#             height,
#         )

#         tbb = TransformedBbox(bb, trans)
#         image = BboxImage(tbb)
#         image.set_data(self.image_data)

#         self.update_prop(image, orig_handle, legend)
#         return [l, image]

class SimulationCompare(object):
    simulations: list[ShearSimulation]

    def __init__(self, simulations):
        self.simulations = simulations

    # def print_lowest_volumefraction_stress(self):
    #     for (i, simulation) in enumerate(self.simulations):
    #         lowest_yy = (simulation.l_normal_stress_ave[0]) / (simulation.particletemplate.particle.density *
    #                                                            simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2)

    #         lowest_xy = (simulation.l_shear_stress_ave[0]) / (simulation.particletemplate.particle.density *
    #                                                           simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2)

    #         print("vf= ", simulation.volume_fractions[0],
    #               " normal", i, "= ", lowest_yy, " shear", i, "= ", lowest_xy)

    def stress_vs_vf_graph_compare(self, use_fortran=True, use_liggghts=True, general_folder_name="", series_name=""):
        if general_folder_name == "":
            general_folder_name = "series_" + \
                self.simulations[0].root_folder_name
        if series_name == "":
            series_name = "series_" + \
                self.simulations[0].particletemplate.particle.file_shape_name

        try:
            os.makedirs(general_folder_name)
        except OSError:
            # print(OSError)
            print("General folder name for graph already exists")

        # First we need to load all the data
        for simulation in self.simulations:
            #     # # Clear all the data
            #     # simulation.clear_liggghts_data()
            #     # simulation.clear_fortran_data()
            #     # Load all the data! (and remake stress vs time graphs into their respective folders)
            if use_liggghts:
                simulation.ligghts_graph_stress_vs_time(
                    additional_save_path=general_folder_name)
            if use_fortran:
                simulation.fortran_graph_stress_vs_time(
                    additional_save_path=general_folder_name)
            simulation.load_vf_vs_stress(
                use_liggghts=use_liggghts, use_fortran=use_fortran)

        # We are only using the first simulation to get the monodisperse data
        vfLunmono, pnLunmono, snLunmono, _coldissipation = self.simulations[0].monodisperse_compute(
        )

        # Now that all the data is loaded, we can graph the stress vs volume fraction in on nice
        plt.figure(1)

        plt.semilogy(vfLunmono, pnLunmono, 'k-',
                     linewidth=2, label='Kinetic Theory')

        markers = ["o", "v", "x", "s", "*", "D", "+", "p"]
        colors = ["#e05252",
                  "#e1893f", "#D6B11F", "#91b851", "#52a0e0",  "#1F73B8", "#7768ae", "brown"]
        # colors = ['Red', 'Orange', 'Pink',
        #           'Green', 'LightBlue', 'Blue', "Purple"]
        labelnames = ["Curl 0", "Curl 1",
                      "Curl 2", "Curl 3", "Curl 4", "Curl 5", "Curl 6", "Rod 3"]

        # curl_images = []
        # if series_name == "curls":
        #     curl_images.append(HandlerLineImage(
        #         "./CURLS/pictures/curl_0_thumbnail.png"))
        #     curl_images.append(HandlerLineImage(
        #         "./CURLS/pictures/curl_1_thumbnail.png"))
        #     curl_images.append(HandlerLineImage(
        #         "./CURLS/pictures/curl_2_thumbnail.png"))
        #     curl_images.append(HandlerLineImage(
        #         "./CURLS/pictures/curl_3_thumbnail.png"))
        #     curl_images.append(HandlerLineImage(
        #         "./CURLS/pictures/curl_4_thumbnail.png"))
        #     curl_images.append(HandlerLineImage(
        #         "./CURLS/pictures/curl_5_thumbnail.png"))

        plots = []
        for (i, simulation) in enumerate(self.simulations):
            plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("Solid volume fraction (ν)")

            if use_fortran:
                plt.plot(simulation.f_volume_fractions, simulation.f_normal_stress_ave / (simulation.particletemplate.particle.density *
                                                                                          simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2), color=colors[i], marker=markers[i], linewidth=0.8, linestyle='dashed', label="Fortran Normal " + simulation.particletemplate.particle.file_shape_name)

            if use_liggghts:
                plt.plot(simulation.l_volume_fractions, simulation.liggghts_middle50quartile_last_third_normal / (simulation.particletemplate.particle.density *
                                                                                                                  simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2), color=colors[i], marker=markers[i], linewidth=0.8, linestyle='dotted', label=labelnames[i])

            # simulation.add_literature_data_to_graph(
                # is_normal=True,  color=colors[i])

        plt.legend(loc='upper left')
        # plt.legend(
        #     # plots,
        #     labelnames,
        #     # handler_map={
        #     #     plots[0]: curl_images[0],
        #     #     plots[1]: curl_images[1],
        #     #     plots[2]: curl_images[2],
        #     #     plots[3]: curl_images[3],
        #     #     plots[4]: curl_images[4],
        #     #     plots[5]: curl_images[5],
        #     # },
        #     # frameon=False,
        #     handlelength=5.0,
        #     labelspacing=0.0,
        #     fontsize=14,
        #     borderpad=0.15,
        #     loc=2,
        #     handletextpad=0.2,
        #     borderaxespad=0.15,
        # )

        plt.savefig(general_folder_name +
                    "/normal_stress_vs_vf_{}.pdf".format(series_name), dpi=250)
        plt.clf()

        plt.figure(1)
        plt.semilogy(vfLunmono, snLunmono, 'k-',
                     linewidth=2, label='Kinetic Theory')
        plots = []
        for (i, simulation) in enumerate(self.simulations):
            plt.ylabel("$(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("Solid volume fraction (ν)")

            if use_fortran:
                plt.plot(simulation.f_volume_fractions, simulation.f_shear_stress_ave / (simulation.particletemplate.particle.density *
                                                                                         simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2), color=colors[i], marker=markers[i], linewidth=0.8, linestyle='dashed', label="Fortran Shear " + simulation.particletemplate.particle.file_shape_name)
            if use_liggghts:
                print(simulation.liggghts_middle50quartile_last_third_shear)
                plt.plot(simulation.l_volume_fractions, simulation.liggghts_middle50quartile_last_third_shear / (simulation.particletemplate.particle.density *
                                                                                                                 simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2), color=colors[i], marker=markers[i], linewidth=0.8, linestyle='dotted', label=labelnames[i])
            # simulation.add_literature_data_to_graph(
            #     is_normal=False, color=colors[i])
        plt.legend(loc='upper left')
        # plt.legend(
        #     # plots,
        #     labelnames,
        #     # handler_map={
        #     #     plots[0]: curl_images[0],
        #     #     plots[1]: curl_images[1],
        #     #     plots[2]: curl_images[2],
        #     #     plots[3]: curl_images[3],
        #     #     plots[4]: curl_images[4],
        #     #     plots[5]: curl_images[5],
        #     # },
        #     # frameon=False,
        #     handlelength=5.0,
        #     labelspacing=0.0,
        #     fontsize=14,
        #     borderpad=0.15,
        #     loc=2,
        #     handletextpad=0.2,
        #     borderaxespad=0.15,
        # )
        plt.savefig(general_folder_name + "/shear_stress_vs_vf_{}.pdf".format(
            series_name), dpi=250)
        plt.clf()

    def effective_projected_area(self, use_fortran=False, use_liggghts=True, general_folder_name="", series_name="", test_lowest_vf_count=0):
        import overlapping_circles as oc
        if general_folder_name == "":
            general_folder_name = "effective_projected_area_" + \
                self.simulations[0].root_folder_name
        if series_name == "":
            series_name = "effective_projected_area_" + \
                self.simulations[0].particletemplate.particle.file_shape_name

        try:
            os.makedirs(general_folder_name)
            import overlapping_circles as oc
        except OSError:
            # print(OSError)
            print("General folder name for effective_projected_area already exists")

        if use_liggghts:
            ave_effective_projected_area = []
            stress_yy = []
            stress_xy = []

            plt.figure(1)
            plt.figure(2)

            start_stop = [[43046000, 45051000], [
                33046000, 35051000], [34001000, 35051000]]
            colors = ["#e05252", "#e1893f", "#D6B11F",
                      "#91b851", "#52a0e0",  "#1F73B8", "#7768ae", "brown"]

            markers = ["o", "v", "x", "s", "*", "D", "+"]

            line_colors = ["brown","black"]
            for i in range(0, 2):
                # Assumes only one size radii per simulation
                ave_effective_projected_area.append([])
                stress_yy.append([])
                stress_xy.append([])
                for simulation in self.simulations:
                    radius = simulation.particletemplate.particle.r[0]
                    print("started {0}".format(simulation.volume_fractions[i]))
                    extra = ""
                    if simulation.extra != "":
                        extra = "_"+ simulation.extra
                    ligghts_folder = 'liggghts_'+simulation.root_folder_name + extra
                    cpi_folder = 'cpi_'+simulation.root_folder_name + \
                        '_'

                    projected_area = []
                    delta = 0
                    for j in range(start_stop[i][0], start_stop[i][1], simulation.body_position_print_count[i]):

                        found_files = False
                        try:
                            cpi_file = open(ligghts_folder + '/' + cpi_folder +
                                            str(i) + '/cpi_' + str(delta + j) + '.txt', "r")
                            found_files = True
                        except OSError:
                            # print(OSError)
                            print("Could not find:" + ligghts_folder + '/' +
                                  cpi_folder + str(i) + '/cpi_' + str(delta + j) + '.txt')
                            delta -= 1000
                            continue

                        # get particle rotations
                        particles_quats = np.empty(
                            (simulation.particle_count[i], 4))
                        with cpi_file:
                            for j in range(simulation.particle_count[i]):
                                line = cpi_file.readline()
                                values = line.split()
                                particles_quats[j][0] = values[5]
                                particles_quats[j][1] = values[6]
                                particles_quats[j][2] = values[7]
                                particles_quats[j][3] = values[8]

                        # initalize particles
                        particles = np.empty(
                            (simulation.particle_count[i], len(simulation.particletemplate.particle.r), 3))
                        for k in range(simulation.particle_count[i]):
                            for (j, (x, y, z)) in enumerate(zip(simulation.particletemplate.particle.x, simulation.particletemplate.particle.y, simulation.particletemplate.particle.z)):
                                particles[k][j][0] = x
                                particles[k][j][1] = y
                                particles[k][j][2] = z

                        # Rotate each particle
                        for k in range(simulation.particle_count[i]):
                            for j in range(len(simulation.particletemplate.particle.r)):
                                particles[k][j] = point_rotation_by_quaternion(
                                    particles[k][j], particles_quats[k])

                        # Get Average Projected area
                        for j in range(simulation.particle_count[i]):
                            nodes = []
                            radii = []
                            for k in range(len(simulation.particletemplate.particle.r)):
                                nodes.append(
                                    [particles[j][k][1], particles[j][k][2]])
                                radii.append(radius)

                            nodes = np.array(nodes)
                            radii = np.array(radii)

                            area = oc.getArea(nodes, radii, graph_test=False)
                            projected_area.append(area)

                    equiv_radius = simulation.particletemplate.particle.equvi_diameter / 2.0
                    sphere_projected_area =  math.pi * equiv_radius**2
                    ave_effective_projected_area[i].append(
                        (np.mean(projected_area)/sphere_projected_area)**-2)

                    simulation.liggghts_graph_stress_vs_time_specific(i)
                    simulation.load_vf_vs_stress(
                        use_liggghts=True, use_fortran=False)
                    stress_yy[i].append(simulation.l_normal_stress_ave[i] / (simulation.particletemplate.particle.density *
                                                                             simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2))
                    stress_xy[i].append(simulation.l_shear_stress_ave[i] / (simulation.particletemplate.particle.density *
                                                                            simulation.shearstrainrate**2 * simulation.particletemplate.particle.equvi_diameter**2))


            for i in range(0, 2):
                plt.scatter(ave_effective_projected_area[i], stress_yy[i], marker=markers[i], linewidth=1,
                            color=colors)
                
                m, b = np.polyfit(
                    ave_effective_projected_area[i], stress_yy[i], 1)
                regression = np.array(ave_effective_projected_area[i]) * m + b
                plt.plot(ave_effective_projected_area[i], regression,
                         color=line_colors[i], linestyle="dashed",label="VF = {0}".format(simulation.volume_fractions[i]))

            plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("$(A_{eff}^{-2})$")
            plt.legend()
            plt.savefig(general_folder_name +
                        "/eff_proj_area_yy_{}.pdf".format(series_name))


            plt.clf()
            for i in range(0, 2):
                plt.scatter(ave_effective_projected_area[i], stress_xy[i], marker=markers[i], linewidth=1,
                            color=colors)
               
                m, b = np.polyfit(
                    ave_effective_projected_area[i], stress_xy[i], 1)
                regression = np.array(ave_effective_projected_area[i]) * m + b
                plt.plot(ave_effective_projected_area[i], regression,
                         color=line_colors[i], linestyle="dashed", label="VF = {0}".format(simulation.volume_fractions[i]))
            
            
            plt.ylabel("$(σ_{xy}) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("$(A_{eff}^{-2})$")

            plt.legend()
            plt.savefig(general_folder_name + "/eff_proj_area_xy_{}.pdf".format(
                series_name))
        

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


    def interlocking_time_historgram_command_print(self, use_fortran=False, use_liggghts=True, general_folder_name="", series_name="", test_lowest_vf_count=0):
        if use_liggghts:
            for simulation in self.simulations:
                ligghts_folder = 'liggghts_'+simulation.root_folder_name +"_"+ simulation.extra
                cpi_folders = ""

                start_count = ""
                for j in range(0, 4):
                    # Assumes only one size radii per simulation

                   

                    # print("started {0}".format(simulation.volume_fractions[j]))
                    
                    cpi_folder = 'cpi_'+simulation.root_folder_name + \
                        '_' + str(j)
                    cpi_folders += cpi_folder + " "

                    simulation.liggghts_graph_stress_vs_time_specific(j)
                    s_length = int(len(simulation.l_shear_stress_vs_time[j]) *
                                   simulation.stress_vs_time_cutoff_range[j])
                    start_count += str(s_length * simulation.body_position_print_count[j])  + " "

                print("./interlocking_detection domain {0:.6f} {1:.6f} {2:.6f} particle {3} simulation {4} volumefractions {5}startcount {6}".format(simulation.domain[1][0],simulation.domain[1][1],simulation.domain[1][2], simulation.root_folder_name, ligghts_folder, cpi_folders, start_count))

                    
                


    def interlocking_time_historgram(self, use_fortran=False, use_liggghts=True, general_folder_name="", series_name="", test_lowest_vf_count=0):
        import overlapping_circles as oc
        if general_folder_name == "":
            general_folder_name = "Interlocking_Histograms_" + \
                self.simulations[0].root_folder_name
        if series_name == "":
            series_name = "Interlocking_Histograms_" + \
                self.simulations[0].particletemplate.particle.file_shape_name

        try:
            os.makedirs(general_folder_name)
            import overlapping_circles as oc
        except OSError:
            # print(OSError)
            print("General folder name for Interlocking_Histograms already exists")

        if use_liggghts:
            for simulation in self.simulations:
                for j in range(0, len(simulation.volume_fractions)):
                    # Assumes only one size radii per simulation
                    
                    frames = 50000#simulation.body_position_print_count[j]
                    if j <= 3:
                        frames = 5000

                    dt = simulation.delta_time * simulation.relaxationtime[j]
                    print(frames)
                    print(dt)

                    print("started {0}".format(simulation.volume_fractions[j]))
                    ligghts_folder = 'liggghts_'+simulation.root_folder_name +"_"+ simulation.extra
                    cpi_folder = 'cpi_'+simulation.root_folder_name + \
                        '_'

                    try:
                        collisions_file = open(ligghts_folder + '/' + cpi_folder +
                                               str(j) + '/array.txt', "r")
                    except OSError:
                        # print(OSError)
                        print("Could not find:" + ligghts_folder + '/' +
                              cpi_folder + str(j) + '/array.txt')
                    with collisions_file:
                        line = collisions_file.readline()
                        line = line[:-2]
                        string_values = line.split(", ")

                        a = []
                        for k in range(len(string_values)):
                            a.append(float(string_values[k]))

                        a = np.array(a)
                        print(np.max(a))
                        a = a * frames * dt * 100


                        print(np.max(a))

                        # x_first = np.linspace(1 * frames * dt * 100, 20 * frames * dt * 100, max_value)
                        # p = np.polyfit(x_first, np.log(values[:20]), 1)

                        # # Convert the polynomial back into an exponential
                        # a = np.exp(p[1])
                        # b = p[0]
                        # x_fitted = np.linspace(np.min(x_first), np.max(x_first), 100)
                        # y_fitted = a * np.exp(b * x_fitted)

                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111)
                        # print(a)
                        # counts, bins, bars = plt.hist(a, 
                        #                               bins=np.arange(
                        #     frames * dt * 100, 5000 * frames * dt * 100, 10 * frames * dt * 100),
                        #       ec="white", color="blue")
                        counts, bins, bars = plt.hist(a, 
                                                      bins=np.arange(
                            0.0, 5.0, 0.1),
                              ec="black", color="blue")

                        # print("bins")
                        # print(bins)
                        # print("bars")
                        # print(counts)

                        # test_count = 0
                        # for bin in bins:
                        #     if bin >= cutoff:
                        #         break
                        #     else:
                        #         test_count += 1

                        # summ = 0.0
                        # for k in range(test_count, len(counts)):
                        #     summ += counts[k] * bins[k]

                        # result = summ

                        # center_bins = []
                        # for k in range(0, len(bins)-1):
                        #     center_bins.append((bins[k] + bins[k+1])/2)

                        # count = 0
                        # for k in range(0, len(counts)):
                        #     if counts[k] > 1.0:
                        #         count += 1
                        #     else:
                        #         count -= 1
                        #         break
                        # if j > 3:
                        #     p = np.polyfit(center_bins[:count], np.log(counts[:count]), 1)
                        #     aa = np.exp(p[1])
                        #     bb = p[0]
                        #     x_fitted = np.linspace(
                        #         np.min(center_bins[:count]), np.max(center_bins[:count]), 100)
                        #     y_fitted = aa * np.exp(bb * x_fitted)   
                    # plt.plot(x_fitted, y_fitted)
                    # plt.plot(center_bins,counts)

                    #
                    # result = integrate.quad(
                    #     lambda x: aa * np.exp(bb * x) * x, 1.25, 4)

                    # print(file_dir_1[i] + file_dir_2[j])

                    # if j == 4:
                    #     # result[0]  use this instead of integration
                    #     vf_30_results.append(result)
                    # if j == 5:
                    #     vf_40_results.append(result)
                    # if j == 6:
                    #     vf_45_results.append(result)

                    # print(len(center_bins))
                    # print(len(counts))

                        plt.ylim([1, 40000])
                        plt.xlim([0.0,5.0])
                        plt.xticks([0.0,1.0,2.0,3.0, 4.0],fontsize=25)
                        plt.yticks(fontsize=25)
                        plt.yscale("log")
                        plt.xlabel("Interlocking Duration ($\.γ$t)", fontsize=34)
                        plt.ylabel("Number of Interlocks", fontsize=34)

                        plt.savefig(general_folder_name + "/histogram_{0}_{1}.pdf".format(simulation.root_folder_name, j),
                            bbox_inches="tight",
                        )
        #                 continue
        #             collision_counts = np.zeros((1800))
        #             k = 0
        #             per_particle_collision_count = np.zeros((simulation.particle_count[j], simulation.particle_count[j]))
        #             list_of_particles = []
        #             with collisions_file:
        #                 for line in collisions_file:
        #                     stringvalues = line.split()
        #                     if stringvalues[0] == "file":

        #                         collision_counts[k] = count
        #                         k = int(stringvalues[1])
        #                         count = 0
        #                         list_of_particles = []
        #                     elif float(stringvalues[0]) >= 0 and float(stringvalues[1]) >= 0:
        #                         if not int(stringvalues[0]) in list_of_particles:
        #                             count += 1
        #                             list_of_particles.append(int(stringvalues[0]))
        #                         if not int(stringvalues[1]) in list_of_particles:
        #                             count += 1
        #                             list_of_particles.append(int(stringvalues[1]))

        #                         value_one = int(stringvalues[0])
        #                         value_two = int(stringvalues[1])

        #                         if value_one < value_two:
        #                             per_particle_collision_count[value_one][value_two] += 1
        #                         else:
        #                             per_particle_collision_count[value_two][value_one] += 1

        # per_particle_total_time = np.zeros((particle_count[j]))
        # per_particle_redudant_count = np.zeros((particle_count[j]))

        # for k in range(particle_count[j]):
        #     for l in range(particle_count[j]):
        #         if per_particle_collision_count[k][l] > 0:
        #             per_particle_total_time[k] += per_particle_collision_count[k][l]
        #             per_particle_total_time[l] += per_particle_collision_count[k][l]
        #             per_particle_redudant_count[k] += 1
        #             per_particle_redudant_count[l] += 1

    # def log_normal_compare(self,use_fortran=False, use_liggghts=True, general_folder_name="", series_name="",volume_fraction=[7]):
    #     if general_folder_name == "":
    #         general_folder_name = "lognorm_compare_" + \
    #             self.simulations[0].root_folder_name
    #     if series_name == "":
    #         series_name = "lognorm_compare_" + \
    #             self.simulations[0].particletemplate.particle.file_shape_name

    #     try:
    #         os.makedirs(general_folder_name)
    #         import overlapping_circles as oc
    #     except OSError:
    #         # print(OSError)
    #         print("General folder name for lognorm_compare already exists")
    #     if use_liggghts:

    #         for i in range(volume_fraction[0], np.max(volume_fraction)+1):
    #             for simulation in self.simulations:


    def interlocking_time_graph(self, use_fortran=False, use_liggghts=True, general_folder_name="", series_name="", test_lowest_vf_count=0):
        import overlapping_circles as oc
        if general_folder_name == "":
            general_folder_name = "Interlocking_Histograms_" + \
                self.simulations[0].root_folder_name
        if series_name == "":
            series_name = "Interlocking_Histograms_" + \
                self.simulations[0].particletemplate.particle.file_shape_name

        try:
            os.makedirs(general_folder_name)
            import overlapping_circles as oc
        except OSError:
            # print(OSError)
            print("General folder name for Interlocking_Histograms already exists")
        total_times = []
        if use_liggghts:


            markers = [ "v", "x", "s", "*", "D"]
            colors = [
                    "#e1893f", "#D6B11F", "#91b851", "#52a0e0",  "#1F73B8"]
            # colors = ['Red', 'Orange', 'Pink',
            #           'Green', 'LightBlue', 'Blue', "Purple"]
            labelnames = [ "Curl 1",
                      "Curl 2", "Curl 3", "Curl 4", "Curl 5"]

            
            for (i,simulation) in enumerate(self.simulations):
                total_times.append([])
                for j in range(0, len(simulation.volume_fractions)):
                    # Assumes only one size radii per simulation
                    
                    frames = 50000#simulation.body_position_print_count[j]
                    if j <= 3:
                        frames = 5000

                    dt = simulation.delta_time * simulation.relaxationtime[j]
                   
                    ligghts_folder = 'liggghts_'+simulation.root_folder_name +"_"+ simulation.extra
                    cpi_folder = 'cpi_'+simulation.root_folder_name + \
                        '_'

                    try:
                        collisions_file = open(ligghts_folder + '/' + cpi_folder +
                                               str(j) + '/array.txt', "r")
                    except OSError:
                        # print(OSError)
                        print("Could not find:" + ligghts_folder + '/' +
                              cpi_folder + str(j) + '/array.txt')
                    with collisions_file:
                        line = collisions_file.readline()
                        line = line[:-2]
                        string_values = line.split(", ")

                        a = []
                        for k in range(len(string_values)):
                            a.append(float(string_values[k]))

                        a = np.array(a)
                        a = a * frames * dt * 100
                        total_normalizing_time = frames * dt * 1000

                        total_times[i].append(np.sum(a)/total_normalizing_time)
            
        plt.figure(1)
       
        for (i, simulation) in enumerate(self.simulations):
            plt.ylabel("Normalized Interlocking Time")
            plt.xlabel("Solid volume fraction (ν)")

            if use_liggghts:
               
                plt.plot(simulation.volume_fractions, total_times[i], color=colors[i], marker=markers[i], linewidth=0.8, linestyle='dotted', label=labelnames[i])
            
        plt.legend(loc='upper left')

        plt.savefig(general_folder_name + "/normalized_interlocking_time_{}.pdf".format(
            series_name), dpi=250)
        plt.yscale("log")
        plt.xscale("log")
        plt.savefig(general_folder_name + "/normalized_log_interlocking_time_{}.pdf".format(
            series_name), dpi=250)
        plt.clf()

        plt.figure(1)
       
        for (i, simulation) in enumerate(self.simulations):
            plt.ylabel("$(σ_{yy}) / (ρd_{v}^{2} γ^{2})$")
            plt.xlabel("Normalized Interlocking Time")

            if use_liggghts:
               
                plt.plot(total_times[i], simulation.l_normal_stress_ave, color=colors[i], marker=markers[i], linewidth=0.8, linestyle='dotted', label=labelnames[i])
            
        plt.legend(loc='upper left')
        plt.yscale("log")
        plt.savefig(general_folder_name + "/total_times_vs_stress_{}.pdf".format(
            series_name), dpi=250)
        plt.clf()




# # Just for fun color maps stuff
# def test_plot_color_maps():
#     N = 30
#     test_cmaps = ['Dark2', 'rainbow', 'Wistia', 'Set2']
#     # segmented_cmaps = [matplotlib.colors.ListedColormap(
#     #     plt.get_cmap(t)(np.linspace(0, 1, N))) for t in test_cmaps]
#     nrows = len(test_cmaps)
#     gradient = np.linspace(0, 1, 256)
#     gradient = np.vstack((gradient, gradient))
#     cmap_category = 'test'
#     cmap_list = test_cmaps
#     fig, axes = plt.subplots(nrows=nrows)
#     fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#     axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

#     for ax, name in zip(axes, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#         pos = list(ax.get_position().bounds)
#         x_text = pos[0] - 0.01
#         y_text = pos[1] + pos[3]/2.
#         fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axes:
#         ax.set_axis_off()

#     plt.show()
