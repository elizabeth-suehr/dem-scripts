from ast import Num
import os
from re import X
import sys
import numpy as np
import xml.etree.ElementTree as ET
import operator
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import copy
from numba import jit
from numba.experimental import jitclass
from numba import int32, float32

# Holds consituient sphere data
spec = [
    ("x", float32),
    ("y", float32),
    ("z", float32),
]


@jitclass(spec)
class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __copy__(self):
        return type(self)(self.x, self.y, self.z)
    


# Holds composite particle data
class Particle:
    # def __init__(self, id):
    #     self.points = []
    #     self.id = id

    def __init__(self, id, points):
        self.points = points
        self.id = id

    def add_point(self, point):
        self.points.append(point)

    # Defines the AABB for the particle
    def redefine_min_max(self, radius):
        self.max_x = max([x.x for x in self.points]) + radius
        self.max_y = max([x.y for x in self.points]) + radius
        self.max_z = max([x.z for x in self.points]) + radius
        self.min_x = min([x.x for x in self.points]) - radius
        self.min_y = min([x.y for x in self.points]) - radius
        self.min_z = min([x.z for x in self.points]) - radius

        self.width =  self.max_x - self.min_x
        self.depth =  self.max_z - self.min_z
        self.height = self.max_y - self.min_y

    # Sort algorithm along z axis
    def __lt__(self, other):
        return self.min_y < other.min_y
    
    def __copy__(self):
        return Particle(self.id, self.points)

@jit(nopython=True)
def check_collision(particle_1, particle_2):


    ### radius is 1 but problem can be scaled
    R = 0.13924170e-3

    # Defines three points of the triangle to check collisions
    aaaa = np.array([particle_2[0].x, particle_2[0].y, particle_2[0].z])
    bbbb = np.array([particle_2[2].x, particle_2[2].y, particle_2[2].z])
    cccc = np.array([particle_2[4].x, particle_2[4].y, particle_2[4].z])
    
    # Goes over each sphere is the first particle and checks the triangle made from the second particle
    is_collision = False
    for point in particle_1:
        
        A =  np.array([point.x,point.y,point.z]) - aaaa
        B =  np.array([point.x,point.y,point.z]) - bbbb
        C =  np.array([point.x,point.y,point.z]) - cccc
        # """
        # Most simple check:
        # is one of the vertices indside
        # """
        # print(np.linalg.norm( A ), np.linalg.norm( A ) < R )
        # print(np.linalg.norm( B ), np.linalg.norm( B ) < R )
        # print(np.linalg.norm( C ), np.linalg.norm( C ) < R )

        if np.linalg.norm( A ) < R or np.linalg.norm( B ) < R or np.linalg.norm( C ) < R:
            is_collision = True
            # return is_collision

        # print(is_collision)
        # """
        # checking if one edge cuts the sphere
        # this uses simple derivatives of the distance function
        # """
        for F,G in [ ( B, A ), (C, B), (A, C)]:
            a =  F - G
            s = -np.dot( a, G )/ np.dot( a, a )
            # print("s: ", s, s > 0 and s < 1)
            d  = np.linalg.norm( G + s * a )
            # print("d: ", d,  d < R)
            ### if both are true, it is cutting
            # print("---------")
            if s > 0 and s < 1 and d < R:
                is_collision = True
                # return is_collision

        # """
        # checking if the sphere cuts the area
        # e.g in the extreme case of (but not restricted to) a sphere
        # passing through
        # """
        a = B - A
        c = C - A
        aa = np.dot( a, a)
        cc = np.dot( c, c)
        ac = np.dot( a, c)
        aA = np.dot( a, A)
        cA = np.dot( c, A)

        mi = np.array([[ cc, -ac ],[ -ac, aa ]])
        mi /= ( aa * cc - ac**2 ) ### div by det

        st = np.dot( mi, np.array([ -aA, -cA ]) )
        s=st[0]
        t=st[1]
        P = A + s * a + t * c

        # """
        # If this is larger than R we can stop here
        # if otherwise we detect if P inside triangle by repeating the stuff 
        # above with respect to B
        # """
        a2 = A - B
        c2 = C - B
        aa2 = np.dot( a2, a2 )
        cc2 = np.dot( c2, c2 )
        ac2 = np.dot( a2, c2 )
        aB2 = np.dot( a2, B )
        cB2 = np.dot( c2, B )

        mi2 = np.array( [[ +cc2, -ac2 ] ,[ -ac2, +aa2 ]])
        mi2 /= ( aa2 * cc2 - ac2**2 ) 
        uv = np.dot(mi2, np.array([ -aB2, -cB2 ]) )
        u = uv[0]
        v = uv[1]
        P2 = B + u * a2 + v * c2
        # print("must be identical")
        # print(P, np.linalg.norm( P ) < R)
        # print(P2, np.linalg.norm( P2 ) < R)


        # if np.linalg.norm( P ) < R and np.linalg.norm( P2 ) < R:
        #     collision = True


        # print("is inside if all 4 are positive")
        # print(s, t, u, v)

        if s > 0 and t > 0 and u > 0 and v > 0 and np.linalg.norm( P ) < R and np.linalg.norm( P2 ) < R:
            is_collision = True
            # return is_collision
        # print(is_collision)

        ### finally some plotting
        # verts = [ [ A, B, C  ] ]
        # srf = Poly3DCollection( verts, alpha=.9, facecolor='#800000' )
        # col='b'
        if is_collision:
            # col = 'g'
            # fig = plt.figure()
            # ax = fig.add_subplot( 1, 1, 1, projection='3d' )
            # ax.plot_wireframe( x, y, z, color=col )
            # # ax.plot( [ 0, P[0] ], [ 0, P[1] ], [ 0, P[2] ] )
            # ax.add_collection3d(srf)
            # plt.show()
            return True

    # Return if no collision with first particle
    # if false_count == 5:
    #     return True

    # Define collision checking triangle from first particle to check each sphere of second particle
    aaaa = np.array([particle_1[0].x, particle_1[0].y, particle_1[0].z])
    bbbb = np.array([particle_1[2].x, particle_1[2].y, particle_1[2].z])
    cccc = np.array([particle_1[4].x, particle_1[4].y, particle_1[4].z])

    # cycle through each sphere to see if any from particle 2 collide with triangle 1
    for point in particle_2:
        A =  np.array([point.x,point.y,point.z]) - aaaa
        B =  np.array([point.x,point.y,point.z]) - bbbb
        C =  np.array([point.x,point.y,point.z]) - cccc
        
        # """
        # Most simple check:
        # is one of the vertices indside
        # """
        # print(np.linalg.norm( A ), np.linalg.norm( A ) < R )
        # print(np.linalg.norm( B ), np.linalg.norm( B ) < R )
        # print(np.linalg.norm( C ), np.linalg.norm( C ) < R )

        if np.linalg.norm( A ) < R or np.linalg.norm( B ) < R or np.linalg.norm( C ) < R:
            is_collision = True
            # return is_collision

        # print(is_collision)
        # """
        # checking if one edge cuts the sphere
        # this uses simple derivatives of the distance function
        # """
        for F,G in [ ( B, A ), (C, B), (A, C)]:
            a =  F - G
            s = -np.dot( a, G )/ np.dot( a, a )
            # print("s: ", s, s > 0 and s < 1)
            d  = np.linalg.norm( G + s * a )
            # print("d: ", d,  d < R)
            ### if both are true, it is cutting
            # print("---------")
            if s > 0 and s < 1 and d < R:
                is_collision = True
                # return is_collision

       # """
        # checking if the sphere cuts the area
        # e.g in the extreme case of (but not restricted to) a sphere
        # passing through
        # """
        a = B - A
        c = C - A
        aa = np.dot( a, a)
        cc = np.dot( c, c)
        ac = np.dot( a, c)
        aA = np.dot( a, A)
        cA = np.dot( c, A)

        mi = np.array([[ cc, -ac ],[ -ac, aa ]])
        mi /= ( aa * cc - ac**2 ) ### div by det

        st = np.dot( mi, np.array([ -aA, -cA ]) )
        s=st[0]
        t=st[1]
        P = A + s * a + t * c

        # """
        # If this is larger than R we can stop here
        # if otherwise we detect if P inside triangle by repeating the stuff 
        # above with respect to B
        # """
        a2 = A - B
        c2 = C - B
        aa2 = np.dot( a2, a2 )
        cc2 = np.dot( c2, c2 )
        ac2 = np.dot( a2, c2 )
        aB2 = np.dot( a2, B )
        cB2 = np.dot( c2, B )

        mi2 = np.array( [[ +cc2, -ac2 ] ,[ -ac2, +aa2 ]])
        mi2 /= ( aa2 * cc2 - ac2**2 ) 
        uv = np.dot(mi2, np.array([ -aB2, -cB2 ]) )
        u = uv[0]
        v = uv[1]
        P2 = B + u * a2 + v * c2
        # print("must be identical")
        # print(P, np.linalg.norm( P ) < R)
        # print(P2, np.linalg.norm( P2 ) < R)


        # if np.linalg.norm( P ) < R and np.linalg.norm( P2 ) < R:
        #     collision = True


        # print("is inside if all 4 are positive")
        # print(s, t, u, v)

        if s > 0 and t > 0 and u > 0 and v > 0 and np.linalg.norm( P ) < R and np.linalg.norm( P2 ) < R:
            is_collision = True
            # return is_collision
        # print(is_collision)
        col='b'
        if is_collision:
        #     col = 'g'
        #     fig = plt.figure()
        #     ax = fig.add_subplot( 1, 1, 1, projection='3d' )
        #     ax.plot_wireframe( x, y, z, color=col )
        #     # ax.plot( [ 0, P[0] ], [ 0, P[1] ], [ 0, P[2] ] )
        #     ax.add_collection3d(srf)
        #     plt.show()
            return True
    return is_collision
        


curl_directory = "curl_3/"
File_directory = ["vf_5/"]

start_count = [61000000]
# 61200000, , 61000000, 61000000, 22750000

for loop in range(len(File_directory)):

    Files = []
    # Positions = np.empty((141,10000,3))
    particle_list = []
    # collisions_array = np.empty(shape=(0, 3))
    original_stdout = sys.stdout

    radius = 0.13924170e-3

    domain = [0.0072429, 0.0072429, 0.0036215]

    # Load only 60 files of data from the vtp data
    for i in range(1800):
        filename = (
            "vtp_files/"
            + curl_directory
            + File_directory[loop]
            + str(start_count[loop] + i * 2000)
            + "bal.vtp"
        )
        with open(os.path.join("./", filename)) as f:
            fileString = f.read()
            Files.append(ET.fromstring(fileString))

        empty_list = []
        particle_list.append(empty_list)
        for child in Files[i][0][0][0]:
            Text = child.text
            lines = Text.split("\n")

            lines = lines[1:-1]

            particle_count = 0
            particle = Particle(particle_count, [])
            count = 0
            for j in range(len(lines)):
                values = lines[j].split()
                point = Point(float(values[0]), float(values[1]), float(values[2]))
                particle.add_point(point)
                count += 1
                if count > 4:
                    count = 0
                    particle_list[i].append(particle)
                    particle_count += 1
                    particle = Particle(particle_count, [])
        f.close()

    # Loop for removing unwanted particles and checking collisions
    particles_to_remove = []
    particles_along_boundary = []
    for i in range(len(particle_list)):

        particles_along_boundary.append([])

        print("File: " + str(i))
        f = open(
            "./collision_check/"
            + curl_directory
            + File_directory[loop]
            + str(i)
            + "_col.data",
            "a",
        )

        # Get every particle that is cut off by domain edge
        for j in range(len(particle_list[i])):
            particle_list[i][j].redefine_min_max(radius)

            width = particle_list[i][j].max_x - particle_list[i][j].min_x
            height = particle_list[i][j].max_y - particle_list[i][j].min_y
            depth = particle_list[i][j].max_z - particle_list[i][j].min_z

            if width > 0.0072429 - radius or height > 0.0072429 - radius or depth > 0.0036215 - radius:
                # print(
                #         str(particle_list[i][j].id)
                #         + " "
                #         + str(particle_list[i][j].width) + " "
                #         + str(particle_list[i][j].depth))
                
                particles_to_remove.append(particle_list[i][j])
                if width > 0.0072429 - radius and height < 0.0072429 - radius:
                    if depth > 0.0036215 - radius:
                        temp_1 = copy.copy(particle_list[i][j])
                        temp_2 = copy.copy(particle_list[i][j])
                        temp_3 = copy.copy(particle_list[i][j])
                        temp_4 = copy.copy(particle_list[i][j])

                        for point in temp_1.points:
                            if point.x > domain[0] / 2.0:
                                point.x -= domain[0]
                            if point.z > domain[2] / 2.0:
                                point.z -= domain[2]
                        for point in temp_2.points:
                            if point.x < domain[0] / 2.0:
                                point.x += domain[0]
                            if point.z < domain[2] / 2.0:
                                point.z += domain[2]
                        for point in temp_3.points:
                            if point.x > domain[0] / 2.0:
                                point.x -= domain[0]
                            if point.z < domain[2] / 2.0:
                                point.z += domain[2]
                        for point in temp_4.points:
                            if point.x < domain[0] / 2.0:
                                point.x += domain[0]
                            if point.z > domain[2] / 2.0:
                                point.z -= domain[2]


                        
                        temp_1.redefine_min_max(radius)
                        temp_2.redefine_min_max(radius)
                        temp_3.redefine_min_max(radius)
                        temp_4.redefine_min_max(radius)

                        if temp_1.width < domain[0] / 2.0 and temp_1.depth < domain[2]:
                            particles_along_boundary[i].append(temp_1)
                        if temp_2.width < domain[0] / 2.0 and temp_2.depth < domain[2]:
                            particles_along_boundary[i].append(temp_2)
                        if temp_3.width < domain[0] / 2.0 and temp_3.depth < domain[2]:
                            particles_along_boundary[i].append(temp_3)
                        if temp_4.width < domain[0] / 2.0 and temp_4.depth < domain[2]:
                            particles_along_boundary[i].append(temp_4)
                    else:
                        temp_1 = copy.copy(particle_list[i][j])
                        temp_2 = copy.copy(particle_list[i][j])
                        for point in temp_1.points:
                            if point.x > domain[0] / 2.0:
                                point.x -= domain[0]
                        for point in temp_2.points:
                            if point.x < domain[0] / 2.0:
                                point.x += domain[0]
                        temp_1.redefine_min_max(radius)
                        temp_2.redefine_min_max(radius)
                        if temp_1.width < domain[0] / 2.0 and temp_1.depth < domain[2]:
                            particles_along_boundary[i].append(temp_1)
                        if temp_2.width < domain[0] / 2.0 and temp_2.depth < domain[2]:
                            particles_along_boundary[i].append(temp_2)
                elif depth > 0.0036215 - radius and height < 0.0072429 - radius:
                    temp_1 = copy.copy(particle_list[i][j])
                    temp_2 = copy.copy(particle_list[i][j])
                    for point in temp_1.points:
                        if point.z > domain[2] / 2.0:
                            point.z -= domain[2]
                    for point in temp_2.points:
                        if point.z < domain[2] / 2.0:
                            point.z += domain[2]
                    temp_1.redefine_min_max(radius)
                    temp_2.redefine_min_max(radius)
                    if temp_1.width < domain[0] / 2.0 and temp_1.depth < domain[2]:
                            particles_along_boundary[i].append(temp_1)
                    if temp_2.width < domain[0] / 2.0 and temp_2.depth < domain[2]:
                        particles_along_boundary[i].append(temp_2)

        # Remove every particle
        # print(len(particles_to_remove))
        for j in range(len(particles_to_remove) - 1, -1, -1):
            particle_list[i].remove(particles_to_remove[j])
        del particles_to_remove[:]
        particles_to_remove = []


        particle_list[i].sort()

        for j in range(len(particles_along_boundary[i])):
            particles_along_boundary[i][j].redefine_min_max(radius)
        
        particles_along_boundary[i].sort()

        # Sweet and prune algorithm along z-axis
        for j in range(len(particle_list[i])):
            for k in range(j + 1, len(particle_list[i])):
                if particle_list[i][j].max_y < particle_list[i][k].min_y:
                    break
                sep = check_collision(
                    particle_list[i][j].points, particle_list[i][k].points
                )

                if  sep:
                    f.write(
                        str(particle_list[i][j].id)
                        + " "
                        + str(particle_list[i][k].id)
                        + "\n"
                    )
                    print(
                        str(particle_list[i][j].id) + " " + str(particle_list[i][k].id)
                    )

        # for j in range(len(particles_along_boundary[i])):
        #     for k in range(j + 1, len(particles_along_boundary[i])):

        #         if (
        #             particles_along_boundary[i][j].id
        #             == particles_along_boundary[i][k].id
        #         ):
        #             continue
        #         if (
        #             particles_along_boundary[i][j].max_z
        #             < particles_along_boundary[i][k].min_z
        #         ):
        #             break
        #         sep = check_collision(
        #             particles_along_boundary[i][j].points,
        #             particles_along_boundary[i][k].points,
        #         )

        #         if  sep:
        #             f.write(
        #                 str(particles_along_boundary[i][j].id)
        #                 + " "
        #                 + str(particles_along_boundary[i][k].id)
        #                 + "\n"
        #             )
        #             print(
        #                 str(particles_along_boundary[i][j].id)
        #                 + " "
        #                 + str(particles_along_boundary[i][k].id)
        #             )

        for j in range(len(particle_list[i])):
            for k in range(len(particles_along_boundary[i])):
                if particle_list[i][j].max_y < particles_along_boundary[i][k].min_y:
                    break
                sep = check_collision(
                    particle_list[i][j].points, particles_along_boundary[i][k].points
                )

                if  sep:
                    f.write(
                        str(particle_list[i][j].id)
                        + " "
                        + str(particles_along_boundary[i][k].id)
                        + "\n"
                    )
                    print(
                        str(particle_list[i][j].id)
                        + " "
                        + str(particles_along_boundary[i][k].id)
                        + "along_boundary "
                        + str(particles_along_boundary[i][k].width) + " "
                        + str(particles_along_boundary[i][k].depth)
                    )

        f.close()
        # for point in particle_list[i][j].points:
        #     print(str(point.x)+" "+str(point.y)+" "+str(point.z))
        # for point in particle_list[i][k].points:
        #     print(str(point.x)+" "+str(point.y)+" "+str(point.z))
