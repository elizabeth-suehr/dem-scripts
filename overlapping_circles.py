#!/usr/bin/env python3
"""
Original idea by red Ants Aasma at
http://stackoverflow.com/a/1667789/2122529

The code was reformed from
https://gist.github.com/versusvoid/a86d1e6ed0c64c6c80b0a908ac9f1865
MadRunner<3
"""

import numpy as np
import numpy.random as rnd
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import collections
import sys

π = np.pi

# Params
xmax = 5e-4
rmax = 1
""" Allowable distance between to points error """
ε = 1e-12


class IntegerBoundedSet(object):
    def __init__(self, N):
        self.len = 0
        self.N = N
        self.set = [False] * N

    def __len__(self):
        return self.len

    def add(self, i):
        assert type(i) == int and i >= 0 and i < self.N
        if not self.set[i]:
            self.set[i] = True
            self.len += 1

    def discard(self, i):
        assert type(i) == int and i >= 0 and i < self.N
        if self.set[i]:
            self.set[i] = False
            self.len -= 1


def polygon_area(p):
    assert len(p) > 3
    """
    See http://mathworld.wolfram.com/PolygonArea.html for formula.
    Negation due CW direction.
    """
    return (
        -sum(
            [
                p1[0] * p2[1] - p2[0] * p1[1]
                for p1, p2 in zip(
                    p, itertools.islice(itertools.cycle(p), 1, len(p) + 1)
                )
            ]
        )
        / 2
    )


""" Computes cartesian coordinates of intersection by circle center and angle """


def edge(n, φ, nodes, radiuses):
    return nodes[n] + radiuses[n] * np.array([np.cos(φ), np.sin(φ)])


def normalize(α):
    while α > π:
        α -= 2 * π
    while α < -π:
        α += 2 * π
    """
    Make sure every circle has same angle value
    for every other circle, when more than two 
    circles intersect in one point.
    """
    return round(α, 9)


def getArea(nodes, radiuses, graph_test=False):
    # '''
    # Make some circles intersect in one point.
    # '''
    # K = 4
    # ns = rnd.permutation(N)[:K]
    # p = rnd.uniform(-xmax, +xmax, 2)
    # nodes[ns] = p + rnd.uniform(-rmax/2, +rmax/2, (K, 2))
    # radiuses[ns] = np.linalg.norm(nodes[ns] - p, axis=1)
    N = len(radiuses)

    """
    Per circle intersections.
    """
    intersections = [[] for i in range(N)]
    """
    Flags marking circles fully within other
    circles.
    """
    covered = [False] * N
    """
    Intersection point is 'left' if it is by left side
    when looking from this circle center to other. Left
    intersection is always after right in CCW direction.
    """
    Intersection = collections.namedtuple("Intersection", "n, φ, left, φ2")

    for n1, n2 in itertools.combinations(range(N), 2):
        if covered[n1] or covered[n2]:
            continue

        p1 = nodes[n1]
        p2 = nodes[n2]
        d = np.linalg.norm(p2 - p1)
        r1 = radiuses[n1]
        r2 = radiuses[n2]
        """ If one of circles fully within other circle """
        if max(r1, r2) + ε >= d + min(r1, r2):
            if r1 < r2:
                covered[n1] = True
                intersections[n1] = None
            else:
                covered[n2] = True
                intersections[n2] = None

            continue

        if d + ε >= r1 + r2:
            continue

        """ Refer to http://mathworld.wolfram.com/Circle-CircleIntersection.html for math """
        t = d**2 - r2**2 + r1**2
        x = t / (2.0 * d)
        y = np.sqrt(4.0 * d**2 * r1**2 - t**2) / (2.0 * d)

        """
        φ1 and φ2 are computed in coordinate system where
        circle `n1` is at (a, b) and circle `n2` is at (a + d, b)
        for arbitrary `a` and `b`.
        This system is rotated from our by angle θ. We use it to
        adjust final intersection angle, which can be than used to
        compute cartesian coordinates of intersection and correctly
        order intersection points on circle.
        """

        φ1 = np.arctan2(y, x)
        φ2 = np.arctan2(y, d - x)
        θ = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        n1_n2_φ1 = normalize(φ1 + θ)
        n1_n2_φ2 = normalize(-φ1 + θ)
        intersections[n1].append(Intersection(n2, n1_n2_φ1, True, n1_n2_φ2))
        intersections[n1].append(Intersection(n2, n1_n2_φ2, False, n1_n2_φ1))

        n2_n1_φ1 = normalize(-π + (φ2 + θ))
        n2_n1_φ2 = normalize(π - (φ2 - θ))
        intersections[n2].append(Intersection(n1, n2_n1_φ1, True, n2_n1_φ2))
        intersections[n2].append(Intersection(n1, n2_n1_φ2, False, n2_n1_φ1))

    patches = []
    for n in range(N):
        if covered[n]:
            continue

        if len(intersections[n]) == 0:
            patches.append(n)
            continue

        intersections[n].sort(
            key=lambda iss: (iss.φ, iss.φ2 + (π * 2 if iss.φ2 < iss.φ else 0))
        )

        """ Flags whether this intersections lies inside other circle """
        in_between = [False] * len(intersections[n])

        """
        We need to mark intersections that lie inside other circles.
        It is done in two CCW passes over intersections array.
        Thus we enter (open) circle on right intersection and close
        on left.
        """
        assert len(intersections[n]) % 2 == 0
        opened = IntegerBoundedSet(N)
        i = 0

        while i < len(intersections[n]):
            """If we met closing intersection"""
            if intersections[n][i].left:
                """Drop circle from set of opened"""
                opened.discard(intersections[n][i].n)

                """
                If there are other open circles,
                then mark this intersection.
                """
                if len(opened) > 0:
                    in_between[i] = True
            else:
                """
                Otherwise we first have to check
                for opened circles.
                """
                if len(opened) > 0:
                    in_between[i] = True

                """ ... and than open new one """
                opened.add(intersections[n][i].n)

            i += 1

        """ Second pass closes circles that are still opened """
        i = 0
        while len(opened) != 0:
            if intersections[n][i].left:
                opened.discard(intersections[n][i].n)

            if len(opened) != 0:
                in_between[i] = True

            i += 1

        """ Filter intersections that lay inside other circles """
        intersections[n] = [
            iss for flag, iss in zip(in_between, intersections[n]) if not flag
        ]
        if len(intersections[n]) == 0:
            continue

        """ Add circular parts """
        assert len(intersections[n]) > 1
        i = 0 if intersections[n][0].left else 1
        while i < len(intersections[n]):
            patches.append(
                (
                    n,
                    intersections[n][i].φ,
                    intersections[n][(i + 1) % len(intersections[n])].φ,
                )
            )
            i += 2

    polygons = []
    for n1 in range(N):
        if covered[n1] or len(intersections[n1]) == 0:
            continue

        while len(intersections[n1]) > 0:

            """
            We select any left intersection, than build polygon
            going CW from vertex to vertex until return to first circle.
            """

            assert len(intersections[n1]) > 1, intersections[n1]
            """ Find any left intersection. """
            i, iss = next(p for p in enumerate(intersections[n1]) if p[1].left)
            intersections[n1].pop(i)

            polygon = []
            polygon.append(edge(n1, iss.φ, nodes, radiuses))

            prev_n = n1
            current_n = iss.n
            while current_n != n1:
                try:
                    """
                    Find matching right intersection in adjoint circle.
                    Can be done faster by clever indexing.
                    """
                    i = next(
                        p[0]
                        for p in enumerate(intersections[current_n])
                        if p[1].n == prev_n and not p[1].left
                    )
                except:
                    print(*intersections, file=sys.stderr, sep="\n")
                    print(n1, prev_n, current_n, iss, file=sys.stderr)
                    raise

                j = (i + 1) % len(intersections[current_n])
                assert j != i and intersections[current_n][j].left
                """ Find next intersection """
                iss = intersections[current_n][j]

                polygon.append(nodes[current_n])
                polygon.append(edge(current_n, iss.φ, nodes, radiuses))

                """ Drop used pair of intersections """
                for k in sorted([i, j], reverse=True):
                    intersections[current_n].pop(k)

                prev_n = current_n
                current_n = iss.n

            """ Drop last matching intersection in first circle """
            for i, iss in enumerate(intersections[n1]):
                if not iss.left and iss.n == prev_n:
                    intersections[n1].pop(i)
                    break

            polygon.append(nodes[n1])
            polygons.append(polygon)

    axes = plt.axes()
    if graph_test:
        for i in range(N):
            axes.annotate(str(i), (nodes[i, 0], nodes[i, 1] + 0.1))

    total_area = 0.0
    for p in patches:
        area = None
        if type(p) == int:
            area = π * radiuses[n] ** 2
            # print('Circle', p, 'gives', area, 'area')
            if graph_test:
                axes.add_patch(
                    plt.Circle(
                        (nodes[p, 0], nodes[p, 1]),
                        radius=radiuses[p],
                        alpha=0.2,
                        color="blue",
                    )
                )
        else:
            n, φ1, φ2 = p
            if φ2 < φ1:
                area = (2 * π + φ2 - φ1) * radiuses[n] ** 2 / 2
            else:
                area = (φ2 - φ1) * radiuses[n] ** 2 / 2
            # print('Circle', n, "'s part from", φ1,
            #       'to', φ2, 'gives', area, 'area')
            if graph_test:
                axes.add_patch(
                    Wedge(
                        (nodes[n, 0], nodes[n, 1]),
                        radiuses[n],
                        φ1 * 180 / π,
                        φ2 * 180 / π,
                        color="green",
                        alpha=0.2,
                    )
                )

        total_area += area

    for i, p in enumerate(polygons):
        area = polygon_area(p)
        # print("Polygon's #", i, ' area is ', area, sep='')
        total_area += area
        # axes.add_patch(plt.Polygon(np.array(p), alpha=0.2, color="red"))

    # print('Total area:', total_area)
    if graph_test:
        axes.scatter(nodes[:, 0], nodes[:, 1])
        axes.set_aspect("equal", "datalim")
        plt.xlim(-xmax, xmax)
        plt.ylim(-xmax, xmax)
        plt.show()

    return total_area


# print(getArea(nodes,radiuses))
