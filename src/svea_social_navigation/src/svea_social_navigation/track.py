from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from casadi import vertcat


@dataclass
class Arc:
    """
    Creates an arc section based on curvature, where curvature radius is 1/curvature
    Args:
        length: Length of arc
        curvature:  > 0 left turn | = 0 straight | < 0 right turn
    """
    length: float = 0
    curvature: float = 0

    @classmethod
    def from_circle_segment(cls, radius: float, angle: float, deg: bool = True) -> 'Arc':
        """
        Create the arc from a circle segment representation.
        Args:
            radius: Size of circle, must be non-zero. > 0 left turn, < 0 right turn.
            angle: Angle that sets the circle, must be non-zero.
            deg: angle is in degrees if true, otherwise angle is in radians.
        """
        assert radius != 0, 'radius must be non-zero'
        assert angle != 0, 'angle must be non-zero'
        if deg:
            angle = np.deg2rad(angle)
        length = angle*np.abs(radius)
        curvature = 1/radius
        return Arc(length, curvature)

    @property
    def radius(self) -> float:
        return float("inf") if self.curvature == 0 else 1 / self.curvature


@dataclass
class TrackPoint:
    x_s: float
    y_s: float
    s: float
    curvature: float
    phi_s: float
    s_0_arc: float  # position of first point in arc along the s-direction
    # angle of first point in arc (= angle of last point in previous arc)
    phi_0_arc: float

    @property
    def as_tuple(self):
        return (self.x_s, self.y_s, self.s, self.curvature, self.phi_s, self.s_0_arc, self.phi_0_arc)


class Track:
    L_LANE = 0.5
    S_LANE_OBS = 5

    def __init__(self, arcs: List[Arc], x_s0=0, y_s0=0, phi_s0=0, flip=False, diff=0, POINT_DENSITY=1000):
        self.arcs = arcs
        self.x_s0 = x_s0
        self.y_s0 = y_s0
        self.phi_s0 = phi_s0
        self.e_shift = 0
        self.diff = diff
        self.flip = flip
        self.POINT_DENSITY = POINT_DENSITY  # [#points/meter]

        self.points, self.points_array, self._track_length = self._create_track()
        self.R, self.T = self._create_transformation_components()

        self.next_track = None

        self.road_users = {}

    def update_road_user(self, name, state):
        dist = np.linalg.norm(self.points_array[:2].T - np.array([state.x, state.y]),
                              axis=-1)
        if dist.min() < 0.4:
            self.road_users[name] = state
        elif name in self.road_users:
            del self.road_users[name]

    def connects_to(self, other):
        self.next_track = other

    def _create_transformation_components(self):
        R = np.array([[np.cos(self.phi_s0), -np.sin(self.phi_s0)],
                      [np.sin(self.phi_s0),  np.cos(self.phi_s0)]])
        T = np.array([[self.x_s0],
                      [self.y_s0]])
        return R, T

    def _create_track(self) -> Tuple[List[TrackPoint], np.ndarray, float]:
        phi_s0 = np.arctan2(np.sin(self.phi_s0), np.cos(self.phi_s0))
        x_s0 = self.x_s0
        y_s0 = self.y_s0
        points = [TrackPoint(
            x_s0, y_s0, 0, self.arcs[0].curvature, phi_s0, 0, phi_s0)]
        # points = [TrackPoint(0, 0, 0, self.arcs[0].curvature, 0, self.phi_s0)]
        points_array = np.array([points[0].as_tuple]).T
        track_length = 0
        s_0_arc = 0
        phi_0_arc = phi_s0
        for arc in self.arcs:
            length = arc.length
            track_length += length

            n_points = int(length * self.POINT_DENSITY)
            delta_L = length / n_points
            delta_theta = delta_L * arc.curvature

            for _ in range(n_points):
                # get last point in track
                prev = points[-1]
                # transform from last point in track to the next point
                x_s_next = prev.x_s - delta_L * np.cos(np.pi - prev.phi_s)
                y_s_next = prev.y_s + delta_L * np.sin(np.pi - prev.phi_s)
                s_next = prev.s + delta_L
                phi_s_next = (prev.phi_s + delta_theta) % (2 * np.pi)
                s_0_arc_next = s_0_arc
                phi_0_arc_next = phi_0_arc
                # create next point based on transformed values
                next_point = TrackPoint(
                    x_s_next, y_s_next, s_next, arc.curvature, phi_s_next, s_0_arc_next, phi_0_arc_next)
                # add next point to track
                points.append(next_point)
                points_array = np.hstack(
                    (points_array, np.array([next_point.as_tuple]).T))
            # update start position for the next arc section
            s_0_arc += length
            # update start angle for the next arc section
            phi_0_arc = points[-1].phi_s
        for i, point in enumerate(points):
            point.x_s -= self.e_shift * np.sin(point.phi_s)
            point.y_s += self.e_shift * np.cos(point.phi_s)
            points_array[0, i] -= self.e_shift * np.sin(point.phi_s)
            points_array[1, i] += self.e_shift * np.cos(point.phi_s)

        if self.flip:
            points_array[2, :] = points_array[2, ::-1]
            points_array[5, :] = track_length - points_array[5, :]
            points_array[6, :] = points_array[6, :] - np.pi
            for i, point in enumerate(points):
                point.s = points_array[2, i]
                point.s_0_arc = points_array[5, i]
                point.phi_0_arc = points_array[6, i]
            points_array = points_array[:, ::-1]
            points.reverse()

        return points, points_array, track_length

    @property
    def length(self):
        return self._track_length

    @property
    def cartesian(self):
        return self.points_array[0, :], self.points_array[1, :]

    def __iter__(self):
        for x, y in zip(*self.cartesian):
            yield (x, y)
        if self.next_track is not None:
            yield from self.next_track

    def to_global(self, s: float, e: float = 0) -> Tuple[float, float]:
        """
        to_global takes the traveled distance along track, s, and
        the lateral distance from track, e, and returns the coordinates of the
        corresponding point in the global frame.

        Args:
            s (float): position along the track
            e (flaot): lateral distance from track

        Returns:
            position ([float,float]): x and y positions in the global frame
        """
        point_idx = np.argmin(np.abs(self.points_array[2, :] - s))
        x_s = self.points[point_idx].x_s
        y_s = self.points[point_idx].y_s
        phi_s = self.points[point_idx].phi_s

        x = x_s - e * np.sin(phi_s)
        y = y_s + e * np.cos(phi_s)

        return x, y

    def to_local(self, x: float, y: float) -> Tuple[float, float]:
        """
        to_local takes the position in x and y in the global frame and
        returns the coordinates of the corresponding s, e points in the
        local frame.

        Args:
            x (float): x-position in the global frame
            y (float): y-position in the global frame

        Returns:
            s (float): position along the track
            e (flaot): lateral distance from track

            position ([float,float]): traveled distance along track, s, and lateral distance from track, e.
        """
        point_idx = np.argmin(
            np.linalg.norm(self.points_array[:2, :] - vertcat(x, y), axis=0))
        x_s = self.points[point_idx].x_s
        y_s = self.points[point_idx].y_s
        phi_s = self.points[point_idx].phi_s

        # take the computation of e which corresponds to division by the largest number to avoid division by zero, or near zero values
        e_idx = np.argmax((abs(np.sin(phi_s)), abs(np.cos(phi_s))))
        if e_idx == 0:
            e = (x_s - x) / np.sin(phi_s)
        else:
            e = (y - y_s) / np.cos(phi_s)
        s = self.points[point_idx].s

        return s, e