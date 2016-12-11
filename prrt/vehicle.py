from abc import ABCMeta, abstractmethod, abstractproperty
from prrt.primitive import PointR2, PoseR2S1
from typing import Tuple, List
import numpy as np
import math
from prrt.grid import WorldGrid


class Vehicle(metaclass=ABCMeta):
    """
    Manage a vehicle with a specific car_vertices(polygon) and kinematic constraints
    """

    def __init__(self, v_max: float, alpha_max: float, a_max: float, w_max: float):
        """

        :type v_max: float maximum absolute linear velocity - unit [m/s]  TODO : check this
        :type alpha_max: float maximum absolute steering angle - unit [rad]
        :type a_max: float maximum absolute linear velocity  ---- MB: what's the difference between a_max and v_max ?
                                                                  in the equations you have used always v_max
        :type w_max: float maximum absolute rotational velocity - unit [rad/2]
        """
        self.v_max = v_max
        self.alpha_max = alpha_max
        self.a_max = a_max
        self.w_max = w_max
        self.shape = ()  # type: Tuple[PointR2]
        self.alpha_resolution = np.deg2rad(5)   # MB: should this be a PTG parameter ??
        self.pose = PoseR2S1(0., 0., 0.)

    def set_vertices(self, polygon: Tuple[PointR2]):
        """
        Sets the vertices of vehicle. Each vertex is a 2D point (x,y)
        vertices are relative to the robot center
        """
        self.shape = polygon

    def get_vertex(self, idx: int) -> PointR2:
        return self.shape[idx]

    def get_vertices_at_pose(self, pose: PoseR2S1) -> List[PointR2]:
        result = []
        for vertex in self.shape:
            point = PointR2()
            point.x = pose.x + math.cos(pose.theta) * vertex.x - math.sin(pose.theta) * vertex.y
            point.y = pose.y + math.sin(pose.theta) * vertex.x + math.cos(pose.theta) * vertex.y
            result.append(point)
        return result

    @abstractmethod
    def execute_motion(self, K: float, w: float, dt: float) -> PoseR2S1:
        pass

    @abstractmethod
    def plot(self, axes, world: WorldGrid, pose: PoseR2S1, color='b'):
        pass

    @abstractproperty
    @property
    def phi(self):
        return float('nan')

    @abstractproperty
    @phi.setter
    def phi(self, angle: float):
        pass


class Car(Vehicle):
    """
    A non-articulated vehicle with a rectangle shape defined by four vertices
    """

    def set_vertices(self, vertices: Tuple[PointR2]):
        assert len(vertices) == 4, '4 vertices expected, received {0}'.format(len(vertices))
        super(Car, self).set_vertices(vertices)

    def execute_motion(self, K: float, w: float, dt: float):
        """  HERE has to be better defined in terms of control vector and state vector
             :type K: power of velocity command - no unit  --- is K = 1 or -1 only?
             :type w: angular velocity command - unit [rad/s]
             :type dt: time derivative [s] ? TODO: check the unit
        """
        self.pose.x += math.cos(self.pose.theta) * K * self.v_max * dt
        self.pose.y += math.sin(self.pose.theta) * K * self.v_max * dt
        self.pose.theta += w * dt
        return self.pose

    def plot(self, axes, world: WorldGrid, pose: PoseR2S1, color='b'):
        vertex_count = 4
        vertices = self.get_vertices_at_pose(pose)
        for j in range(vertex_count):
            a = vertices[j % vertex_count]
            b = vertices[(j + 1) % vertex_count]
            if world is None:
                axes.plot([a.x, b.x], [a.y, b.y], color)
            else:
                ia = PointR2(world.x_to_ix(a.x), world.y_to_iy(a.y))
                ib = PointR2(world.x_to_ix(b.x), world.y_to_iy(b.y))
                axes.plot([ia.x, ib.x], [ia.y, ib.y], color)

    @property
    def phi(self):
        return float('nan')

    @phi.setter
    def phi(self, angle: float):
        pass


class ArticulatedVehicle(Vehicle):
    """
    Articulated vehicle represented by towing tractor, link and trailer

    #TODO: in the car, you have divided the parameters of the Ackermann steering model Vs. the geometric vertexes
           for the definition of the motion, you are using a single track model,
           so the width is NOT used in the equations

    :type v_max: float maximum absolute linear velocity - unit [m/s]  TODO : check this
    :type alpha_max: float maximum absolute steering angle - unit [rad]
    :type a_max: float maximum absolute linear velocity  ---- MB: what's the difference between a_max and v_max ?
                                                              here the same, in the equation you use v_max not a_max
    :type w_max: float maximum absolute rotational velocity - unit [rad/2]
    :type tractor_w: width of the tractor
    :type tractor_l: length of the tractor
    :type link_l:
    :type trailer_w:
    :type trailer_l:
    """
    def __init__(self, v_max: float, alpha_max: float, a_max: float, w_max: float, tractor_l: float, tractor_w: float,
                 link_l: float, trailer_l: float, trailer_w: float):
        super(ArticulatedVehicle, self).__init__(v_max, alpha_max, a_max, w_max)
        self._phi = 0.0  # phi is the angle between the link and trailer
        # vertices numbered as shown below
        #
        #       8----------------7       2-------1
        #       |                |       |       |
        #   T_W |                5-------4   o   | H_W
        #       |                |  L_L  |       |
        #       9----------------6       3-------0
        #              T_L                  H_L
        c = PoseR2S1(0, 0, 0.)
        p0 = c.compose_point(PointR2( tractor_l / 2., -tractor_w / 2.))
        p1 = c.compose_point(PointR2( tractor_l / 2.,  tractor_w / 2.))
        p2 = c.compose_point(PointR2(-tractor_l / 2.,  tractor_w / 2.))
        p3 = c.compose_point(PointR2(-tractor_l / 2., -tractor_w / 2.))
        p4 = c.compose_point(PointR2(-tractor_l / 2., 0.))
        p5 = c.compose_point(PointR2(-tractor_l / 2. - link_l, 0.))
        pivot = PoseR2S1(p5.x, p5.y, self._phi)
        p6 = pivot.compose_point(PointR2(0, -trailer_w / 2.))
        p7 = pivot.compose_point(PointR2(0, trailer_w / 2.))
        p8 = pivot.compose_point(PointR2(-trailer_l, trailer_w / 2.))
        p9 = pivot.compose_point(PointR2(-trailer_l, -trailer_w / 2.))
        self.shape = (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9)
        self._tractor_w = tractor_w
        self._tractor_l = tractor_l
        self._link_l = link_l
        self._trailer_w = trailer_w
        self._trailer_l = trailer_l
        self._pivot = pivot

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, angle: float):
        self._phi = angle
        # adjust the trailer vertices based on the new phi
        self._pivot.theta = -self._phi
        p6 = self._pivot.compose_point(PointR2(0, -self._trailer_w / 2.))
        p7 = self._pivot.compose_point(PointR2(0, self._trailer_w / 2.))
        p8 = self._pivot.compose_point(PointR2(-self._trailer_l, self._trailer_w / 2.))
        p9 = self._pivot.compose_point(PointR2(-self._trailer_l, -self._trailer_w / 2.))
        self.shape = (*self.shape[:6], p6, p7, p8, p9)


    def execute_motion(self, K: int, w: float, dt: float) -> PoseR2S1:
        """
        Articulated vehicle motion execution calls two functions in case of forward/backward motion
             :type K: power of velocity command - no unit
             :type w: angular velocity command - unit [rad/s]
             :type dt: time derivative [s] ? TODO: check the unit
        """
        if K == 1:  # MB: should this be >0 or <0 ?
            (self.pose, self._phi) = self._sim_move_forward(self.pose, w, self._phi, dt)
        elif K == -1:
            (rev_pose, rev_phi) = self._sim_reverse(self.pose, self._phi)
            (new_pose, new_phi) = self._sim_move_forward(rev_pose, w, rev_phi, dt)
            (self.pose, self._phi) = self._sim_reverse(new_pose, new_phi)
        return self.pose

    def _sim_reverse(self, init_pose: PoseR2S1, phi: float):
        """
        Articulated vehicle motion execution calls two functions in case of forward/backward motion
             :type init_pose: unit [m] ?
             :type phi: unit [rad] ?

             TODO: better clarify the equations with references []

             does it consider the virtual backward motion? can we explain the rationale ?
        """
        x_rev = init_pose.x - (self._trailer_l / 2.) * math.cos(init_pose.theta) - \
                (self._tractor_l / 2 + self._link_l) * math.cos(init_pose.theta + phi)
        y_rev = init_pose.y - (self._trailer_l / 2) * math.sin(init_pose.theta) - \
                (self._tractor_l / 2 + self._link_l) * math.sin(init_pose.theta + phi)
        theta_rev = init_pose.theta + phi + np.pi
        phi_rev = -phi
        return PoseR2S1(x_rev, y_rev, theta_rev), phi_rev

    def _sim_move_forward(self, init_pose: PoseR2S1, w: float, phi: float, dt: float) -> (PoseR2S1, float):
        """
        Articulated vehicle motion execution calls two functions in case of forward/backward motion
             :type init_pose: unit [m] ?
             :type w: angular velocity command - unit [rad/s]
             :type phi: unit [rad] ?
             :type dt: time derivative [s] ? TODO: check the unit

             TODO: better clarify the equations with references [] - Lamirax maybe?
             F. Lamiraux, S. Sekhavat, and J.-P. Laumond. Motion planning and ontrol for hilare pulling a trailer.
             Robotis and Automation, IEEE Transations on, 15(4):640-652, Aug 1999
        """
        final_pose = PoseR2S1()
        final_pose.x = init_pose.x + self.v_max * dt * math.cos(init_pose.theta)
        final_pose.y = init_pose.y + self.v_max * dt * math.sin(init_pose.theta)
        final_pose.theta = init_pose.theta + w * dt
        final_pose_phi = phi - dt * ((self.v_max / (self._trailer_l / 2)) * math.sin(phi) - (
            (self._tractor_l / 2 + self._link_l) * w / (self._trailer_l / 2)) * math.cos(phi) - w)

        return final_pose, final_pose_phi

    def plot(self, axes, world: WorldGrid, pose: PoseR2S1, color='b'):
        """
             The plot function
        """
        vertices = self.get_vertices_at_pose(pose)
        for j in range(len(vertices) - 1):
            a = vertices[j]
            b = vertices[j + 1]
            if world is None:
                axes.plot([a.x, b.x], [a.y, b.y], color)
            else:
                ia = PointR2(world.x_to_ix(a.x), world.y_to_iy(a.y))
                ib = PointR2(world.x_to_ix(b.x), world.y_to_iy(b.y))
                axes.plot([ia.x, ib.x], [ia.y, ib.y], color)

        a = vertices[3]
        b = vertices[0]
        if world is None:
            axes.plot([a.x, b.x], [a.y, b.y], color)
        else:
            ia = PointR2(world.x_to_ix(a.x), world.y_to_iy(a.y))
            ib = PointR2(world.x_to_ix(b.x), world.y_to_iy(b.y))
            axes.plot([ia.x, ib.x], [ia.y, ib.y], color)

        a = vertices[9]
        b = vertices[6]
        if world is None:
            axes.plot([a.x, b.x], [a.y, b.y], color)
        else:
            ia = PointR2(world.x_to_ix(a.x), world.y_to_iy(a.y))
            ib = PointR2(world.x_to_ix(b.x), world.y_to_iy(b.y))
            axes.plot([ia.x, ib.x], [ia.y, ib.y], color)
