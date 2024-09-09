#!/usr/bin/python3

#
#   Developer : Alexey Zakharov (alexey.zakharov@vectioneer.com)
#   All rights reserved. Copyright (c) 2017 VECTIONEER.
#

import os
from . import motionSL_pb2


class Waypoint(object):
    """Class represents a waypoint of the motion path

        Args:
            pose(list(double)): pose in Cartesian or joint space
            smoothing_factor(double): waypoint smoothing factor in the range [0..1]
            next_segment_velocity_factor(double) segment velocity factor in the range [0..1]

    """

    def __init__(self, pose, smoothing_factor=0.1, next_segment_velocity_factor=1.0):
        self.pose = pose
        self.smoothing_factor = smoothing_factor
        self.next_segment_velocity_factor = next_segment_velocity_factor


class PoseTransformer(object):
    """Convert Cartesian tooltip to joint angles and the other way round

        Args:
            req(motorcortex.Request): reference to a Request instance
            motorcortex_types(motorcortex.MessageTypes): reference to a MessageTypes instance
    """

    def __init__(self, req, motorcortex_types):
        try:
            self.__MotionSpec = motorcortex_types.getNamespace("motion_spec")
        except AttributeError:
            path = os.path.dirname(motionSL_pb2.__file__)
            motorcortex_types.load([{'proto': motionSL_pb2, 'hash': path + '/motionSL_hash.json'}])

        self.__motorcortex_types = motorcortex_types
        self.__req = req

    def calcCartToJointPose(self, cart_coord=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            ref_joint_coord_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        """Converts Cartesian tooltip pose to joint coordinates

            Args:
                cart_coord(list(double)): Cartesian coordinates of the tooltip
                ref_joint_coord_rad(list(double)): actual joint coordinates, rad

            Returns:
                motion_spec.CartToJoint: Joint angles, which correspond to Cartesian coordinates,
                with respect to actual joint positions.

        """

        cart_to_joint_req = self.__MotionSpec.CartToJoint()
        cart_to_joint_req.cartpose.coordinates.extend(cart_coord)
        cart_to_joint_req.jointpose.coordinates.extend(ref_joint_coord_rad)
        cart_to_joint_req.carttwist.coordinates.extend(ref_joint_coord_rad)
        cart_to_joint_req.jointtwist.coordinates.extend(ref_joint_coord_rad)
        cart_to_joint_req.frame_type = self.__MotionSpec.TOOLTIP

        return self.__req.send(self.__motorcortex_types.encode(cart_to_joint_req)).get()

    def calcJointToCartPose(self, joint_coord_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            cart_coord=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        """Converts joint coordinates to Cartesian tooltip pose.

            Args:
                joint_coord_rad(list(double)): joint coordinates, rad
                cart_coord(list(double)): actual Cartesian tooltip pose

            Returns:
                motion_spec.JointToCart: Cartesian tooltip pose, which correspond to joint angles,
                with respect to the actual pose.

        """

        joint_to_cart_req = self.__MotionSpec.JointToCart()
        joint_to_cart_req.cartpose.coordinates.extend(cart_coord)
        joint_to_cart_req.jointpose.coordinates.extend(joint_coord_rad)
        joint_to_cart_req.carttwist.coordinates.extend(joint_coord_rad)
        joint_to_cart_req.jointtwist.coordinates.extend(joint_coord_rad)

        return self.__req.send(self.__motorcortex_types.encode(joint_to_cart_req)).get()


class MotionProgram(object):
    """Class represents a motion program of the manipulator

        Args:
            req(motorcortex.Request): reference to a Request instance
            motorcortex_types(motorcortex.MessageTypes): reference to a MessageTypes instance
    """

    def __init__(self, req, motorcortex_types):
        self.__Motorcortex = motorcortex_types.getNamespace("motorcortex")
        self.__MotionSpec = motorcortex_types.getNamespace("motion_spec")
        if not self.__MotionSpec:
            path = os.path.dirname(motionSL_pb2.__file__)
            motorcortex_types.load([{'proto': motionSL_pb2, 'hash': path + '/motionSL_hash.json'}])
            self.__MotionSpec = motorcortex_types.getNamespace("motion_spec")

        self.__motorcortex_types = motorcortex_types
        self.__req = req

        self.__motion_program = self.__MotionSpec.MotionProgram()

        self.__cmd_counter = 1
        self.__id = 1

    def clear(self):
        """Clears all commands in the program"""
        self.__motion_program = self.__MotionSpec.MotionProgram()
        self.__cmd_counter = 1

    def addCommand(self, command, type):
        """Adds a command to the program

            Args:
                command(motion_spec.MotionCommand): motion command from motionSL.proto
                type(motion_spec.MOTIONTYPE): type of the motion command
        """
        motion_cmd = self.__motion_program.commandlist.add()
        motion_cmd.id = self.__cmd_counter
        motion_cmd.commandtype = type
        motion_cmd.commandarguments = command.SerializeToString()
        self.__cmd_counter = self.__cmd_counter + 1

    def addMoveC(self, waypoint_list, angle, velocity=0.1, acceleration=0.2,
                 rotational_velocity=3.18, rotational_acceleration=6.37,
                 ref_joint_coord_rad=[]):
        """Adds a MoveC(circular move) command to the program

            Args:
                waypoint_list(list(WayPoint)): a list of waypoints
                angle(double): rotation angle, rad
                velocity(double): maximum velocity, m/sec
                acceleration(double): maximum acceleration, m/sec^2
                rotational_velocity(double): maximum joint velocity, rad/sec
                rotational_acceleration(double): maximum joint acceleration, rad/sec^2
                ref_joint_coord_rad: reference joint coordinates for the first waypoint

        """

        move_c = self.__MotionSpec.MoveC()
        move_c.constraint.type = self.__MotionSpec.VELANDACC
        move_c.constraint.velacc_values.vMax = velocity
        move_c.constraint.velacc_values.aMax = acceleration
        move_c.angle = angle
        move_c.constraint.velacc_values.omegaMax = rotational_velocity
        move_c.constraint.velacc_values.alfaMax = rotational_acceleration
        move_c.referenceJoint.coordinates.extend(ref_joint_coord_rad)

        for waypoint in waypoint_list:
            ms_waypoint_ref = move_c.waypoints.add()
            ms_waypoint_ref.constraint.type = self.__MotionSpec.POSITION
            ms_waypoint_ref.segmentVelocity = waypoint.next_segment_velocity_factor
            ms_waypoint_ref.constraint.factor = waypoint.smoothing_factor
            ms_waypoint_ref.pose.coordinates.extend(waypoint.pose)

        self.addCommand(move_c, self.__MotionSpec.ARC)

    def addMoveL(self, waypoint_list, velocity=0.1, acceleration=0.2,
                 rotational_velocity=3.18, rotational_acceleration=6.37,
                 ref_joint_coord_rad=[]):
        """Adds a MoveL(Linear move) command to the program

            Args:
                waypoint_list(list(WayPoint)): a list of waypoints
                velocity(double): maximum velocity, m/sec
                acceleration(double): maximum acceleration, m/sec^2
                rotational_velocity(double): maximum joint velocity, rad/sec
                rotational_acceleration(double): maximum joint acceleration, rad/sec^2
                ref_joint_coord_rad: reference joint coordinates for the first waypoint

        """

        move_l = self.__MotionSpec.MoveL()
        move_l.constraint.type = self.__MotionSpec.VELANDACC
        move_l.constraint.velacc_values.vMax = velocity
        move_l.constraint.velacc_values.aMax = acceleration
        move_l.constraint.velacc_values.omegaMax = rotational_velocity
        move_l.constraint.velacc_values.alfaMax = rotational_acceleration
        move_l.referenceJoint.coordinates.extend(ref_joint_coord_rad)

        for waypoint in waypoint_list:
            ms_waypoint_ref = move_l.waypoints.add()
            ms_waypoint_ref.constraint.type = self.__MotionSpec.POSITION
            ms_waypoint_ref.segmentVelocity = waypoint.next_segment_velocity_factor
            ms_waypoint_ref.constraint.factor = waypoint.smoothing_factor
            ms_waypoint_ref.pose.coordinates.extend(waypoint.pose)

        self.addCommand(move_l, self.__MotionSpec.CARTMOTION)

    def addMoveJ(self, waypoint_list, rotational_velocity=3.18, rotational_acceleration=6.37):
        """Adds MoveJ(Joint move) command to the program

            Args:
                waypoint_list(list(WayPoint)): a list of waypoints
                rotational_velocity(double): maximum joint velocity, rad/sec
                rotational_acceleration(double): maximum joint acceleration, rad/sec^2

        """

        move_j = self.__MotionSpec.MoveJ()
        move_j.constraint.type = self.__MotionSpec.VELANDACC
        move_j.constraint.velacc_values.omegaMax = rotational_velocity
        move_j.constraint.velacc_values.alfaMax = rotational_acceleration
        move_j.constraint.velacc_values.vMax = 0
        move_j.constraint.velacc_values.aMax = 0

        for waypoint in waypoint_list:
            ms_waypoint_ref = move_j.waypoints.add()
            ms_waypoint_ref.constraint.type = self.__MotionSpec.TIME
            ms_waypoint_ref.segmentVelocity = waypoint.next_segment_velocity_factor
            ms_waypoint_ref.constraint.factor = waypoint.smoothing_factor
            ms_waypoint_ref.pose.coordinates.extend(waypoint.pose)

        self.addCommand(move_j, self.__MotionSpec.JOINTMOTION)

    def addWait(self, timeout_s, path=None, value=1):
        """Adds Wait command to the program

            Args:
                timeout_s(double): time to wait in seconds
                path(string): path to the parameter that will be compared to value
                value: value that the parameter is compared to

        """

        wait_cmd = self.__MotionSpec.Wait()
        wait_cmd.timeout = timeout_s
        if path is not None:
            wait_cmd.path = path
            wait_cmd.value = value
        self.addCommand(wait_cmd, self.__MotionSpec.WAIT)

    def send(self, program_name='Undefined'):
        """Sends program to the robot

            Args:
                program_name(str): program name

        """

        self.__motion_program.name = program_name
        self.__motion_program.id = self.__id
        self.__id = self.__id + 1

        return self.__req.send(self.__motorcortex_types.encode(self.__motion_program))
