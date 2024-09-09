#!/usr/bin/python3

#
#   Developer : Alexey Zakharov (alexey.zakharov@vectioneer.com)
#   All rights reserved. Copyright (c) 2017 VECTIONEER.
#
from enum import Enum


class States(Enum):
    """List of states of the robot state machine"""
    INIT_M = 0
    OFF_S = 1
    DISENGAGED_S = 2
    RETRACT_S = 3
    ENGAGED_S = 4
    REFERENCING_S = 5
    FORCEDISENGAGED_S = 6
    ESTOP_OFF_S = 7

    OFF_TO_DISENGAGED_T = 102
    OFF_TO_REFERENCING_T = 105
    DISENGAGED_TO_OFF_T = 201
    DISENGAGED_TO_RETRACT_T = 203
    RETRACT_TO_DISENGAGED_T = 302
    DISENGAGED_TO_ENGAGED_T = 204
    ENGAGED_TO_DISENGAGED_T = 402
    TO_FORCEDDISENGAGE_T = 600
    RESET_FORCEDDISENGAGE_T = 602
    TO_ESTOP_T = 700
    RESET_ESTOP_T = 701


class StateEvents(Enum):
    """List of events of the robot state machine"""
    DO_NOTHING_E = -1
    GOTO_OFF_E = 0
    GOTO_DISENGAGED_E = 1
    GOTO_ENGAGED_E = 2
    GOTO_RETRACT_E = 3
    GOTO_REFERENCING_E = 4
    FORCE_DISENGAGE_E = 10
    EMERGENCY_STOP_E = 20
    SAVE_CONFIGURATION_E = 254
    ACKNOWLEDGE_ERROR = 255


class ModeCommands(Enum):
    """List of events of the robot mode machine"""
    GOTO_INIT_E = 0
    GOTO_PAUSE_E = 1
    GOTO_AUTO_RUN_E = 2
    GOTO_MANUAL_JOINT_MODE_E = 3
    GOTO_MANUAL_CART_MODE_E = 4
    GOTO_MOVE_TO_START_E = 5
    GOTO_TORQUE_MODE_E = 8
    GOTO_SEMI_AUTO_E = 9


class Modes(Enum):
    """List of modes of the robot mode machine"""
    INIT_M = 0
    PAUSE_M = 1
    AUTO_RUN_M = 2
    MANUAL_JOINT_MODE_M = 3
    MANUAL_CART_MODE_M = 4
    SEMI_AUTO_M = 5
    TORQUE_M = 6
    MOVE_TO_START_M = 7
    # Transitions
    PAUSE_TO_AUTO_RUN_T = 102
    PAUSE_TO_SEMI_AUTO_T = 105
    PAUSE_TO_MANUAL_JOINT_T = 103
    PAUSE_TO_MANUAL_CART_T = 104
    PAUSE_TO_MOVE_TO_START_T = 107
    MOVE_TO_START_TO_PAUSE_T = 701
    MANUAL_CART_TO_PAUSE_T = 401
    MANUAL_JOINT_TO_PAUSE_T = 301
    MANUAL_CART_TO_MANUAL_JOINT_T = 403
    MANUAL_JOINT_TO_MANUAL_CART_T = 304
    PAUSE_TO_TORQUE_T = 106
    TORQUE_TO_PAUSE_T = 601


class InterpreterStates(Enum):
    """List of states of the interpreter state machine"""
    PROGRAM_STOP_S = 0
    PROGRAM_RUN_S = 1
    PROGRAM_PAUSE_S = 2
    MOTION_NOT_ALLOWED_S = 3
    IN_TRANSITION = 100
    PROGRAM_IS_DONE = 200


class InterpreterEvents(Enum):
    """List of events of the interpreter state machine"""
    PLAY_PROGRAM_E = 2
    MOVE_TO_START = 3
    PAUSE_PROGRAM_E = 4
    STOP_PROGRAM_E = 5
    RESET_INTERPRETER_E = 6


class MotionGeneratorStates(Enum):
    """List of states of the motion generator state machine"""
    WAITING_MOVE_CMD_S = 0
    MOTION_NOT_ALLOWED_S = 2
    MOTION_ALLOWED_S = 3
    READY_TO_START_S = 4
    RUNNING_S = 5
    PAUSING_S = 6
    STOPPING_S = 7
    COMPUTATION_STARTED_S = 8
    TRAJECTORY_READY_S = 10
