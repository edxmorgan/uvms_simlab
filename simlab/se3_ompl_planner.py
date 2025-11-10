# se3_ompl_planner.py
import numpy as np
from functools import partial
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from fcl_checker import FCLWorld
try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception as e:
    raise RuntimeError(
        "OMPL Python bindings not found. Install ompl or ompl-python."
    ) from e


# Bounds
X_MIN, X_MAX = -10000.0, 10000.0
Y_MIN, Y_MAX = -10000.0, 10000.0
Z_MIN, Z_MAX = -10000.0, 0.0


def _valid_with_fcl(rclpy_node:Node, fcl_world:FCLWorld, safety_margin:float, state):
    x, y, z = state.getX(), state.getY(), state.getZ()
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN <= z <= Z_MAX):
        return False
    pw = np.array([x, y, z], float)

    # use a margin, else fall back to collision test
    if safety_margin is not None and safety_margin > 0.0:
        # rclpy_node.get_logger().info(f"********************using min_distance_xyz")
        d = fcl_world.min_distance_xyz(pw)
        # rclpy_node.get_logger().info(f"********************planner points validity {pw[0]}, {pw[1]}, {pw[2]} is in collision distance : {d}")
        return d >= safety_margin
    else:
        # rclpy_node.get_logger().info(f"********************using collision test")
        in_collision = fcl_world.planner_in_collision_at_xyz(pw)
        # rclpy_node.get_logger().info(f"********************planner points validity {pw[0]}, {pw[1]}, {pw[2]} is in collision : {in_collision}")
        return not in_collision


def plan_se3_path(
    rclpy_node,
    start_xyz,
    start_quat_wxyz,
    goal_xyz,
    goal_quat_wxyz,
    time_limit=0.75,
    fcl_world=None,
    safety_margin=0.0,
):
    """
    Return dict with keys xyz, quat_wxyz, count.
    If fcl_world is provided, validity uses its planner sphere against the env.
    base_to_world is a dict with keys quat_wxyz and trans_xyz for transforming states.
    safety_margin, in meters, requires clearance when min distance is available.
    """
    space = ob.SE3StateSpace()
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, X_MIN); bounds.setHigh(0, X_MAX)
    bounds.setLow(1, Y_MIN); bounds.setHigh(1, Y_MAX)
    bounds.setLow(2, Z_MIN); bounds.setHigh(2, Z_MAX)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    checker = ob.StateValidityCheckerFn(partial(_valid_with_fcl, rclpy_node, fcl_world, float(safety_margin)))
    ss.setStateValidityChecker(checker)

    start = ob.State(space)
    goal = ob.State(space)

    # fill poses
    s = start()
    s.setX(float(start_xyz[0])); s.setY(float(start_xyz[1])); s.setZ(float(start_xyz[2]))
    sr = s.rotation()
    sr.w = float(start_quat_wxyz[0]); sr.x = float(start_quat_wxyz[1])
    sr.y = float(start_quat_wxyz[2]); sr.z = float(start_quat_wxyz[3])

    g = goal()
    g.setX(float(goal_xyz[0])); g.setY(float(goal_xyz[1])); g.setZ(float(goal_xyz[2]))
    gr = g.rotation()
    gr.w = float(goal_quat_wxyz[0]); gr.x = float(goal_quat_wxyz[1])
    gr.y = float(goal_quat_wxyz[2]); gr.z = float(goal_quat_wxyz[3])

    # Enforce bounds to normalize SO3 and clamp R3
    space.enforceBounds(s)
    space.enforceBounds(g)

    ss.setStartAndGoalStates(start, goal)

    if not ss.getSpaceInformation().satisfiesBounds(s):
        raise RuntimeError("Start violates bounds after enforceBounds")
    if not ss.getSpaceInformation().satisfiesBounds(g):
        raise RuntimeError("Goal violates bounds after enforceBounds")

    # planner and resolution
    ss.setPlanner(og.BITstar(ss.getSpaceInformation()))
    space.setLongestValidSegmentFraction(0.01)
    ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

    if not ss.solve(time_limit):
        # raise RuntimeError("Planner did not find a solution")
        return {
            "status":False,
            "message":"Planner did not find a solution"
        }

    path = ss.getSolutionPath()
    path.interpolate(50)

    # collect path
    xyz = []
    quat_wxyz = []
    for k in range(path.getStateCount()):
        st = path.getState(k)
        q = st.rotation()
        xyz.append([st.getX(), st.getY(), st.getZ()])
        quat_wxyz.append([q.w, q.x, q.y, q.z])

    return {
        "xyz": np.asarray(xyz, float),
        "quat_wxyz": np.asarray(quat_wxyz, float),
        "count": path.getStateCount(),
        "status":True,
        "message":"Planner found solution"
    }