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

def _resample_by_distance(xyz_np, quat_np, spacing_m=0.20, max_points=2000):
    """
    Resample path at approximately fixed translation spacing.
    xyz_np:  (N, 3)
    quat_np: (N, 4) wxyz
    Returns xyz_rs, quat_rs with first and last included.
    """
    if xyz_np.shape[0] <= 2 or spacing_m <= 0:
        return xyz_np, quat_np

    # cumulative translation distance
    diffs = np.linalg.norm(np.diff(xyz_np, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])

    if total == 0.0:
        return xyz_np[:1], quat_np[:1]

    # target distances, always include 0 and total
    targets = np.arange(0.0, total, spacing_m)
    if targets.size == 0 or targets[-1] < total:
        targets = np.concatenate([targets, [total]])

    # map target distances to indices in the dense path
    idx = np.searchsorted(cum, targets, side="left")
    idx = np.clip(idx, 0, len(cum) - 1)

    # enforce uniqueness and max size
    idx = np.unique(idx)
    if idx.size > max_points:
        stride = int(np.ceil(idx.size / max_points))
        idx = idx[::stride]
        if idx[-1] != len(cum) - 1:
            idx = np.concatenate([idx, [len(cum) - 1]])

    return xyz_np[idx], quat_np[idx]


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
    spacing_m=0.20,           # new: desired waypoint spacing in meters
    dense_interpolation=400,  # new: dense samples before distance resampling
    max_points=2000           # new: cap to avoid flooding downstream
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
    # Densify first for a smoother arc length estimate
    path.interpolate(int(dense_interpolation))

    # Collect the dense path
    xyz_dense, quat_dense = [], []
    for k in range(path.getStateCount()):
        st = path.getState(k)
        q = st.rotation()
        xyz_dense.append([st.getX(), st.getY(), st.getZ()])
        quat_dense.append([q.w, q.x, q.y, q.z])

    xyz_dense = np.asarray(xyz_dense, float)
    quat_dense = np.asarray(quat_dense, float)

    # Resample at fixed spatial spacing to keep PID errors modest
    xyz_rs, quat_rs = _resample_by_distance(
        xyz_dense, quat_dense,
        spacing_m=float(spacing_m),
        max_points=int(max_points)
    )

    return {
        "xyz": xyz_rs,
        "quat_wxyz": quat_rs,
        "count": int(xyz_rs.shape[0]),
        "status": True,
        "message": f"Planner found solution with {xyz_rs.shape[0]} waypoints at ~{spacing_m} m spacing"
    }