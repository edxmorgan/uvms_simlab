# se3_ompl_planner.py
import numpy as np
from functools import partial
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import BPoly
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

# Vehicle bubble, only used for the built in obstacles fallback
VEHICLE_RADIUS = 0.4


# ---------- helpers ----------
def _quat_rotate_wxyz(q, v):
    # q in wxyz, v is 3
    w, x, y, z = q
    vx, vy, vz = v
    # quaternion multiply q * [0,v] * q_conj
    # compute cross terms directly
    # r = v + 2*cross(q_vec, cross(q_vec, v) + w*v)
    qv = np.array([x, y, z], dtype=float)
    v = np.array([vx, vy, vz], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)

def _valid_with_obstacles(obstacles, state):
    x, y, z = state.getX(), state.getY(), state.getZ()
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN <= z <= Z_MAX):
        return False
    p = np.array([x, y, z], float)
    for obs in obstacles:
        if obs.collides(p, margin=VEHICLE_RADIUS):
            return False
    return True

def _valid_with_fcl(fcl_world, base_to_world, safety_margin, state):
    x, y, z = state.getX(), state.getY(), state.getZ()
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN <= z <= Z_MAX):
        return False

    # transform planner state from base frame to world frame for FCL
    if base_to_world is not None:
        q_wxyz = base_to_world["quat_wxyz"]
        t_xyz  = base_to_world["trans_xyz"]
        pw = _quat_rotate_wxyz(q_wxyz, [x, y, z]) + np.asarray(t_xyz, float)
    else:
        pw = np.array([x, y, z], float)

    # if min distance API exists, use a margin, else fall back to collision test
    if hasattr(fcl_world, "min_distance_xyz") and safety_margin is not None and safety_margin > 0.0:
        d = float(fcl_world.min_distance_xyz(pw))
        return d >= safety_margin
    else:
        return not bool(fcl_world.planner_in_collision_at_xyz(pw))


def plan_se3_path(
    start_xyz,
    start_quat_wxyz,
    goal_xyz,
    goal_quat_wxyz,
    time_limit=0.75,
    obstacles=None,
    fcl_world=None,
    base_to_world=None,
    safety_margin=0.0,
):
    """
    Return dict with keys xyz, quat_wxyz, count.
    If fcl_world is provided, validity uses its planner sphere against the env.
    base_to_world is a dict with keys quat_wxyz and trans_xyz for transforming states.
    safety_margin, in meters, requires clearance when min distance is available.
    """
    DEFAULT_OBSTACLES = []
    obstacles = DEFAULT_OBSTACLES if obstacles is None else obstacles

    space = ob.SE3StateSpace()
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, X_MIN); bounds.setHigh(0, X_MAX)
    bounds.setLow(1, Y_MIN); bounds.setHigh(1, Y_MAX)
    bounds.setLow(2, Z_MIN); bounds.setHigh(2, Z_MAX)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    if fcl_world is not None:
        checker = ob.StateValidityCheckerFn(
            partial(_valid_with_fcl, fcl_world, base_to_world, float(safety_margin))
        )
    else:
        checker = ob.StateValidityCheckerFn(
            partial(_valid_with_obstacles, obstacles)
        )

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