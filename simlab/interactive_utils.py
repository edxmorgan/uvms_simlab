from geometry_msgs.msg import Point
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, InteractiveMarker,InteractiveMarkerControl
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from builtin_interfaces.msg import Duration

def compute_bounding_sphere_radius(points, quantile=0.995, pad=0.03):
    """
    Robust radius from the origin for a cloud, using a high quantile,
    then add a small pad.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
        return 0.4  # fallback
    r = np.linalg.norm(pts, axis=1)
    return float(np.quantile(r, quantile) + pad)


def makeBox(fixed, scale, marker_type, initial_pose):
    marker = Marker()
    marker.type = marker_type
    marker.pose = initial_pose
    marker.scale.x = scale * 0.25
    marker.scale.y = scale * 0.25
    marker.scale.z = scale * 0.25

    if fixed:
        marker.color.r = 1.0 
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
    else:
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
    return marker

def makeBoxControl(msg, fixed, interaction_mode, marker_type,
                    scale=1.0, show_6dof=False, initial_pose=Pose(), ignore_dof=[]):
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(makeBox(fixed, scale, marker_type, initial_pose))
    control.interaction_mode = interaction_mode
    msg.controls.append(control)

    if show_6dof:
        if 'roll' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 1.0
            control.orientation.y = 0.0
            control.orientation.z = 0.0
            control.name = "roll"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'surge' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 1.0
            control.orientation.y = 0.0
            control.orientation.z = 0.0
            control.name = "surge"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'yaw' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "yaw"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'heave' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 1.0
            control.orientation.z = 0.0
            control.name = "heave"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'pitch' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 0.0
            control.orientation.z = 1.0
            control.name = "pitch"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

        if 'sway' not in ignore_dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0
            control.orientation.x = 0.0
            control.orientation.y = 0.0
            control.orientation.z = 1.0
            control.name = "sway"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            msg.controls.append(control)

    return control

def make_UVMS_Dof_Marker(name, description, frame_id, control_frame, fixed,
                        interaction_mode, initial_pose, scale,
                        arm_base_pose, show_6dof=False, ignore_dof=[]):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = frame_id
    int_marker.pose = initial_pose
    int_marker.scale = scale
    int_marker.name = name
    int_marker.description = description
    marker_type = Marker.CUBE
    if control_frame == 'task':
        marker_type = Marker.SPHERE
    makeBoxControl(int_marker, fixed, interaction_mode, marker_type,
                        int_marker.scale, show_6dof, Pose(), ignore_dof)
    if control_frame == 'uv':
        makeBoxControl(int_marker, True, InteractiveMarkerControl.NONE, Marker.CUBE, 0.2, False, arm_base_pose, ignore_dof)
    return int_marker

def make_menu_control():
    menu_control = InteractiveMarkerControl()
    menu_control.interaction_mode = InteractiveMarkerControl.MENU
    menu_control.name = "robots_control_menu"
    menu_control.description = "target"
    menu_control.always_visible = True
    return menu_control

def is_point_valid(workspace_hull, vehicle_body_hull, point):
    """
    Returns True if 'point' is in the workspace hull but *not* in the vehicle hull.
    Equivalently, we want:  point ∈ (Workspace \\ Vehicle) = Workspace ∩ (Vehicle)^c
    """
    inside_workspace = np.all(
        np.dot(workspace_hull.equations[:, :-1], point) + workspace_hull.equations[:, -1] <= 0
    )
    inside_vehicle = np.all(
        np.dot(vehicle_body_hull.equations[:, :-1], point) + vehicle_body_hull.equations[:, -1] <= 0
    )
    # accept the point if it is inside the workspace and *not* inside the vehicle hull.
    return inside_workspace and not inside_vehicle


def generate_rov_ellipsoid(a=0.5, b=0.3, c=0.2, num_points=10000):
    points = []
    while len(points) < num_points:
        pt = np.random.uniform(-1, 1, 3)
        if (pt[0]/a)**2 + (pt[1]/b)**2 + (pt[2]/c)**2 <= 1:
            points.append(pt)
    return points

def get_broadcast_tf(stamp, pose, parent_frame, child_frame):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    return t


def dm_to_pose(dm):
    pose = Pose()
    pose.position.x = float(dm[0])
    pose.position.y = float(dm[1])
    pose.position.z = float(dm[2])
    roll = float(dm[3])
    pitch = float(dm[4])
    yaw = float(dm[5])
    q = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose


def pose_to_homogeneous(pose):
    """Convert a geometry_msgs/Pose into a 4x4 homogeneous transformation matrix."""
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    trans = [pose.position.x, pose.position.y, pose.position.z]
    mat = quaternion_matrix(quat)
    mat[0:3, 3] = trans
    return mat

def homogeneous_to_pose(mat):
    """Convert a 4x4 homogeneous transformation matrix into a geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x = mat[0, 3]
    pose.position.y = mat[1, 3]
    pose.position.z = mat[2, 3]
    quat = quaternion_from_matrix(mat)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    return pose

def get_relative_pose(marker_pose, endeffector_pose):
    """
    Compute the relative pose of the endeffector with respect to the marker.
    marker_pose and endeffector_pose should be geometry_msgs/Pose.
    Returns a Pose representing the endeffector pose in the marker's frame.
    """
    T_marker = pose_to_homogeneous(marker_pose)
    T_ee = pose_to_homogeneous(endeffector_pose)
    T_rel = np.dot(np.linalg.inv(T_marker), T_ee)
    return homogeneous_to_pose(T_rel)

# def publish_waypoint_spheres(
#     stamp,
#     frame_id,
#     xyz_np,
#     marker_id,
#     step=3,
#     size_m=0.08,
#     goal_size_m=0.14,
#     goal_color=(0.95, 0.2, 0.2, 1.0),   # RGBA for goal sphere
#     wp_color=(0.1, 0.6, 0.95, 1.0),     # RGBA for waypoint spheres
# ):
#     """
#     Build two markers, a SPHERE_LIST for waypoints and a single SPHERE for the goal.
#     Goal is always the final element of the full path.
#     """
#     if xyz_np is None or len(xyz_np) == 0:
#         return None, None

#     # Subsample for lightweight visualization
#     step = max(1, int(step))
#     pts_vis = xyz_np[::step]

#     # Waypoints, skip the last path point so the goal is separate
#     wp_marker = Marker()
#     wp_marker.header.frame_id = frame_id
#     wp_marker.header.stamp = stamp
#     wp_marker.ns = "planner"
#     wp_marker.id = int(marker_id)
#     wp_marker.type = Marker.SPHERE_LIST
#     wp_marker.action = Marker.ADD
#     wp_marker.scale.x = float(size_m)
#     wp_marker.scale.y = float(size_m)
#     wp_marker.scale.z = float(size_m)
#     wp_marker.pose.orientation.w = 1.0
#     wp_marker.color.r, wp_marker.color.g, wp_marker.color.b, wp_marker.color.a = wp_color

#     wp_marker.points = []
#     if len(pts_vis) > 1:
#         for p in pts_vis[:-1]:
#             pt = Point()
#             pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
#             wp_marker.points.append(pt)

#     # Goal, use the true last point
#     gx, gy, gz = map(float, pts_vis[-1])
#     goal_marker = Marker()
#     goal_marker.header.frame_id = frame_id
#     goal_marker.header.stamp = stamp
#     goal_marker.ns = "planner"
#     goal_marker.id = int(marker_id) + 1
#     goal_marker.type = Marker.SPHERE
#     goal_marker.action = Marker.ADD
#     goal_marker.scale.x = float(goal_size_m)
#     goal_marker.scale.y = float(goal_size_m)
#     goal_marker.scale.z = float(goal_size_m)
#     goal_marker.pose.position.x = gx
#     goal_marker.pose.position.y = gy
#     goal_marker.pose.position.z = gz
#     goal_marker.pose.orientation.w = 1.0
#     goal_marker.color.r, goal_marker.color.g, goal_marker.color.b, goal_marker.color.a = goal_color

#     wp_marker.frame_locked = True
#     goal_marker.frame_locked = True
#     return wp_marker, goal_marker
