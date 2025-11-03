#!/usr/bin/env python3
import numpy as np
import rclpy, fcl
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from mesh_utils import fcl_bvh_from_mesh, make_marker, collect_env_meshes, color


class CollisionNode(Node):
    def __init__(self):
        super().__init__('mesh_collision_node')
        self.declare_parameter('robot_description', '')

        urdf_string = self.get_parameter(
            'robot_description'
        ).get_parameter_value().string_value
        if not urdf_string:
            self.get_logger().error(
                'robot_description param is empty. Did you load it into the param server in launch'
            )
            raise RuntimeError('no robot_description')
    
        # gather meshes from URDF
        robot_links, env_links = collect_env_meshes(urdf_string)

        self.get_logger().info(f'robot_links { [x["link"] for x in robot_links] }')
        self.get_logger().info(f'env_links { [x["link"] for x in env_links] }')

        # build FCL bodies
        self.bodies_robot = self.build_fcl_bodies(robot_links, "robot")
        self.bodies_env   = self.build_fcl_bodies(env_links,   "env")

        # build broadphase managers and register the objects
        self.manager_robot = fcl.DynamicAABBTreeCollisionManager()
        self.manager_env   = fcl.DynamicAABBTreeCollisionManager()

        self.manager_robot.registerObjects(
            [b["fcl_obj"] for b in self.bodies_robot]
        )
        self.manager_env.registerObjects(
            [b["fcl_obj"] for b in self.bodies_env]
        )

        self.manager_robot.setup()
        self.manager_env.setup()

        # TF buffer and publisher
        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)

        self.contact_pub = self.create_publisher(Marker, 'contact_markers', 10)

        # run at 20 Hz
        self.timer = self.create_timer(0.05, self.tick)


    def tick(self):
        ok_list = list(map(self.set_from_tf, self.bodies_robot + self.bodies_env))
        if not np.all(ok_list):
            return

        self.manager_robot.update()
        self.manager_env.update()

        # clear old markers
        clear = Marker()
        clear.header.frame_id = 'world'
        clear.action = Marker.DELETEALL
        self.contact_pub.publish(clear)

        # 1. collision query many to many
        req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        cdata = fcl.CollisionData(request=req)
        self.manager_robot.collide(self.manager_env, cdata, fcl.defaultCollisionCallback)

        # # 2. global clearance using manager distance
        # ddata = fcl.DistanceData()
        # self.manager_robot.distance(self.manager_env, ddata, fcl.defaultDistanceCallback)
        # global_min_dist = ddata.result.min_distance
        # rob_px, rob_py, rob_pz = ddata.result.nearest_points[0] # point on robot surface
        # env_px, env_py, env_pz = ddata.result.nearest_points[1] # point on env surface
        # self.get_logger().info(f'global min dist {global_min_dist:.4f} m')
        
        # map contact pair name tuple to one representative contact point
        # so we only publish one marker per robot link vs env link
        pair_to_point = {}

        # build lookup so we can turn contact.o1 o2 into names
        geom_id_to_name = {}
        for b in self.bodies_robot + self.bodies_env:
            geom_id_to_name[id(b["geom"])] = b["name"]

        for contact in cdata.result.contacts:
            n0 = geom_id_to_name.get(id(contact.o1), "unknown")
            n1 = geom_id_to_name.get(id(contact.o2), "unknown")
            pair_key = tuple(sorted([n0, n1]))

            # save one point for this pair
            if pair_key not in pair_to_point:
                pair_to_point[pair_key] = np.array(contact.pos, dtype=float)

        # now publish one sphere per pair
        CONTACT_MARKER_SIZE = 0.05
        red = color(r=1.0, g=0.1, b=0.1, a=1.0)
        for idx, (pair_key, p_world) in enumerate(pair_to_point.items()):
            sphere = make_marker(
                'contact',
                idx,
                'world',
                CONTACT_MARKER_SIZE,
                p_world,
                red
            )

            sphere.lifetime.sec = 0
            sphere.lifetime.nanosec = int(0.1 * 1e9)
            self.contact_pub.publish(sphere)

        # for pair_key in pair_to_point.keys():
        #     self.get_logger().warn(f'Collision {pair_key[0]} <-> {pair_key[1]}')


    def build_fcl_bodies(self, link_list, kind):
        out = []
        for m in link_list:
            path_abs  = m['uri']
            scale_vec = m['scale']
            xyz       = tuple(m['xyz'])
            rpy       = tuple(m['rpy'])

            bvh, nV, nF = fcl_bvh_from_mesh(path_abs, scale_vec, rpy, xyz)

            obj = fcl.CollisionObject(bvh, fcl.Transform())

            out.append({
                "name":   m['link'],
                "frame":  m['link'],    # assume TF frame matches link
                "fcl_obj": obj,
                "geom":   bvh           # save geometry handle here
            })

            self.get_logger().info(
                f'{kind} body {m["link"]} verts {nV} faces {nF}'
            )
        return out



    def set_from_tf(self, body):
        try:
            t: TransformStamped = self.tf_buf.lookup_transform(
                'world',
                body["frame"],
                rclpy.time.Time()
            )
            q = t.transform.rotation
            p = t.transform.translation

            # update transform in FCL object
            body["fcl_obj"].setTransform(
                fcl.Transform(
                    [q.w, q.x, q.y, q.z],
                    [p.x, p.y, p.z]
                )
            )
            return True
        except Exception:
            return False


def main():
    rclpy.init()
    node = CollisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()