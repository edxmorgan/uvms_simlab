import rclpy
from rclpy.node import Node
from control_msgs.msg import DynamicJointState
import csv
from datetime import datetime

class Joint_Names:
    alpha_axis_a = 'robot_real__axis_a'
    alpha_axis_b = 'robot_real__axis_b'
    alpha_axis_c = 'robot_real__axis_c'
    alpha_axis_d = 'robot_real__axis_d'
    alpha_axis_e = 'robot_real__axis_e'
    floating_base = 'robot_real_IOs'


class Axis_Interface_names:
    position = 'position'
    velocity = 'velocity'
    sim_time = 'sim_time'
    effort = 'effort'
    floating_base_x = 'position.x'
    floating_base_y = 'position.y'
    floating_base_z = 'position.z'
    floating_base_iw = 'orientation.w'
    floating_base_ix = 'orientation.x'
    floating_base_iy = 'orientation.y'
    floating_base_iz = 'orientation.z'


class SystemSensor(Node):
    def __init__(self):
        super().__init__('sensor')

        self.subscription = self.create_subscription(
            DynamicJointState,
            'dynamic_joint_states',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        self.q = None
        self.dq = None
        self.ned_T = None
        self.timestamp = None
        self.efforts = None

        # Create a timestamped filename for the CSV
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"joint_data_{timestamp_str}.csv"
        
        # Open the CSV file and prepare to write data
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write a header row for clarity
        self.csv_writer.writerow([
            'timestamp_alpha_axis_e', 'timestamp_alpha_axis_d', 'timestamp_alpha_axis_c', 'timestamp_alpha_axis_b',
            'q_alpha_axis_e', 'q_alpha_axis_d', 'q_alpha_axis_c', 'q_alpha_axis_b',
            'dq_alpha_axis_e', 'dq_alpha_axis_d', 'dq_alpha_axis_c', 'dq_alpha_axis_b'
        ])

    def listener_callback(self, msg: DynamicJointState):
        # Retrieve current joint positions and velocities
        self.q = self.get_interface_value(
            msg,
            [Joint_Names.alpha_axis_e,
             Joint_Names.alpha_axis_d,
             Joint_Names.alpha_axis_c,
             Joint_Names.alpha_axis_b],
            [Axis_Interface_names.position] * 4
        )

        self.dq = self.get_interface_value(
            msg,
            [Joint_Names.alpha_axis_e,
             Joint_Names.alpha_axis_d,
             Joint_Names.alpha_axis_c,
             Joint_Names.alpha_axis_b],
            [Axis_Interface_names.velocity] * 4
        )

        self.efforts = self.get_interface_value(
            msg,
            [Joint_Names.alpha_axis_e,
             Joint_Names.alpha_axis_d,
             Joint_Names.alpha_axis_c,
             Joint_Names.alpha_axis_b],
            [Axis_Interface_names.effort] * 4
        )

        self.timestamp = self.get_interface_value(
            msg,
            [Joint_Names.alpha_axis_e,
             Joint_Names.alpha_axis_d,
             Joint_Names.alpha_axis_c,
             Joint_Names.alpha_axis_b],
            [Axis_Interface_names.sim_time] * 4
        )

        self.ned_T = self.get_interface_value(
            msg,
            [Joint_Names.floating_base] * 7,
            [
                Axis_Interface_names.floating_base_x,
                Axis_Interface_names.floating_base_y,
                Axis_Interface_names.floating_base_z,
                Axis_Interface_names.floating_base_iw,
                Axis_Interface_names.floating_base_ix,
                Axis_Interface_names.floating_base_iy,
                Axis_Interface_names.floating_base_iz
            ]
        )

        # Compile row data
        row_data = self.timestamp + self.q + self.dq

        # Write the data to the CSV file
        # self.write_data_to_file(row_data)

    def get_interface_value(self, window_item, joint_names, interface_names):
        names = window_item.joint_names
        return [
            window_item.interface_values[names.index(joint_name)].values[
                window_item.interface_values[names.index(joint_name)].interface_names.index(interface_name)
            ]
            for joint_name, interface_name in zip(joint_names, interface_names)
        ]

    def write_data_to_file(self, row_data):
        """Write a single row of data to the CSV file."""
        self.csv_writer.writerow(row_data)
        self.csv_file.flush()

    def destroy_node(self):
        # Close the CSV file when the node is destroyed
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    sensor_node = SystemSensor()
    try:
        rclpy.spin(sensor_node)
    except KeyboardInterrupt:
        sensor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
