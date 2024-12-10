import rclpy
from rclpy.node import Node
import pyspacemouse
import numpy as np
from uvms_interfaces.msg import Command
from pynput import keyboard
import threading

class SpaceMouse(Node):
    def __init__(self):
        super().__init__('space_mouse_node',
                          automatically_declare_parameters_from_overrides=True)


        # Get parameter values
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value

        self.total_no_efforts = self.no_robot * self.no_efforts

        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', 10)
        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

        # Initialize keyboard-controlled variables
        self.kb_lock = threading.Lock()
        self.kb_x = 0.0    # Forward/Backward
        self.kb_y = 0.0    # Left/Right
        self.kb_z = 0.0    # Down
        self.kb_roll = 0.0
        self.kb_pitch = 0.0
        self.kb_yaw = 0.0

        # Start keyboard listener in a separate thread
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.get_logger().info("SpaceMouse node has been initialized with keyboard control.")

    def on_press(self, key):
        with self.kb_lock:
            if key == keyboard.Key.up:
                self.kb_x = 0.8     # Forward
                self.get_logger().debug("Up Arrow Pressed: Forward")
            elif key == keyboard.Key.down:
                self.kb_x = -0.8   # Backward
                self.get_logger().debug("Down Arrow Pressed: Backward")
            elif key == keyboard.Key.left:
                self.kb_y = 0.8    # Left
                self.get_logger().debug("Left Arrow Pressed: Left")
            elif key == keyboard.Key.right:
                self.kb_y = -0.8   # Right
                self.get_logger().debug("Right Arrow Pressed: Right")
            elif key == keyboard.Key.space:
                self.kb_z = 2.5   # Down
                self.get_logger().debug("Spacebar Pressed: Down")
            # Ignore other keys

    def on_release(self, key):
        with self.kb_lock:
            if key == keyboard.Key.up or key == keyboard.Key.down:
                self.kb_x = 0.0
                self.get_logger().debug(f"{key} Released: Stop Forward/Backward")
            elif key == keyboard.Key.left or key == keyboard.Key.right:
                self.kb_y = 0.0
                self.get_logger().debug(f"{key} Released: Stop Left/Right")
            elif key == keyboard.Key.space:
                self.kb_z = 0.0
                self.get_logger().debug("Spacebar Released: Stop Down")
            # Ignore other keys

    def timer_callback(self):
        # Create and publish the command message
        command_msg = Command()
        state = pyspacemouse.read()
        command_msg.command_type = "force"

        real_data = [0.0] * 5

        if state.buttons == [1, 0]:  # open --> right button
            real_data[4] = -2

        elif state.buttons == [0, 1]:  # close --> right button
            real_data[4] = 2

        if state.yaw > 0.0:
            real_data[3] = 2 * state.yaw

        elif state.yaw < -0.5:
            real_data[3] = 2 * state.yaw

        if abs(state.y) > 0.5:
            real_data[2] = 2 * -np.sign(state.y)

        if abs(state.z) > 0.5:
            real_data[1] = 2 * -np.sign(state.z)

        if abs(state.x) > 0.5:
            real_data[0] = 2 * -np.sign(state.x)

        # Combine the data for all robots
        data = []
        
        # Get keyboard-controlled x, y, z
        with self.kb_lock:
            kb_x = self.kb_x
            kb_y = self.kb_y
            kb_z = self.kb_z

        # Add keyboard-controlled x, y, z, roll, pitch, yaw
        data.extend([kb_x, kb_y, kb_z, 0.0, 0.0, 0.0])  # 6 elements

        data.extend(real_data)  # Add real manipulator joint positions (5 elements)

        other_data = [0.0] * (self.total_no_efforts - len(data))
        data.extend(other_data)

        assert len(data) == self.total_no_efforts, f"Data length mismatch. Expected {self.total_no_efforts}, got {len(data)}"
        # Convert all data to float
        dataF = [float(value) for value in data]
        command_msg.input.data = dataF

        # Publish the command
        self.publisher_.publish(command_msg)

    def destroy_node(self):
        self.listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    space_mouse = SpaceMouse()

    if pyspacemouse.open():
        try:
            rclpy.spin(space_mouse)
        except KeyboardInterrupt:
            space_mouse.get_logger().info('SpaceMouse node stopped by KeyboardInterrupt.')
        finally:
            space_mouse.destroy_node()
            rclpy.shutdown()
    else:
        space_mouse.get_logger().error("Failed to open SpaceMouse.")
        space_mouse.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
