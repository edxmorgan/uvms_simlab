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
            try:
                if key == keyboard.Key.up:
                    self.kb_x = 0.3     # Forward
                    self.get_logger().debug("Up Arrow Pressed: Forward")
                elif key == keyboard.Key.down:
                    self.kb_x = -0.3   # Backward
                    self.get_logger().debug("Down Arrow Pressed: Backward")
                elif key == keyboard.Key.left:
                    self.kb_y = -0.3    # Left
                    self.get_logger().debug("Left Arrow Pressed: Left")
                elif key == keyboard.Key.right:
                    self.kb_y = 0.3   # Right
                    self.get_logger().debug("Right Arrow Pressed: Right")
                elif key == keyboard.Key.space:
                    self.kb_z = 2.3     # Down
                    self.get_logger().debug("Spacebar Pressed: Down")
                # Rotational Controls: W, S, A, D, Q, E
                elif hasattr(key, 'char') and key.char is not None:
                    char = key.char.lower()
                    if char == 'w':
                        self.kb_pitch = 0.1  # Pitch Up
                        self.get_logger().debug("Key 'W' Pressed: Pitch Up")
                    elif char == 's':
                        self.kb_pitch = -0.1  # Pitch Down
                        self.get_logger().debug("Key 'S' Pressed: Pitch Down")
                    elif char == 'd':
                        self.kb_roll = 0.1  # Roll Left
                        self.get_logger().debug("Key 'D' Pressed: Roll Left")
                    elif char == 'a':
                        self.kb_roll = -0.1  # Roll Right
                        self.get_logger().debug("Key 'A' Pressed: Roll Right")
                    elif char == 'q':
                        self.kb_yaw = 0.1  # Yaw Left
                        self.get_logger().debug("Key 'Q' Pressed: Yaw Left")
                    elif char == 'e':
                        self.kb_yaw = -0.1  # Yaw Right
                        self.get_logger().debug("Key 'E' Pressed: Yaw Right")
            except AttributeError:
                # Handle special keys if necessary
                pass

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
            # Rotational Controls Release: W, S, A, D, Q, E
            elif hasattr(key, 'char') and key.char is not None:
                char = key.char.lower()
                if char in ['w', 's']:
                    self.kb_pitch = 0.0
                    self.get_logger().debug(f"Key '{char.upper()}' Released: Stop Pitch")
                elif char in ['a', 'd']:
                    self.kb_roll = 0.0
                    self.get_logger().debug(f"Key '{char.upper()}' Released: Stop Roll")
                elif char in ['q', 'e']:
                    self.kb_yaw = 0.0
                    self.get_logger().debug(f"Key '{char.upper()}' Released: Stop Yaw")
            # Ignore other keys

    def timer_callback(self):
        # Create and publish the command message
        command_msg = Command()
        state = pyspacemouse.read()
        command_msg.command_type = "force"

        # Process SpaceMouse input
        real_data = [0.0] * 5

        if state.buttons == [1, 0]:  # Open --> right button
            real_data[4] = -2
        elif state.buttons == [0, 1]:  # Close --> right button
            real_data[4] = 2

        if state.yaw > 0.0:
            real_data[3] = 0.35*state.yaw
        elif state.yaw < -0.5:
            real_data[3] = 0.6*state.yaw

        if abs(state.y) > 0.5:
            real_data[2] = 0.65*-np.sign(state.y)

        if abs(state.z) > 0.5:
            real_data[1] = -np.sign(state.z)

        if abs(state.x) > 0.5:
            real_data[0] = 1.4*-np.sign(state.x)

        # Acquire keyboard-controlled variables
        with self.kb_lock:
            kb_x = self.kb_x
            kb_y = self.kb_y
            kb_z = self.kb_z
            kb_roll = self.kb_roll
            kb_pitch = self.kb_pitch
            kb_yaw = self.kb_yaw

        # Initialize data list
        data = []

        # Apply the same commands to all robots
        for robot_index in range(self.no_robot):
            # Add keyboard-controlled x, y, z, roll, pitch, yaw
            data.extend([kb_x, kb_y, kb_z, kb_roll, kb_pitch, kb_yaw])  # 6 elements

            # Add real_data for manipulator joints (assuming real_data is applicable per robot)
            data.extend(real_data)  # 5 elements

        # Calculate remaining efforts if any
        current_length = len(data)
        if current_length < self.total_no_efforts:
            other_data = [0.0] * (self.total_no_efforts - current_length)
            data.extend(other_data)
        elif current_length > self.total_no_efforts:
            self.get_logger().warning(
                f"Data length ({current_length}) exceeds total_no_efforts ({self.total_no_efforts}). Truncating data."
            )
            data = data[:self.total_no_efforts]

        # Ensure the data length matches
        assert len(data) == self.total_no_efforts, (
            f"Data length mismatch. Expected {self.total_no_efforts}, got {len(data)}"
        )

        # Convert all data to float
        dataF = [float(value) for value in data]
        command_msg.force.data = dataF

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
