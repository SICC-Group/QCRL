#!/home/ubuntu/anaconda3/envs/yfw/bin/python
import importlib
import os
import sys

import numpy as np

sys.path.append("/usr/local/webots/lib/controller/python")
os.environ["WEBOTS_HOME"] = "/usr/local/webots"

try:
    _CSVRobot = importlib.import_module(
        "deepbots.robots.controllers.csv_robot"
    ).CSVRobot
except ModuleNotFoundError:
    _CSVRobot = importlib.import_module(
        "deepbots.robots.controllers.robot_emitter_receiver_csv"
    ).RobotEmitterReceiverCSV


class CSVRobot(_CSVRobot):
    def initialize_comms(self, emitter_name="emitter", receiver_name="receiver"):
        if hasattr(self, "robot"):
            emitter = self.robot.getDevice(emitter_name)
            receiver = self.robot.getDevice(receiver_name)
            receiver.enable(self.timestep)
            return emitter, receiver
        return super().initialize_comms(emitter_name, receiver_name)

    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:

            # Webots versions differ: Receiver payload may be str or bytes.
            
            try:
                string_message = self.receiver.getString()
            except AttributeError:
                data = self.receiver.getData()
                if isinstance(data, (bytes, bytearray)):
                    string_message = data.decode("utf-8")
                else:
                    string_message = str(data)

            if isinstance(string_message, (bytes, bytearray)):
                string_message = string_message.decode("utf-8")

            self.use_message_data(string_message.split(","))
            self.receiver.nextPacket()

    def getDevice(self, name):
        if hasattr(self, "robot"):
            return self.robot.getDevice(name)
        return super().getDevice(name)

    def getName(self):
        if hasattr(self, "robot"):
            return self.robot.getName()
        return super().getName()

    def step(self, timestep):
        if hasattr(self, "robot"):
            return self.robot.step(timestep)
        return super().step(timestep)

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, "..", "..")
sys.path.append(exp_path)
from config import parser


#初始化传感器和电机
#接收上层下发动作并执行
#把本机传感器数据回传给上层（supervisor）

class Epuck2Robot(CSVRobot):
    def __init__(self, args):
        super().__init__(timestep=args.timestep//args.interval)
        '''self.left_wheel_sensor = self.getDevice("left wheel sensor")
        self.right_wheel_sensor = self.getDevice("right wheel sensor")
        self.left_wheel_sensor.enable(self.timestep)
        self.right_wheel_sensor.enable(self.timestep)'''
        self.args = args
        self.interval = args.interval
        self.max_speed = 6.28
        self.speeds = [self.max_speed, self.max_speed]
        self.robot_name = self.getName()[-1]
        self.timestep = args.timestep
        self.num_agents = args.num_agents
        self.ps_sensor_value = [] # 8 values

        self.init_device_variables()
        
        
    def init_device_variables(self):
        self.ps_sensor = []
        for i in range(8):
            ps = self.getDevice(f"ps{i}")
            ps.enable(self.timestep // self.interval)
            self.ps_sensor.append(ps)

        # self.gyro = self.getDevice("gyro")
        # self.gyro.enable(self.timestep//self.interval)
        # self.accelerometer = self.getDevice("accelerometer")
        # self.accelerometer.enable(self.timestep//self.interval)

        self.wheels = []
        for wheel_name in ['left wheel motor', 'right wheel motor']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

    def run(self):
        """
        This method is required by Webots to update the robot in the
        simulation. It steps the robot and in each step it runs the two
        handler methods to use the emitter and receiver components.

        This method should be called by a robot manager to run the robot.
        """
        i = 0
        while self.step(self.timestep//self.interval) != -1:
            self.handle_receiver()
            # if self.robot_name in [1, '1']:
                # print("t")
                # print(i)
            if (i + 1) % self.interval == 0:
                # if self.robot_name in [1, '1']:
                #     print(self.ps_sensor_value)
                self.handle_emitter()
                i = -1
                # if self.robot_name in [1, '1']:
                #     print("s")
            i += 1

    def create_message(self):
        # Read the sensor value, convert to string and save it in a list
        msg = ['a' + self.robot_name]
        # msg.extend(self.gyro.getValues())  # 3 values
        # msg.extend(self.accelerometer.getValues())  # 3 values
        msg.extend([self.wheels[0].getVelocity()])  # 1 value
        msg.extend([self.wheels[1].getVelocity()])  # 1 value
        self.ps_sensor_value = [ps.getValue() for ps in self.ps_sensor]  # 8 values
        msg.extend(self.ps_sensor_value)
        return msg

    def use_message_data(self, message):
        robot_idx = int(self.robot_name) - 1
        action_offset = robot_idx * 2
        alive_offset = 2 * self.num_agents + robot_idx

        try:
            left_ratio = float(message[action_offset])
            right_ratio = float(message[action_offset + 1])
        except (ValueError, IndexError):
            # Ignore malformed command packets and keep previous wheel command.
            return

        # if self.robot_name in [1, '1']:
        #     print("action:", message)
        if message[alive_offset] == '0':
            for i in range(2):
                self.wheels[i].setPosition(0.0)
                self.wheels[i].setVelocity(0.0)
            return
        
        left_ratio = float(np.clip(left_ratio, -1.0, 1.0))
        right_ratio = float(np.clip(right_ratio, -1.0, 1.0))
        self.speeds[0] = self.max_speed * left_ratio
        self.speeds[1] = self.max_speed * right_ratio
        # if message[-1] == "eval":
        #     dis = 160
        #     right_obstacle = any(v > dis for v in self.ps_sensor_value[:3])
        #     left_obstacle = any(v > dis for v in self.ps_sensor_value[5:8])
        #     front_obstacle = right_obstacle & left_obstacle
        #     back_obstacle = any(v > dis for v in self.ps_sensor_value[3:5])

        #     if front_obstacle:
        #         self.speeds[0] = -0.5 * self.max_speed
        #         self.speeds[1] = -0.5 * self.max_speed
        #     elif left_obstacle:
        #         self.speeds[0] = 0.5 * self.max_speed
        #         self.speeds[1] = -0.5 * self.max_speed
        #     elif right_obstacle:
        #         self.speeds[0] = -0.5 * self.max_speed
        #         self.speeds[1] = 0.5 * self.max_speed
        #     elif back_obstacle:
        #         self.speeds[0] = 0.5 * self.max_speed
        #         self.speeds[1] = 0.5 * self.max_speed
        
        for i in range(2):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(self.speeds[i])

            
if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    # Create the robot controller object and run it
    robot_controller = Epuck2Robot(args)
    robot_controller.run()  # Run method is implemented by the framework, just need to call it
