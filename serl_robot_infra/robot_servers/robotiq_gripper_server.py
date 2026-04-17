import subprocess
import rospy
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg

from robot_servers.gripper_server import GripperServer


class RobotiqGripperServer(GripperServer):
    def __init__(self, gripper_device="/dev/ttyUSB0", use_rtu=True):
        """
        Initialize Robotiq gripper server.
        
        Args:
            gripper_device: For RTU mode, use serial device path (e.g., /dev/ttyUSB0)
                           For TCP mode, use IP address (e.g., 192.168.1.11)
            use_rtu: True for RS-485/USB connection, False for TCP/Ethernet connection
        """
        super().__init__()
        self.use_rtu = use_rtu
        
        # Select the appropriate node based on connection type
        if use_rtu:
            node_name = "Robotiq2FGripperRtuNode.py"
        else:
            node_name = "Robotiq2FGripperTcpNode.py"
        
        self.gripper = subprocess.Popen(
            [
                "rosrun",
                "robotiq_2f_gripper_control",
                node_name,
                gripper_device,
            ],
            stdout=subprocess.PIPE,
        )
        self.gripper_state_sub = rospy.Subscriber(
            "Robotiq2FGripperRobotInput",
            inputMsg.Robotiq2FGripper_robot_input,
            self._update_gripper,
            queue_size=1,
        )
        self.gripperpub = rospy.Publisher(
            "Robotiq2FGripperRobotOutput",
            outputMsg.Robotiq2FGripper_robot_output,
            queue_size=1,
        )
        self.gripper_command = outputMsg.Robotiq2FGripper_robot_output()

    def activate_gripper(self):
        self.gripper_command = self._generate_gripper_command("a", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def reset_gripper(self):
        self.gripper_command = self._generate_gripper_command("r", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)
        self.activate_gripper()

    def open(self):
        self.gripper_command = self._generate_gripper_command("o", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def close(self):
        self.gripper_command = self._generate_gripper_command("c", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def move(self, position):
        self.gripper_command = self._generate_gripper_command(position, self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def close_slow(self):
        self.gripper_command = self._generate_gripper_command("cs", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def _update_gripper(self, msg):
        """internal callback to get the latest gripper position."""
        self.gripper_pos = 1 - msg.gPO / 255

    def _generate_gripper_command(self, char, command):
        """Update the gripper command according to the character entered by the user."""
        if char == "a":
            command = outputMsg.Robotiq2FGripper_robot_output()
            command.rACT = 1
            command.rGTO = 1
            command.rSP = 255
            command.rFR = 30

        elif char == "r":
            command = outputMsg.Robotiq2FGripper_robot_output()
            command.rACT = 0
            command.rSP = 255

        elif char == "c":
            command.rPR = 255
            command.rSP = 255
        
        elif char == "cs":
            command.rPR = 255
            command.rSP = 50

        elif char == "o":
            command.rPR = 0
            command.rSP = 255

        # If the command entered is a int, assign this value to rPR
        # (i.e., move to this position)
        try:
            command.rPR = int(char)
            if command.rPR > 255:
                command.rPR = 255
            if command.rPR < 0:
                command.rPR = 0
        except ValueError:
            pass
        return command
