import multiprocessing
import numpy as np
from franka_env.spacemouse import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()
        self.closed = False

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0]
        self.stop_event = multiprocessing.Event()

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while not self.stop_event.is_set():
            try:
                state = pyspacemouse.read_all()
                action = [0.0] * 6
                buttons = [0, 0]
                if len(state) == 2:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw,
                        -state[1].y, state[1].x, state[1].z,
                        -state[1].roll, -state[1].pitch, -state[1].yaw
                    ]
                    buttons = state[0].buttons + state[1].buttons
                elif len(state) == 1:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw
                    ]
                    buttons = state[0].buttons

                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
            except (BrokenPipeError, EOFError, ConnectionResetError):
                break
            except Exception:
                if self.stop_event.is_set():
                    break

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        try:
            action = self.latest_data["action"]
            buttons = self.latest_data["buttons"]
        except (BrokenPipeError, EOFError, KeyError):
            action = [0.0] * 6
            buttons = [0, 0]
        return np.array(action), buttons
    
    def close(self):
        if self.closed:
            return
        self.closed = True

        self.stop_event.set()

        if hasattr(self, "process") and self.process.is_alive():
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)

        if hasattr(self, "manager"):
            self.manager.shutdown()
