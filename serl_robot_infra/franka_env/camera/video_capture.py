import queue
import threading
import time
import numpy as np

class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.last_frame = None
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self.t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.last_frame = frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        # print(self.name, self.q.qsize())
        try:
            return self.q.get(timeout=5)
        except queue.Empty as error:
            if self.last_frame is not None:
                return self.last_frame
            raise RuntimeError(f"Timed out waiting for camera frame: {self.name}") from error

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.close()
