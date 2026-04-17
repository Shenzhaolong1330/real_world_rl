import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name
        available_serials = self.get_device_serial_numbers()
        print(serial_number)
        print(available_serials)
        if serial_number not in available_serials:
            raise RuntimeError(
                f"RealSense camera '{name}' expects serial '{serial_number}', "
                f"but available devices are: {available_serials}. "
                "Please update REALSENSE_CAMERAS in task config."
            )
        self.serial_number = serial_number
        self.depth = depth
        self.dim = dim
        self.fps = fps
        self.exposure = exposure
        self.timeout_error_count = 0
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self._start_pipeline()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def _start_pipeline(self):
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, self.dim[0], self.dim[1], rs.format.bgr8, self.fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, self.dim[0], self.dim[1], rs.format.z16, self.fps)
        self.profile = self.pipe.start(self.cfg)
        self.s = self.profile.get_device().query_sensors()[0]
        self.s.set_option(rs.option.exposure, self.exposure)

    def _restart_pipeline(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self._start_pipeline()

    def read(self):
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=1000)
            self.timeout_error_count = 0
        except RuntimeError as error:
            if "Frame didn't arrive" in str(error):
                self.timeout_error_count += 1
                if self.timeout_error_count >= 3:
                    self._restart_pipeline()
                    self.timeout_error_count = 0
                return False, None
            raise
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
