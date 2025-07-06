import os
import sys

import imageio
import numpy as np
import cv2 # Added for text overlay
import utils


class VideoRecorder(object):
    """
    Records videos of environment episodes, optionally overlaying intrinsic reward.
    """
    def __init__(self, root_dir, height=256, width=256, fps=25, camera_id=0):
        """
        Args:
            root_dir (str or None): Directory to save videos. If None, saving is disabled.
            height (int): Height of the video frames.
            width (int): Width of the video frames.
            fps (int): Frames per second for the video.
            camera_id (int): ID of the camera to use for rendering if an environment is passed.
        """
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.fps = fps
        self.camera_id = camera_id
        self.frames = []
        self.skill_id = None # Stores the skill_id for the current video, set via init()

    def init(self, enabled=True, skill_id=None):
        """
        Initializes the recorder for a new video.
        Args:
            enabled (bool): Whether to enable recording for this video.
            skill_id (int, optional): The skill ID for the current episode, used for naming.
        """
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if skill_id is not None:
            self.skill_id = skill_id

        # Note: The actual filename (e.g., incorporating timestamp and skill_id)
        # is determined by the caller (Workspace.evaluate) when calling save().

    def record(self, env_or_frame, intrinsic_reward=None):
        """
        Records a single frame.
        Args:
            env_or_frame (object or np.ndarray): Either the environment to render from,
                                                 or a pre-rendered frame (HWC, RGB).
            intrinsic_reward (float, optional): Intrinsic reward value to overlay on the frame.
        """
        if not self.enabled:
            return

        if isinstance(env_or_frame, np.ndarray):
            frame = env_or_frame
        else: # Assuming it's an environment object
            frame = env_or_frame.render(mode='rgb_array',
                                        height=self.height,
                                        width=self.width,
                                        camera_id=self.camera_id)

        if frame is None: # Should not happen if render is called correctly
            return

        # Ensure frame is writeable for OpenCV operations if it's a numpy array
        if isinstance(frame, np.ndarray) and not frame.flags.writeable:
            frame = frame.copy()

        # if intrinsic_reward is not None:
        #     # Convert frame to BGR format for OpenCV text rendering
        #     frame_bgr = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_RGB2BGR)
        #     text = f"IR: {intrinsic_reward:.4f}" # Format intrinsic reward
        #     # Define text properties
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 0.3
        #     font_color = (0, 255, 0)  # Green color in BGR
        #     thickness = 2
        #     text_position = (15, 15)  # Position (x, y) from top-left
        #     # Add text to frame
        #     cv2.putText(frame_bgr, text, text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)
        #     # Convert frame back to RGB for storage/display
        #     frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        self.frames.append(frame)

    def save(self, file_name):
        """
        Saves the recorded frames as a video file.
        Args:
            file_name (str): Name of the video file (e.g., "step_skill_id.mp4").
        """
        if self.enabled and self.frames:
            path = os.path.join(self.save_dir, file_name)
            try:
                imageio.mimsave(path, self.frames, fps=self.fps)
            except Exception as e:
                print(f"Error saving video: {e}")
                print(f"Details: Frames count: {len(self.frames)}, "
                      f"First frame shape: {self.frames[0].shape if self.frames else 'N/A'}, "
                      f"dtype: {self.frames[0].dtype if self.frames else 'N/A'}")
            # Clear frames after saving to prepare for the next recording session
            self.frames = []
