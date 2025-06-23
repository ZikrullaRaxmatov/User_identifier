import av
from streamlit_webrtc import VideoProcessorBase

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None  # store latest frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

