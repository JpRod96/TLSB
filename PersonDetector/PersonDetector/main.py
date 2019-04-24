from videoProcessor import VideoProcessor
import util
from VideoProcessManager import VideoProcessManager

FRAMES_NRO=5

#videoProcessor = VideoProcessor()
#videoProcessor.process("sample.mp4", FRAMES_NRO);

vid = VideoProcessManager(5)

vid.processPath("C:/Users/Jp.SANDRO-HP/Desktop/wipi.mp4")
vid.processPath("C:/Users/Jp.SANDRO-HP/Desktop")