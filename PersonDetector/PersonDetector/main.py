from videoProcessor import VideoProcessor
import util
from VideoProcessManager import VideoProcessManager

framesNro=5
picSize=200

#videoProcessor = VideoProcessor()
#videoProcessor.process("sample.mp4", FRAMES_NRO);

vid = VideoProcessManager(framesNro, picSize)

vid.processPath("C:/Users/Jp.SANDRO-HP/Desktop")