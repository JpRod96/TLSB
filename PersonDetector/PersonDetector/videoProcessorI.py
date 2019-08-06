import abc

class VideoProcessorI(abc.ABC):
    @abc.abstractmethod
    def process(self, videoPath):
        pass