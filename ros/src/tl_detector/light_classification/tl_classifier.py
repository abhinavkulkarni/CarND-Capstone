from styx_msgs.msg import TrafficLight
from scripts.tl_classifier import TLClassifier as tlc

class TLClassifier(object):
    def __init__(self, model):  # TODO: add default model location
        self.classifier = tlc(model)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        return self.classifier.get_classification(image)
