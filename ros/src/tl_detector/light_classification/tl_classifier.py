from styx_msgs.msg import TrafficLight
from scripts.tl_classifier import TLClassifier as tlc
from scripts.tl_detector import TLDetector as tld

CLASSES = ['Red', 'Yellow', 'Green', 'NoTrafficLight']


class TLClassifier(object):
    def __init__(self, model):
        self.detector = tld()
        self.classifier = tlc(model)
        self.classifier_dict = {
            'Red': TrafficLight.RED,
            'Yellow': TrafficLight.YELLOW,
            'Green': TrafficLight.GREEN,
            'NoTrafficLight': TrafficLight.UNKNOWN
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # First get the bounding box from SSD detector
        bbox = self.detector.get_detection(image)
        if sum(bbox)==0:
            return TrafficLight.UNKNOWN
        else:
            cropped_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
#             return TrafficLight.GREEN
            result = self.classifier.get_classification(cropped_image)
            return self.classifier_dict[result]
        return CLASSES[3]