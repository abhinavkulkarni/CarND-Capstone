#!/usr/bin/env python

import os
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from PIL import Image

MODEL_CKPT_PATH = '../models'
THRESHOLD = 0.25
MIN_DETECTION_SIZE = 10


class TLDetector(object):

    def __init__(self, model='ssd_mobilenet_v1_coco_2017_11_17'):
        """Loads the object detector
        Args:
            model: model to be used
                   (ssd_mobilenet_v1_coco_2017_11_17 or ssd_inception_v2_coco_2017_11_17)
        """
        path_to_ckpt = os.path.join(MODEL_CKPT_PATH, model, 'frozen_inference_graph.pb')
        self.graph = tf.Graph()

        # configuration for possible GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.graph.as_default():
            graph_def = tf.GraphDef()

            with tf.gfile.GFile(path_to_ckpt, 'rb') as f:
                serialized_graph = f.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.graph, config=config)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def get_detection(self, image, viz=False):
        """Determines the locations of the traffic light in the image
        Args:
            image: camera image
            viz: whether or not to vizualize the detection
        Returns:
            list of integers: coordinates [x_left, y_up, x_right, y_down]
        """

        with self.graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = \
                self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                              feed_dict={self.image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            cls = classes.tolist()

            # Find the first occurence of traffic light detection id = 10
            idx = next((i for i, v in enumerate(cls) if v == 10.), None)
            # If there is no detection
            if idx is None:
                box = [0, 0, 0, 0]
                print('no detection!')

            # detection with less confidence
            elif scores[idx] <= THRESHOLD:
                box = [0, 0, 0, 0]
                print('low confidence:', scores[idx])

            # If there is a detection and its confidence is high enough
            else:
                height, width = image.shape[0:2]
                box = boxes[idx]
                box = np.array([int(box[0]*height), int(box[1]*width),
                                int(box[2]*height), int(box[3]*width)])
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]

                # bells and whistles
                # ignore if the box is too small
                if (box_h < MIN_DETECTION_SIZE) or (box_w < MIN_DETECTION_SIZE):
                    box = [0, 0, 0, 0]
                    print('box too small!', box_h, box_w)

                else:
                    print(box)
                    print('localization confidence: ', scores[idx])

                    if viz:
                        cv2.rectangle(img, (box[1], box[0]),
                                      (box[3], box[2]), (0, 255, 0), 3)
                        cv2.imshow('Image', image)
                        cv2.waitKey()

        return box


if __name__ == '__main__':
    import argparse
    import timer

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='../data/sample/sim',
                        help="path to some sample images")
    parser.add_argument("--model", default='ssd_mobilenet_v1_coco_2017_11_17',
                        help="ssd_mobilenet_v1_coco_2017_11_17 "
                             "or ssd_inception_v2_coco_2017_11_17")

    parser.add_argument('--viz', dest='viz', action='store_true')

    args = parser.parse_args()

    image_paths = glob(os.path.join(args.image_dir, '*.jpg'))

    clock = timer.Timer()

    detector = TLDetector(model=args.model)
    for i, image_path in enumerate(image_paths, start=1):
        img = Image.open(image_path)
        img = np.asarray(img, dtype="uint8")
        img_copy = np.copy(img)
        print('Processing (%d/%d): %s' % (i, len(image_paths), image_path))

        # ignore the first image for calculating average processing time
        # since first image takes too long
        if i == 1:
            detector.get_detection(img, args.viz)
            continue

        clock.tic()
        detector.get_detection(img, args.viz)
        clock.toc()

    print('Average localization time for %d images = %.3f s' %
          (clock.calls, clock.average_time))
