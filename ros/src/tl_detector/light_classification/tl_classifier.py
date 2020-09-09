
from styx_msgs.msg import TrafficLight
import yaml
import os
import rospy
import cv2
import tensorflow as tf
import numpy as np

IMAGE_H = 300
IMAGE_W = 300

class TLClassifier(object):
    def __init__(self):
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Tf graph and sess
        self.model_graph = None
        self.session = None

        self.load_graph(os.path.dirname(os.path.realpath(__file__)) + self.config['model'])

        self.classes = {1: TrafficLight.RED,
                2: TrafficLight.YELLOW,
                3: TrafficLight.GREEN,
                4: TrafficLight.UNKNOWN}

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        class_index, probability = self.predict(image)

        if class_index is not None:
            return class_index
        else:
            # If we detected nothing, just return none
            return TrafficLight.UNKNOWN

    def load_graph(self, model_path):
        """Loads a frozen inference graph"""
        # Create config
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # Create model
        self.model_graph = tf.Graph()

        # Create session
        with tf.Session(graph=self.model_graph, config=config) as sess:
            self.session = sess
            graph_def = tf.GraphDef()

            # Load model
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)

                # Import graph definition
                tf.import_graph_def(graph_def, name='')

    def process_image(self, image):
        # Resize image to fit training size
        image = cv2.resize(image, (IMAGE_W, IMAGE_H))

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict(self, image):
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.model_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')

        image = self.process_image(image)

        (boxes, scores, classes) = self.session.run(
            [detection_boxes, detection_scores, detection_classes],
            feed_dict={image_tensor: np.expand_dims(image, axis=0)})

        # Remove unnecessary dimensions
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)

        min_score = 0.5

        # For each bounding box
        for i, box in enumerate(boxes):

            # If detection prob is higher than minimum
            if scores[i] > min_score:

                # Determin class of each detection
                detected_class = self.classes[classes[i]]
                return detected_class, scores[i]

        return None, None