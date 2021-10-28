import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
import cv2
import numpy as np


croppedImagepath = "images/" + "crop1.jpeg"


class DetectNumberPlate:
    def __init__(self, path_to_frozen_graph, path_to_label):
        self.path_to_label = path_to_label
        self.path_to_frozen_graph = path_to_frozen_graph

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          self.od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.path_to_frozen_graph, 'rb') as fid:
            self.serialized_graph = fid.read()
            self.od_graph_def.ParseFromString(self.serialized_graph)
            tf.import_graph_def(self.od_graph_def, name='')

        self.category_index = label_map_util.create_category_index_from_labelmap(self.path_to_label, use_display_name=True)

        print("all set")

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


    def predict(self, image_path):
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.

        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        # cv2.imshow('images', image_np)
        #
        # cv2.waitKey(0)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped_image = self.get_bounding_box(image, output_dict['detection_boxes'], image_path)
        return cropped_image

    def get_bounding_box(self, image, boxes, image_path):
        (H, W) = image.shape[:2]
        k = 0
        for plateBox in boxes:
            # Draw the plate box rectangle in red
            # scale the bounding box from the range [0, 1] to [W, H]

            (startY, startX, endY, endX) = plateBox
            startX = int(startX * W)
            startY = int(startY * H)
            endX = int(endX * W)
            endY = int(endY * H)
            k = k + 1

            # croppedimage = crop(imagePath, (startX, startY, endX, endY), croppedImagepath)


            image_obj = Image.open(image_path)

            cropped_image = image_obj.crop((startX, startY, endX, endY))
            cropped_image = cropped_image.convert("L")
            # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            cropped_image.save(croppedImagepath)
            cropped_image_cv = cv2.imread(croppedImagepath)
            return cropped_image_cv


