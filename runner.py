import os
import matplotlib.pyplot as plt
import numpy as np
from six import BytesIO
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

TYPE="1_mobile_net_v2"
# TYPE="3_efficient_det_d0"
# TYPE="4_retina_net_50"
# TYPE="5_efficient_det_d1"
# TYPE="6_faster_rcnn"

CKPT_DICT = {
"1_mobile_net_v2":"ckpt-16",
"3_efficient_det_d0":"ckpt-19",
"4_retina_net_50":"ckpt-17",
"5_efficient_det_d1":"ckpt-18",
"6_faster_rcnn":"ckpt-16"
}

#! Global Configurations


def load_image_into_numpy_array(path):
    """ Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        path: the file path to the image

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


pipeline_config = 'trials/'+TYPE+'/config/pipeline.config'
model_dir = 'trials/'+TYPE+'/final_model'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
ckpt.restore(os.path.join(model_dir, CKPT_DICT[TYPE])).expect_partial()

# Setup Detection model
detect_fn = get_model_detection_function(detection_model)

# Setup Label Map
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(
    label_map, use_display_name=True)

image_path = 'image.png'
image_np = load_image_into_numpy_array(image_path)


input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

print(detections)

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.30,
    agnostic_mode=False,
)
print("category_index {}".format(category_index))
plt.figure(figsize=(12, 16))
plt.imshow(image_np_with_detections)
plt.savefig('image_with_detection.png')