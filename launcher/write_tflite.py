import tensorflow as tf
import os
import cv2
  # Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_saved_model("trials/1_mobile_net_v2/tflite_graph/saved_model")
model_no_quant_tflite = converter.convert()

# Save the model to disk
open("mobileNet-v2-noquant.tflite", "wb").write(model_no_quant_tflite)

image_name_list = os.listdir("duckietownDB/frames")
images = []
for item in image_name_list:
  images.append(cv2.imread(os.path.join("duckietownDB/frames", item)))

# Convert the model to the TensorFlow Lite format with quantization
def representative_dataset():
  for image in images:
    yield([image.reshape(240, 320,3)])
# Set the optimization flag.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
open("mobileNet-v2.tflite", "wb").write(model_tflite)