import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("trials/1_mobile_net_v2/tflite_graph/saved_model") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('mobileNet-v2.tflite', 'wb') as f:
  f.write(tflite_model)