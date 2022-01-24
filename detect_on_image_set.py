import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import statistics
from PIL import Image

import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as vis
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')
matplotlib.use('TkAgg')

# Declaring necessary director and file paths
MODELS_DIR = os.path.join(os.getcwd(), 'exported-models')
MODEL_NAME = 'mobilenetv2_final_45k'
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)
IMAGESET_DIR = os.path.join(os.getcwd(), 'testimages')
CKPT_PATH = os.path.join(MODEL_DIR, 'checkpoint/')
CONFIG_PATH = os.path.join(MODEL_DIR, 'pipeline.config')
LABELS_PATH = os.path.join(os.getcwd(), 'annotations/label_map.pbtxt')

detection_times = []
total_times = []

@tf.function
def detect_facemasks(img_tensor):
    img_tensor, shapes = mask_detector.preprocess(img_tensor)
    predictions = mask_detector.predict(img_tensor, shapes)
    detections = mask_detector.postprocess(predictions, shapes)

    return detections, predictions, tf.reshape(shapes, [-1])

# Creating list of paths to test images
img_paths = []
for (path, dirnames, filenames) in os.walk(IMAGESET_DIR):
    img_paths.extend(os.path.join(path, name) for name in filenames)

# Building the detection model from trained checkpoint
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
model_config = configs['model']
mask_detector = model_builder.build(model_config=model_config, is_training=False)
checkpoint = tf.compat.v2.train.Checkpoint(model=mask_detector)
checkpoint.restore(os.path.join(CKPT_PATH, 'ckpt-0')).expect_partial()

# Creating detected class dictionary
categories = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=False)

i = 1
fig = plt.figure()

# Main program loop
for img in img_paths[7:8]:
    img_as_np_array = np.array(Image.open(img))

    # Measuring total times accounting for image processing
    total_start = time.time()
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_as_np_array, 0), dtype=tf.float32)
    
    #Mesuring detection times
    detection_start = time.time()
    detections, predictions, shapes = detect_facemasks(img_tensor)
    detection_times.append(time.time() - detection_start)
    # End of measurement (detection)

    vis.visualize_boxes_and_labels_on_image_array(
        img_as_np_array,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + 1).astype(int),
        detections['detection_scores'][0].numpy(),
        categories,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=0.45,
        agnostic_mode=False
    )
    total_times.append(time.time() - total_start)
    # End of measurement (total)
 
    fig.add_subplot(1,1, i)
    plt.imshow(img_as_np_array)
    i += 1

plt.show()

print(detection_times)
print(total_times)
print(statistics.mean(detection_times))
print(statistics.mean(total_times))
