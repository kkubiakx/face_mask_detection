import time
import tensorflow as tf
from object_detection.utils import label_map_util, config_util, visualization_utils as vis
from object_detection.builders import model_builder
import os
import cv2
import numpy as np

tf.get_logger().setLevel('ERROR')

MODELS_DIR = os.path.join(os.getcwd(), 'exported-models')
MODEL_NAME = 'mobilenetv2_final_45k'
CKPT_PATH = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
CONFIG_PATH = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
LABELS_PATH = os.path.join(os.getcwd(), 'annotations/label_map.pbtxt')

@tf.function
def detect_fn(image):
    image, shapes = model.preprocess(image)
    predictions = model.predict(image, shapes)
    detections = model.postprocess(predictions, shapes)

    return detections, predictions, tf.reshape(shapes, [-1])


configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
model_config = configs['model']
model = model_builder.build(model_config=model_config, is_training=False)
checkpoint = tf.compat.v2.train.Checkpoint(model=model)
checkpoint.restore(os.path.join(CKPT_PATH, 'ckpt-0')).expect_partial()

categories = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=False)
cam_capture = cv2.VideoCapture(0)

i = 0
start = time.time()

while True:
    retval, img = cam_capture.read()
    
    img_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    detections, predictions, shapes = detect_fn(img_tensor)

    img_with_detections = img.copy()

    vis.visualize_boxes_and_labels_on_image_array(
        img_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + 1).astype(int),
        detections['detection_scores'][0].numpy(),
        categories,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=0.7,
        agnostic_mode=False
    )

    cv2.imshow('MASK DETECTION', cv2.resize(img_with_detections, (800,600)))
    i += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print(time.time() - start)
print(i)
cam_capture.release()
cv2.destroyAllWindows()
