import tensorflow as tf
import tf2onnx

# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.dtypes.cast(y_true, dtype = tf.float32)
    y_pred = tf.dtypes.cast(y_pred, dtype = tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

model =  tf.keras.models.load_model('./models_h5/PNEUMONIA/cnn_segmentation_pneumonia.h5', custom_objects={'iou_bce_loss':iou_bce_loss, 'mean_iou': mean_iou, 'iou_loss': iou_loss})
spec = (tf.TensorSpec((None, 256, 256, 1), tf.float32, name="input"),)
output_path = 'cnn_segmentation_pneumonia_opset13.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
