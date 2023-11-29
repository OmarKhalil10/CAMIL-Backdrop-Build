import tensorflow as tf
import tf2onnx

model =  tf.keras.models.load_model('./models_h5/LUNG_CANCER/lung_cancer_model.h5')
spec = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input"),)
output_path = 'lung_cancer_model_opset13.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)