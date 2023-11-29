import tensorflow as tf
import tf2onnx

model =  tf.keras.models.load_model('./models_h5/COVID19/covid_classifier_model.h5')
spec = (tf.TensorSpec((None, 200, 200, 3), tf.float32, name="input"),)
output_path = 'covid_classifier_model_opset13.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)