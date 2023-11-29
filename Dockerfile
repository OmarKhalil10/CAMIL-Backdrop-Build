# Use Python37
FROM python:3.7

## Step 1:
# Create a working directory
WORKDIR /app

## Step 2:
# Copy source code to working directory
COPY app.py settings.py models.py requirements.txt vars.env Procfile runtime.txt /app/
COPY templates /app/templates/
COPY static /app/static/
COPY uploads /app/uploads/
COPY models_onnx/LUNG_CANCER/lung_cancer_model_opset13.onnx /app/models_onnx/LUNG_CANCER/lung_cancer_model_opset13.onnx
COPY models_onnx/COVID19/covid_classifier_model_opset13.onnx /app/models_onnx/COVID19/covid_classifier_model_opset13.onnx
COPY models_onnx/PNEUMONIA/cnn_segmentation_pneumonia_opset13.onnx /app/models_onnx/PNEUMONIA/cnn_segmentation_pneumonia_opset13.onnx

## Step 3:
# Install packages from requirements.txt
RUN pip install -r requirements.txt

## Step 4:
# Expose port 8080
EXPOSE 8080

# Run app.py at container launch
CMD ["python", "app.py"]