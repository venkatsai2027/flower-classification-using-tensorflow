class_names =['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
num_classes = 5
import tensorflow as tf
from tensorflow import keras
model1=tf.keras.models.load_model('with_augmentation.h5')

import gradio as gr

# Use This Below Code to Take Input From Local Library
def predict_image(img):
  img_4d=img.reshape(-1,180,180,3)
  prediction=model1.predict(img_4d)[0]
  return {class_names[i]: float(prediction[i]) for i in range(5)}
image = gr.inputs.Image(shape=(180,180))
label = gr.outputs.Label(num_top_classes=5)
gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')


# Use This Below Commented Code to Take Input From Your Webcam

# def predict_image(img):
#   img_4d=img.reshape(-1,180,180,3)
#   prediction=model1.predict(img_4d)[0]
#   return {class_names[i]: float(prediction[i]) for i in range(5)}
# image = gr.inputs.Image(source="webcam",shape=(180,180))
# label = gr.outputs.Label(num_top_classes=5)
# gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')