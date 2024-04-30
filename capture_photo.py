import cv2 as cv
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import serial
import os

model = tf.keras.models.load_model('Best_Model.h5')
classes = ['Daun', 'Kertas', 'Logam Ferro', 'Logam non-ferro', 'Plastik']
print("test")

def detect():
    camera = cv.VideoCapture(0)
    res, frame = camera.read()
    if res:
        while True:
            camera.set(cv.CAP_PROP_AUTO_EXPOSURE,0.25)
            
            result, frame = camera.read()
            break
            cv.waitKey(5000)
        processed_frame = cv.resize(frame, (224, 224))
        processed_frame = preprocess_input(processed_frame)
        processed_frame = np.expand_dims(processed_frame, axis=0)  # Menambahkan dimensi batch
        predictions = model.predict(processed_frame)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = classes[predicted_class_index]
        predicted_prob = predictions[0][predicted_class_index]
        cv.putText(frame, f"{predicted_class};{round(predicted_prob, 2)}", (30,30),  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        camera.release()
        cv.imwrite("MyPhoto.png", frame)
        print("Success capture!")
        print(f"Class: {predicted_class} {classes}, prob: {predicted_prob}")
        print(f"{predicted_class_index}")
        #ser = serial.Serial('/dev/ttyUSB0', 9600)
        #ser.write(f"{predicted_class_index}".encode())
        #ser.close()
    else:
        print("Fail to capture")

while True:
    select = input("Gas Foto ga? Y/N\n")
    if select == 'Y' or select == 'y':
        detect()
    else:
        print("Stop running the code")
        break
