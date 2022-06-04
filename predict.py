
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image,ImageOps
import numpy as np
import cv2
from tkinter.filedialog import askopenfile
model = load_model('model.h5')

data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)


file = askopenfile(filetypes =[('file selector', '*')])
image = Image.open(str(file.name))
path=cv2.imread(str(file.name))
size = (150, 150)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
list=prediction[0]
classes = np.argmax(prediction, axis = 1)
print(classes)
for i in classes:
    if i==1:
        a="pico"
    else:
        a='raspery'
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(path, 
                a, 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
cv2.imshow("frame",path)