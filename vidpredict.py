
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image,ImageOps
import numpy as np
import cv2
from tkinter.filedialog import askopenfile
model = load_model('model.h5')

data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)

root=Tk()
root.geometry('500x500')

def vid():
    cap=cv2.VideoCapture(0)
    count = 0
    while True:
        ret,frame=cap.read()
        z=cap.get(10)
        #cv2.imshow("frame",frame)
        cv2.resize(frame,(100,100))
        
        #image=cv2.
        cv2.imwrite("out.jpg" , frame)
        image = Image.open("out.jpg")
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
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
                a="ras"
            else:
                a='pico'
        print(type(a))
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame,a, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51,255,51), 1)
        cv2.putText(frame, 
                    a, 
                    (50, 50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
        
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def img():
    file = askopenfile(filetypes =[('file selector', '*')])
    image = Image.open(str(file.name))
    path=cv2.imread(str(file.name))
    size = (224, 224)
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

    for i in classes:
        if i==1:
            a="potholes"
        else:
            a='normal'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(path, 
                    a, 
                    (50, 50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
    cv2.imshow("frame",path)
b1=Button(root,command=vid,text='video',font=('time',12))
b1.place(x=150,y=250)
b2=Button(root,command=img,text='image',font=('time',12))
b2.place(x=300,y=250)
root.mainloop()