from fileinput import close
from pyexpat import model
import tkinter
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from tkinter.tix import TEXT
from turtle import width
import cv2
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfile, askopenfilename
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.models import load_model

main = tkinter.Tk()
main.title("LUNG CANCER DETECTION USING CNN") #designing main screen
main.geometry("1300x1200")

global filename
global test_filename

classes = ['adenocarcinoma','benign','squamous']

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")



def test_upload():
   
    global test_filename
    test_filename = askopenfilename(initialdir = "data/test")


def predict():

        global test_filename
        model = load_model('saved-model/ResNet50(200ep)/model.h5')

        lab = ['Adenocarcinoma','Benign','Squamous Cell Carcinoma']

        img = image.load_img(test_filename, target_size=(125, 125))
        img = img_to_array(img)
        img=img/255
        img=np.expand_dims(img,[0])
        answer=model.predict(img)
        y_class = answer.argmax(axis=-1)
        y=" ".join(str(x) for x in y_class)
        y=int(y)
        res = lab[y]
        # messagebox.showinfo("Prediction",f"Your predicted class is {[res]}")

        img = cv2.imread(test_filename,1)
        img = cv2.resize(img, (400,400))
        cv2.putText(img,res, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 2)
        cv2.imshow('Disease Identified as : '+res, img)
        cv2.waitKey(0)

def close():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(main, text='LUNG CANCER DETECTION USING CONVOLUTIONAL NEURAL NETWORKS(CNN)')
title.config(bg='Lavender', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=3)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Lung Cancer Image Dataset", command=upload,width=35,height=2,background='lavender')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  


imageButton = Button(main, text="Upload Test Images", command=test_upload,width=35,height=2,background='lavender')
imageButton.place(x=440,y=550)
imageButton.config(font=font1)

predictButton = Button(main, text="Predict Cancer", command=predict,width=35,height=2,background='lavender')
predictButton.place(x=50,y=650)
predictButton.config(font=font1)

closeButton = Button(main, text="close", command=close,width=35,height=2,background='lavender')
closeButton.place(x=440,y=650)
closeButton.config(font=font1)

#================================================label==============================================

w = Label(main, text ='PHASE 1 PROJECT', font = "times 18",fg='black',width=29,justify="left",foreground='gold',background='black')
w.pack(padx='20px')

msg = Message( main,font = "times 18",text = "Guided by:\nDr.V.Balu(Asst.Professor)\n\nStudent Details: \nK.H.V.VIJAY KAMAL - 11199A128\nK.VIKAS REDDY - 11199M014",
                width=1000,justify="left",foreground='gold',background='black')  
msg.pack(padx='20px')  

w.place(x=875,y=550)
msg.place(x=875,y=590)


abstract = Message( main,font = "times 16",text = "Abstract:\nLung Cancer is one of the leading life taking cancer worldwide. Early detection and treatment are crucial for patient recovery.Medical professionals use histopathological images of biopsied tissue from potentially infected areas of lungs for diagnosis.Most of the time, the diagnosis regarding the types of lung cancer are error-prone and time-consuming.Convolutional Neural networks can identify and classify lung cancer types with greater accuracy in a shorter period, which is crucial for determining patients' right treatment procedure and their survival rate.Benign tissue, Adenocarcinoma, and squamous cell carcinoma are considered in this research work.\n\nKeywords:\nConvolutional Neural Network (CNN), Deep Learning, Lung Cancer,Histopathological Image",
                    width=1000,justify="left",foreground='black',background='white')
abstract.pack(padx='20px') 

abstract.place(x=150,y=190)

#================================================label==============================================

main.config(bg='teal')
main.mainloop()
