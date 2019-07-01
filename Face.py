import numpy as np
import cv2
import pickle

c=0
face_cascade=cv2.CascadeClassifier('data\haarcascade_frontalface_alt2.xml')
recognizer =cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels={"person_name":1}
with open("label.pkl",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}


cap=cv2.VideoCapture(0)


while(True):

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in face:
        
        roi_gray=gray[y:y+h,x:x+w] ##y_cordstart-y_cordheight_Y,x_cordstart-c_cordlength
        roi_color=frame[y:y+h,x:x+w]
        id_,conf=recognizer.predict(roi_gray)
        print(conf)
        if conf>=67:
            
            #print(id_)
            print(labels[id_])
            img=cv2.imread('C:/Users/dania/Desktop/python/Face Recognition/Smile.png',1)
            img2=cv2.imread('C:/Users/dania/Desktop/python/Face Recognition/Poker.png',1)
            cv2.imshow('smile',img)
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            cv2.destroyWindow('Poker')
        
        else:
            cv2.destroyWindow('smile')
            cv2.imshow('Poker',img2)
        img_item="my-image1.png"
       
        cv2.imwrite(img_item,roi_color)

        color=(255,0,0) ##BGR
        stroke=2
        width=x+w
        height=y+h
        #print(str(width)+" "+str(height))
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
