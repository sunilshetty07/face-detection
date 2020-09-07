# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 06:05:41 2020

@author: sunil shetty
"""


#import libraries
import os   
import cv2
import numpy




class VideoCamera(object):
    def __init__(self,index=0):
        self.video=cv2.VideoCapture(0)
        self.index=index
        print(self.video.isOpened())

    def release(self):
        self.video.release()
        
    def get_frame(self,in_grayscale=False):
        _,frame=self.video.read()
        if in_grayscale:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return frame

webcam=VideoCamera()

base_dir = os.path.dirname(__file__)
print(base_dir)
#import the models provided in the OpenCV repository
model = cv2.dnn.readNetFromCaffe(base_dir+'/face_detection_model/deploy.prototxt', base_dir+'/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
#loop through all the files in the folder
name=input('person: ')
folder=base_dir+"/dataset/" + name.lower() #input name
if not os.path.exists(folder):
    os.mkdir(folder)
    counter=0
    timer=0
    while counter<50:
        image=webcam.get_frame()
        image1 = image.copy()
        #accessing the image.shape tuple and taking the first two elements which are height and width
        (h, w) = image.shape[:2]
        #get our blob which is our input image after mean subtraction, normalizing, and channel swapping
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        #input the blob into the model and get back the detections from the page using model.forward()
        model.setInput(blob)
        detections = model.forward()
        count=0
        #Iterate over all of the faces detected and extract their start and end points
        for i in range(0, detections.shape[2]):
          box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")
    
          confidence = detections[0, 0, i, 2]
          #if the algorithm is more than 50% confident that the detection is a face, show a rectangle around it
          if (confidence > 0.5):
              cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
              count=count+1
              frame = image[startY:endY, startX:endX]
          #print("Extracted " + str(count) + " faces from all images")
        key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
        if key == ord("c") or key == ord("C"):
            cv2.imwrite(folder+'/'+ str(counter) + '.jpg',image1)
            counter=counter+1
            print("image of person"+name+" no: "+str(counter)+" is saved.")
        cv2.putText(image,"press c to store photo | press esc to exit",(5,image.shape[0]-5),cv2.FONT_HERSHEY_PLAIN,1.2,(150,24,170),2,cv2.LINE_AA)
        cv2.imshow(name,image)
        if cv2.waitKey(40) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
        #save the modified image to the Output folder
    webcam.release()
    cv2.destroyAllWindows()
else:
    print("this name already exists\nIs this same person or different person")
    inp=input("yes/no:\n")
    inp=inp.lower()
    if inp=='yes':
        counter=0
        timer=0
        for file in os.listdir(folder):
            counter=counter+1
        while True:
            image=webcam.get_frame()
            image1 = image.copy()
            #accessing the image.shape tuple and taking the first two elements which are height and width
            (h, w) = image.shape[:2]
            #get our blob which is our input image after mean subtraction, normalizing, and channel swapping
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            #input the blob into the model and get back the detections from the page using model.forward()
            model.setInput(blob)
            detections = model.forward()
            count=0
            #Iterate over all of the faces detected and extract their start and end points
            for i in range(0, detections.shape[2]):
              box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")
        
              confidence = detections[0, 0, i, 2]
              #if the algorithm is more than 50% confident that the detection is a face, show a rectangle around it
              if (confidence > 0.5):
                  cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                  count=count+1
                  frame = image[startY:endY, startX:endX]
              #print("Extracted " + str(count) + " faces from all images")
            key = cv2.waitKey(1) & 0xFF
    
    	# if the `q` key was pressed, break from the loop
            if key == ord("c") or key == ord("C"):
                cv2.imwrite(folder+'/'+ str(counter) + '.jpg',image1)
                counter=counter+1
                print("image of person "+name+" no: "+str(counter)+" is saved.")
            cv2.putText(image,"press c to store photo | press esc to exit",(5,image.shape[0]-5),cv2.FONT_HERSHEY_PLAIN,1.2,(150,24,170),2,cv2.LINE_AA)
            cv2.imshow(name,image)
            if cv2.waitKey(40) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
            #save the modified image to the Output folder
        webcam.release()
        cv2.destroyAllWindows()
    if inp=='no':
        name=input('person: ')
        folder=base_dir+"/dataset/" + name.lower() #input name
        if not os.path.exists(folder):
            os.mkdir(folder)
            counter=0
            timer=0
            while counter<10:
                image=webcam.get_frame()
                image1 = image.copy()
                #accessing the image.shape tuple and taking the first two elements which are height and width
                (h, w) = image.shape[:2]
                #get our blob which is our input image after mean subtraction, normalizing, and channel swapping
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                #input the blob into the model and get back the detections from the page using model.forward()
                model.setInput(blob)
                detections = model.forward()
                count=0
                #Iterate over all of the faces detected and extract their start and end points
                for i in range(0, detections.shape[2]):
                  box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                  (startX, startY, endX, endY) = box.astype("int")
            
                  confidence = detections[0, 0, i, 2]
                  #if the algorithm is more than 16.5% confident that the detection is a face, show a rectangle around it
                  if (confidence > 0.5):
                      cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                      count=count+1
                      frame = image[startY:endY, startX:endX]
                  #print("Extracted " + str(count) + " faces from all images")
                key = cv2.waitKey(1) & 0xFF
        
        	# if the `q` key was pressed, break from the loop
                if key == ord("c") or key == ord("C"):
                    cv2.imwrite(folder+'/'+ str(counter) + '.jpg',image1)
                    counter=counter+1
                    print("image of person"+name+" no: "+str(counter)+" is saved.")
                    cv2.putText(image,"press c to store photo | press esc to exit",(5,image.shape[0]-5),cv2.FONT_HERSHEY_PLAIN,1.2,(150,24,170),2,cv2.LINE_AA)
                cv2.imshow(name,image)
                if cv2.waitKey(40) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    break
                #save the modified image to the Output folder
            webcam.release()
            cv2.destroyAllWindows()
    else:
        print("pressed esc | incorrect value entered")

print("process completed")
