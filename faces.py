import numpy as np
import cv2
import pickle
import os
import time
from PIL import Image
from gtts import gTTS
import urllib.request

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
language='en'

#To check the internet connection
def connect(host='http://www.google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False

def audio(speak):
	netflag=connect()
	if netflag==False:
		print("For Voice Interaction, PLEASE PROVIDE THE INTERNET")
	else:
		output=gTTS(text=speak,lang=language)
		output.save("output.mp3")
		os.system("start output.mp3")
		time.sleep(2)

#Trannig is done here
def tranning():
	image_dir=os.path.join(BASE_DIR,"images")
	face_cascade = cv2.CascadeClassifier(BASE_DIR+'/cascades/data/haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	current_id=0
	label_ids={}
	y_labels=[]
	x_train=[]
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith("png") or file.endswith("jpg"):
				path=os.path.join(root,file)
				label=os.path.basename(root).replace(" ","-").lower()
				#print(label, path)
				if label in label_ids:
					pass
				else:
					label_ids[label]=current_id
					current_id+=1
				id_=label_ids[label]
				#print(label_ids)
				#y_labels.append(label)
				#x_train.append(path)
				pil_image=Image.open(path).convert("L")
				size=(550,550)
				final_image=pil_image.resize(size, Image.ANTIALIAS)
				image_array=np.array(final_image,"uint8")
				print(image_array)
				faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
				for (x,y,w,h) in faces:
					roi=image_array[y:y+h,x:x+w]
					x_train.append(roi)
					y_labels.append(id_)
	#print(y_labels)
	#print(x_train)
	with open("label.pickle",'wb') as f:
		pickle.dump(label_ids,f)
	recognizer.train(x_train,np.array(y_labels))
	recognizer.save("trainner.yml")



#After training
face_cascade=cv2.CascadeClassifier(BASE_DIR+'/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier(BASE_DIR+'/cascades/data/haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier(BASE_DIR+'/cascades/data/haarcascade_smile.xml')

#recognizer =cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()

trainfile = BASE_DIR+'/trainner.yml'
if not os.path.exists(trainfile):
	tranning()
	print('Trannig completed')
else:
	print('Trannig data set is exists')

recognizer.read("trainner.yml")
label={"person_name":1}
with open("label.pickle",'rb') as f:
    og_lables=pickle.load(f)
    labels={v:k for k,v in og_lables.items()}

cap = cv2.VideoCapture(0)
prevname='empty'

while(True):
	
	ret, frame = cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces =face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for(x,y,w,h) in faces:
		#print(x,y,w,h)
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=frame[y:y+h,x:x+w]
		id_, conf=recognizer.predict(roi_gray)
		#print(int(conf))
		if conf<100:
			name=labels[id_]
			print(name)
			font=cv2.FONT_HERSHEY_SIMPLEX
			color=(255, 255, 255)
			stroke=2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			smile=smile_cascade.detectMultiScale(roi_gray)
			eyes=eye_cascade.detectMultiScale(roi_gray)
			
			sayhello="Helllooo"+name 
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			if prevname!=name:
				audio(sayhello)
				prevname=name
				break
			break
		else:
			cv2.putText(frame, "unkown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			print("unkown")
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			unkown_face="I cant recognize you,so please type your name in command prompt so that in can remember you nexttime"
			audio(unkown_face)
			name=input("Enter your name \n")
			saveimginto = BASE_DIR+'/images/'+name
			if not os.path.exists(saveimginto):
				os.makedirs(saveimginto)
				print("Directory " , saveimginto ,  " Created ")
				takingphoto="I am taking a photo of you so please focus on camera"
				audio(takingphoto)
				towards=["please look at the camera","please turn towards your left","please turn towards your right"]
				i=0
				k=0
				for k in range(3):
					audio(towards[k])
					for j in range(5):
						ret, frame = cap.read()
						if ret == False:
							break
						cv2.imwrite(os.path.join(saveimginto,name+str(i)+'.jpg'),frame)
						i+=1
						time.sleep(1)
				succesfulcap="Thanks for co operating, succesfuuullly captured images of your"
				audio(succesfulcap)
				tranning()
				icanxtime="nexttime surly i can recognize you"+name
				audio(icanxtime)


			else:
				print("Directory " , name ,  " already exists")
				nameexist="Given name is already exists so please try other name"
				audio(nameexist)
			break

	cv2.imshow('Face recognizer by ganesh',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()