#A Gender and Age Detection program by Mahesh Sawant
#Modified for better real-time detection

import cv2
import math
import argparse
import time

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

# Load models
print("Loading models...")
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
print("Models loaded successfully!")

# Initialize webcam
print("Starting webcam...")
video=cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam. Please check your camera connection.")
    exit()

padding=20
frame_count = 0
start_time = time.time()
fps = 0

print("Real-time Gender and Age Detection started!")
print("Press 'q' to quit")

while True:
    # Read frame
    hasFrame, frame = video.read()
    if not hasFrame:
        print("Error: Could not read from webcam")
        break
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    # Detect faces
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    # Display FPS
    cv2.putText(resultImg, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add instructions
    cv2.putText(resultImg, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if not faceBoxes:
        cv2.putText(resultImg, "No face detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Real-time Gender and Age Detection", resultImg)
    else:
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                      min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                      :min(faceBox[2]+padding, frame.shape[1]-1)]

            try:
                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                
                # Gender detection
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                
                # Age detection
                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                
                # Display results
                label = f"{gender}, {age}"
                cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            except:
                # Handle errors in processing the face
                cv2.putText(resultImg, "Error processing face", (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        
        # Show the result
        cv2.imshow("Real-time Gender and Age Detection", resultImg)
    
    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or ESC
        break

# Clean up
video.release()
cv2.destroyAllWindows()
print("Program ended")
