"""-----------------------------------------------------------------------------------------------------
                    Project - real-time image classification system 
                    ML Model - Using  Convolutional Neural Network (CNN)
                    (Author name - Diksha Kolikal)
--------------------------------------------------------------------------------------------------------
Problem statement: Use a pre-trained CNN model (MobileNetV2) to recognize objects without 
                   having to train from scratch
--------------------------------------------------------------------------------------------------------"""

####################################################################
#    Importing Libraries
####################################################################

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input,decode_predictions

def ImageClassifier():
    # 1) Load pre-trained CNN (ImageNet) 
    model = MobileNetV2(weights="imagenet")

     # 2) Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Error: could not read frame")
            break

         # 3) Preprocess for MobileNetV2: BGR -> RGB, resize to 224x224, scale
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img,(224,224))
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)
        x = preprocess_input(x)

         # 4) Predict
        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds,top=1)[0][0]
        label = f"{decoded[1]}:{decoded[2]*100:.1f}%"

        # 5) Overlay prediction on the frame 
        cv2.putText(frame,label,(16,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Real-time CNN Classification (MobileNetV2)",frame)

         # 6) Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 7) Cleanup
    cap.release()
    cv2.destroyAllWindows()


####################################################################
#   Entry point function 
####################################################################

def main():
    ImageClassifier()

####################################################################
#   Starter 
####################################################################

if __name__=="__main__":
    main()

