import numpy as np
import cv2
from ModulKlasifikasiCitraCNN2 import LoadModel  

def PictureFromCam():
    model = LoadModel("CardModelWeight.h5")  # Load the model using your function
    video = cv2.VideoCapture(2)
    
    card_names = [
     "Closed Card","Two Club","Three Club","Four Club","Five Club","Six Club","Seven Club","Eight Club","Nine Club","Ten Club","Jack Club","Queen Club","King Club","Ace Club",
     "Two Heart","Three Heart","Four Heart","Five Heart","Six Heart","Seven Heart","Eight Heart","Nine Heart","Ten Heart","Jack Heart","Queen Heart","King Heart","Ace Heart",
     "Two Spade","Three Spade","Four Spade","Five Spade","Six Spade","Seven Spade","Eight Spade","Nine Spade","Ten Spade","Jack Spade","Queen Spade","King Spade","Ace Spade",
     "Two Diamonds","Three Diamonds","Four Diamonds","Five Diamonds","Six Diamonds","Seven Diamonds","Eight Diamonds","Nine Diamonds","Ten Diamonds","Jack Diamonds","Queen Diamonds","King Diamonds","Ace Diamonds",
     ]
    
    while True:
        check, frame = video.read()
        if not check:
            break

        FrameResult = frame.copy()
        FrameResult2 = frame.copy()

        preprocessed_frame = PreprocessedFrame.preprocess_image(frame)
        processed_frame, corners_list = ProcessedFrame.findContours(preprocessed_frame, FrameResult, draw=True)

        for corners in corners_list:
        # Process each set of corners
            if len(corners) == 4:  # Assuming 4 corners indicate a single card
                wrap_corners = ProcessedFrame.cardWrap(FrameResult2, corners)
            
            # Resize and reshape the wrapped card for prediction
                wrap_corners_resized = cv2.resize(wrap_corners, (128, 128))
                wrap_corners_resized_reshaped = np.expand_dims(wrap_corners_resized, axis=0)
            
            # predictions for each card
                predictions = model.predict(wrap_corners_resized_reshaped)
                predicted_labels = np.argmax(predictions, axis=1)
                predicted_card_names = [card_names[label] for label in predicted_labels]
            
    
            print("Detected cards:", predicted_card_names) #check
            for i, card_name in enumerate(predicted_card_names):
                text_position = (int(corners[0][0]), int(corners[0][1]) - 10)  # Adjust the text position
                cv2.putText(FrameResult, card_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Draw a rectangle
                cv2.polylines(FrameResult, [np.int32(corners)], True, (0, 255, 0), 2)


        cv2.imshow('Webcam Capture', FrameResult)
        cv2.imshow('Processed Frame', preprocessed_frame)
        #cv2.imshow('Processed Frame', processed_frame)
        #cv2.imshow('Wrap Result', wrap_corners)
        
        cv2.waitKey(1)

        key = cv2.waitKey(1)
        if key == ord('c') or key == 27:
            break
    
    video.release()
    cv2.destroyAllWindows()


class PreprocessedFrame:
    
    def preprocess_image(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 42, 89)
        kernel = np.ones((3, 3))
        dial = cv2.dilate(canny, kernel=kernel, iterations=2)
        return dial

class ProcessedFrame:
   
    def findContours(frame, original, draw=False):
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        proper = sorted(contours, key=cv2.contourArea, reverse=True)
        corners_list = []  # List to store allc potential card corners

        for cnt in proper:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, closed=True)

            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, closed=True)

                if len(approx) == 4:
                    corners = np.float32(approx.reshape(4, 2))
                    corners_list.append(corners)  # Store corners for each potential card
                    if draw:
                        cv2.drawContours(original, [approx], -1, (0, 255, 0), 2)
                        for corner in corners:
                            x, y = corner[0], corner[1]
                            cv2.circle(original, (int(x), int(y)), 5, (255, 0, 0), -1)

        return original, corners_list

    
    def cardWrap(frame, corners):
        width, height = 200,300 
        if len(corners) == 4:
            pts1 = np.float32(corners)
            pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgOutput = cv2.warpPerspective(frame, matrix, (width, height))
            print("Before resizing - imgOutput shape:", imgOutput.shape)  # check
            imgOutput = cv2.resize(imgOutput, (128, 128))  
            
            return imgOutput
        else:
            return frame
        
if __name__ == "__main__":
    PictureFromCam()
