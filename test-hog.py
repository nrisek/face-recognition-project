import face_recognition
import os
import cv2
import pickle
import glob
from collections import Counter

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 3
MODEL = "hog" # hog or cnn

def test():
    known_faces = []
    known_names = []
    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
            encoding = pickle.load(open(f"{KNOWN_FACES_DIR}/{name}/{filename}", "rb"))
            known_faces.append(encoding)
            known_names.append(int(name))

    video_dir = 'videos/test'
    video_files = glob.glob(os.path.join(video_dir, '*.avi'))

    for testVideo in video_files:
        video = cv2.VideoCapture(testVideo)
        print("---Video start---")
        total_matches = 0
        no_matches = 0
        all_matches = []

        while video.isOpened():
            ret, image = video.read()

            if not ret:
                break

            locations = face_recognition.face_locations(image, model=MODEL)
            encodings = face_recognition.face_encodings(image, locations)
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                match = None
                if True in results:
                    match = known_names[results.index(True)]
                    all_matches.append(match)
                    total_matches += 1
                
                if match is None:
                    no_matches += 1
                    total_matches += 1

                top_left = (face_location[3], face_location[0])
                botton_right = (face_location[1], face_location[2])

                color = [183, 250, 0]
                cv2.rectangle(image, top_left, botton_right, color, FRAME_THICKNESS)

                top_left = (face_location[3], face_location[2])
                botton_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, botton_right, color, cv2.FILLED)
                cv2.putText(image, str(match), (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
            
            cv2.namedWindow('Videos', cv2.WINDOW_NORMAL) 
            cv2.resizeWindow('Videos', 1200, 860)
            cv2.imshow("Videos", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        print(f"Total matches = {total_matches}")
        print(Counter(all_matches))
        print(f"No matches = {no_matches}")
        print("---Video end---")
        
test()             



