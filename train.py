import face_recognition
import os
import cv2
import pickle
import glob
import shutil
import numpy as np

KNOWN_IDENTITIES_DIR = "known_identities"
FRAME_THICKNESS = 2
FONT_THICKNESS = 3
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5
MODEL = "cnn" #cnn or hog

kernel = np.array([[0, -0.5, 0],
                   [-0.5, 3, -0.5],
                   [0, -0.5, 0]])
#delete all dirs and subdirs in known_faces
for item in os.listdir(KNOWN_FACES_DIR):
    item_path = os.path.join(KNOWN_FACES_DIR, item)

    if os.path.isdir(item_path):
        try:
            shutil.rmtree(item_path)
        except OSError as e:
            print(f"Error: {item_path} - {e}")

#delete all images of known identities
for file in os.listdir(KNOWN_IDENTITIES_DIR):
    if file.endswith(".png"):
        os.remove(os.path.join(KNOWN_IDENTITIES_DIR, file))

def train():
    known_faces = []
    known_names = []

    next_id = 0

    video_dir = 'videos/train'
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    for trainVideo in video_files:
        video = cv2.VideoCapture(trainVideo)
        while video.isOpened():
            ret, image = video.read()

            if not ret:
                break
            
            locations = face_recognition.face_locations(image, model=MODEL)
            encodings = face_recognition.face_encodings(image, locations)

            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.filter2D(image, -1, kernel)

            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                match = None
                if True in results:
                    match = known_names[results.index(True)]
                else:
                    match = str(next_id)
                    next_id += 1
                    known_names.append(match)
                    known_faces.append(face_encoding)
                    os.mkdir(f"{KNOWN_FACES_DIR}/{match}")
                    pickle.dump(face_encoding, open(f"{KNOWN_FACES_DIR}/{match}/{match}.pkl", "wb"))

                    top_left = (face_location[3], face_location[0])
                    botton_right = (face_location[1], face_location[2])
                    color = [183, 250, 0]
                    cv2.rectangle(image, top_left, botton_right, color, FRAME_THICKNESS)
                    top_left = (face_location[3], face_location[2])
                    botton_right = (face_location[1], face_location[2] +22)
                    cv2.rectangle(image, top_left, botton_right, color, cv2.FILLED)
                    cv2.putText(image, str(match), (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
                    cv2.imwrite(f"{KNOWN_IDENTITIES_DIR}/{match}.png", image) 

            cv2.imshow('Video', image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

train()

                    



