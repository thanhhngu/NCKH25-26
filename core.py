import face_recognition
import cv2
import pickle 
import numpy as np
import requests
import json
from collections import defaultdict

class FaceRecognizer:
    def __init__(self, encodings_file):
        self.encodings_file = encodings_file
        with open(self.encodings_file, "rb") as f:
            self.known_encodings, self.known_names = pickle.load(f)
      
        self.update_count = 0
        self.replace_limit = 2
        self.pending_updates = []

    def send_unlock_signal(self, name, similarity):
        url = ""  # Thay thế bằng URL thực tế
        data = {
            "status": "unlock",
            "user": name,
        }
        if url:
            try:
                response = requests.post(url, json=data)
            except Exception as e:
                print("error", e)
        
   
    def update_encodings(self, name, new_vectors, max_count=50):
        for vec in new_vectors:
            self.known_encodings.append(vec)
            self.known_names.append(name)
        #sort value in encodings_file
        grouped = {}
        for n, vec in zip(self.known_names, self.known_encodings):
            if n not in grouped:
                grouped[n] = []
            grouped[n].append(vec)
                
        new_encodings = []
        new_names = []
        # cut
        for n, vecs in grouped.items():
            if len(vecs) > max_count:
                vecs = vecs[-max_count:]
            new_encodings.extend(vecs)
            new_names.extend([n] * len(vecs))
                
        self.known_encodings = new_encodings
        self.known_names = new_names    
        with open(self.encodings_file, "wb") as f:
            pickle.dump((self.known_encodings, self.known_names), f)

        print(f"Đã thêm vector cho {name}, hiện có {len([n for n in self.known_names if n == name])}/{max_count} vector.")
    
    def recognize(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(image_rgb)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

        # top, right, bottom, left = face_locations[0]
        # face_encoding = face_encodings[0]
        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

            name = "Unknown"
            similarity = 0
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                similarity = (1 - face_distances[best_match_index]) * 100

                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    display_name = f"{name} ({similarity:.2f}%)"
                else:
                    display_name = name

                if similarity >= 70 and matches[best_match_index] and len(self.pending_updates) < 3:
                #     self.update_encodings(name, [face_encoding])
                    self.pending_updates.append((name, face_encoding))
                if similarity >= 70 and matches[best_match_index]:
                    pass
                    # self.send_unlock_signal(name, similarity)
            results.append(((top, right, bottom, left), name))

            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, display_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)
        return image, results
    
    def recognize_url(self, url):
        cap = cv2.VideoCapture(0)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            frame_count += 1
            if frame_count % 3 == 0:   
                frame, results = self.recognize(frame)
                cv2.imshow("ESP32-CAM Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        if self.pending_updates:
            #grouped form { "thanh": [vec1, vec2], "an": [vec3], ... }.
            grouped = defaultdict(list)
            for name, vec in self.pending_updates:
                grouped[name].append(vec)

            for name, vectors in grouped.items():
                self.update_encodings(name, vectors)

            self.pending_updates.clear()
            print("done update encodings for all.")


# recognizer = FaceRecognizer("encodings.pkl")
# image, results = recognizer.recognize("C:/Users/ADMIN/OneDrive/Desktop/z7415063940931_4992fc5ac76a26366f13513541be3f49.jpg")
# cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
# cv2.imshow("Face Recognition", image)
# cv2.resizeWindow("Face Recognition", 800, 600)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for loc, name in results:
#     print("Location:", loc, "=>", name)


# In[ ]:




