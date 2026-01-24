#!/usr/bin/env python
# coding: utf-8

# In[4]:


import face_recognition
import cv2
import pickle 
import numpy as np
class FaceRecognizer:
    def __init__(self, encodings_file):
        self.encodings_file = encodings_file
        with open(self.encodings_file, "rb") as f:
            self.known_encodings, self.known_names = pickle.load(f)
      
        self.update_count = 0
        self.replace_limit = 2

        
    def update_encodings(self, name, new_vectors, max_count = 50):
        indices = [i for i, n in enumerate(self.known_names) if n == name]
        
        if not indices:
            return

        allowed = self.replace_limit - self.update_count
        vectors_to_add = new_vectors[:allowed]

        for vec in vectors_to_add:
            self.known_encodings.append(vec)
            self.known_names.append(name)
            self.update_count += 1


        person_indices = [i for i, n in enumerate(self.known_names) if n == name]
        if len(person_indices) > max_count:
            extra = len(person_indices) - max_count
            for i in sorted(person_indices[:extra], reverse=True):
                del self.known_encodings[i]
                del self.known_names[i]
            
        with open(self.encodings_file, "wb") as f:
            pickle.dump((self.known_encodings, self.known_names), f)
        #print(f"Đã thêm vector cho {name}, hiện có {len([n for n in self.known_names if n == name])} vector.")

            
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

                if similarity >= 80 and matches[best_match_index] and self.replace_limit > self.update_count:
                    self.update_encodings(name, [face_encoding])
                    
            results.append(((top, right, bottom, left), name))

            # Vẽ khung và tên lên ảnh
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, display_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)
        return image, results
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




