print("loading libraries...")
import face_recognition

print("loading images...")
image_of_person_1 = face_recognition.load_image_file("data/photo_2020-01-03_01-09-17.jpg")
image_of_person_3 = face_recognition.load_image_file("data/model.jpg")
# image_of_person_4 = face_recognition.load_image_file("data/unknown.jpg")

print("generating encodings...")
p_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]
p_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]
# face_locations = face_recognition.face_locations(image_of_person_4)
# p_4_face_encoding = face_recognition.face_encodings(image_of_person_4,face_locations,num_jitters=4)
print("face_encoding_1...")
print(p_1_face_encoding)
print("face_encoding_2...")
print(p_3_face_encoding)
print("face_encoding_3...")
# print(face_locations)
# print(p_4_face_encoding)

# create a list of know face encodings
known_face_encodings = [
    p_1_face_encoding,
    p_3_face_encoding
]
print("loading unknown face...")
unknown_face = face_recognition.load_image_file("data/unknown.jpg")
unknown_face_locations = face_recognition.face_locations(unknown_face)
print(unknown_face_locations)
print("generating unknown face encodings...")
unknown_face_encoding = face_recognition.face_encodings(unknown_face,unknown_face_locations,num_jitters=4)
print(unknown_face_encoding)
results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
print(results)
# for face in unknown_face_encoding:
#     results = face_recognition.compare_faces(known_face_encodings,\
#                                    unknown_face_encoding,\
#                                    tolerance=0.6)
#     name = "Unknown"
# 
#     if results[0]: 
#         name = "am_1"
#     elif results[1]:
#         name = "model"
# 
#     print(f"Found {name} in the photo!")
