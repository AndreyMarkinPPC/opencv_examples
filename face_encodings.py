import face_recognition

# load image
image = face_recognition.load_image_file("data/model.jpg")

# generate face encodings
face_encodings = face_recognition.face_encodings(image)
if len(face_encodings) == 0: 
    print("No faces found.")
else:
    print(face_encodings[0])
