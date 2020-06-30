import cv2
import face_recognition
import numpy as np

video_capture = cv2.VideoCapture("C:\\Users\\user1\\Desktop\\k-enter.mp4")

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
obama_image = face_recognition.load_image_file("obama.JPG")
obama_face_encoding= face_recognition.face_encodings(obama_image)[0]
emma_image = face_recognition.load_image_file("emma.jpg")
emma_face_encoding = face_recognition.face_encodings(emma_image)[0]
jaeseok_image = face_recognition.load_image_file("jaeseok.jpg")
jaeseok_face_encoding = face_recognition.face_encodings(jaeseok_image)[0]
seho_image = face_recognition.load_image_file("seho.jpg")
seho_face_encoding = face_recognition.face_encodings(seho_image)[0]
frame_count = 0
# Create arrays of known face encodings and their names
known_face_encodings = [
    biden_face_encoding,
    obama_face_encoding,
    emma_face_encoding,
    jaeseok_face_encoding,
    seho_face_encoding
] 
known_face_names = [
    "Biden",
    "Obama",
    "EmmaStone",
    "YuJaeSeok",
    "JoSeHo"
]
# Save Features from Known Image
#print(known_face_encodings)
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    if(video_capture.get(cv2.CAP_PROP_POS_FRAMES) == video_capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        video_capture.open("C:\\Users\\user1\\Desktop\\k-enter.mp4")

    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame and frame_count%25 == 0 :
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and min(face_distances)<0.5:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)
    frame_count = frame_count+1
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(33)>0:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
