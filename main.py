import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.65
FRAME_TICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'# hog
print ("Loading known faces ")
known_faces = []# Известные лица
known_names = []# Известные имена
# known_id_vk = [] #Известные id (Вконтакте)

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print ("processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model = MODEL)
    encoding = face_recognition.face_encodings(image,locations)
    image =cv2.cvtColor( image, cv2.COLOR_RGB2BGR )
    for face_encoding,face_location in zip(encoding,locations):
        resuls = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in resuls:
            match = known_names[resuls.index(True)]
            print(f"Match found: {match} " )

            #top_left = (face_location[3],face_location[0])
            #bottom_right = (face_location[1],face_location[2])

            #color =[0,255,0]
            #cv2.rectangle(image,top_left,bottom_right,color,FRAME_TICKNESS)

            #top_left = (face_location[3],face_location[2])
            #bottom_right = (face_location[1],face_location[2]+22)

            #cv2.rectangle(image,top_left,bottom_right,color,FRAME_TICKNESS)
            #cv2.putText(image,match, (face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),FONT_THICKNESS )
       # else:
            #top_left = (face_location[3],face_location[0])
            #bottom_right = (face_location[1],face_location[2])

            #color =[0,255,0]
            #cv2.rectangle(image,top_left,bottom_right,color,FRAME_TICKNESS)

           # top_left = (face_location[3],face_location[2])
           # bottom_right = (face_location[1],face_location[2]+22)

            #cv2.rectangle(image,top_left,bottom_right,color,FRAME_TICKNESS)
            #cv2.putText(image,"NONE", (face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),FONT_THICKNESS )
            
    
    #cv2.imshow(filename,image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)

