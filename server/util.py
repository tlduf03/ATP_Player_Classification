import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os
# Mission : How the UI send an imgae to backend server? Base64 (drag&drop)

#private variables
__class_name_to_numbers = {}
__class_number_to_name = {}
__model = None

def classify_images(image_base_64, file_path=None):
    """
    Classify an input image
    The input is either base64 string or file_path
    """
    imgs = get_cropped_image_if_2_eyes(file_path, image_base_64)
    
    result = []
    for img in imgs:
        #scale a single image
        scaled_img = cv2.resize(img,(32,32))
        #transformed image
        img_har = w2d(img,'db1',5)
        scaled_img_har = cv2.resize(img_har, (32,32))

        #vectorize and stack the images
        combined_image = np.vstack((scaled_img.reshape(32*32*3,1), scaled_img_har.reshape(32*32,1)))
        
        len_of_arr = 32*32*3+32*32 #4096
        final = combined_image.reshape(1,len_of_arr).astype(float)
        # print('We got final combinded image!')
        result.append({
            'class':class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final)*100,2).tolist()[0], #probability of input images against other classes
            'class_dictionary': __class_name_to_numbers
        })
    return result
    
def load_saved_artifacts():
    """
    Load the saved artifacts
    """
    print("Loading saved artifacts...")
    global __class_name_to_numbers
    global __class_number_to_name
    with open('server/artifacts/class_dictionary.json', 'r') as f:
        __class_name_to_numbers = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_numbers.items()} 
    
    global __model
    if __model is None:
        print("Loading model...")
        __model = joblib.load('server/artifacts/saved_model.pkl')
    
    print("Loaded artifacts successfully!")
    
        
def get_cv2_image_from_base64_string(b64str):
    """
    Convert encoded base64 image to cv2 image
    """
    encoded_data = b64str.split(',')[1]
   
    nparr = np.frombuffer(base64.b64decode(bytes(encoded_data, "UTF-8")))
    # nparr = np.frombuffer(base64.b64encode(encoded_data),np.uint8)
  
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    get the face images with 2 eyes
    """
    
    face_cascade = cv2.CascadeClassifier('./server/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./server/opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2: #captures only if 2 eyes exist
                cropped_faces.append(roi_color)
    return cropped_faces

def class_number_to_name(class_number):
    """
    Get the name of the class number
    """
    return __class_number_to_name[class_number]

def get_b64_test_img_for_federer():
    with open('./server/b64_test_federer.txt') as f:
        return f.read()

if __name__ == '__main__':
    
    load_saved_artifacts()
    print(classify_images(get_b64_test_img_for_federer(), file_path=None)) # should output label '4' from our dictionary