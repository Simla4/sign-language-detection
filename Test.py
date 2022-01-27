###### LIBRARIES #######
import numpy as np
import cv2
import pickle

from keras.models import model_from_json

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65
cameraNo = 0
#####################################

#### KAMERA NESNESİNİ OLUŞTURDUK
cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)

#### EĞİTTİĞİMİZ MODELİ ÇAĞIRIYORUZ

json_file = open('model_trained.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Model yüklendi")

#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (28, 28))
    img = preProcessing(img)
    cv2.imshow("Processsed Image", img)
    img = img.reshape(1, 28, 28, 1)
    #### TAHMİN
    classIndex = int(loaded_model.predict_classes(img))
    # print(classIndex)
    predictions = loaded_model.predict(img)
    # print(predictions)
    probVal = np.amax(predictions)
    print(classIndex, probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + "   " + str(probVal),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break