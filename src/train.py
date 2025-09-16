import tensorflow as tf
import cv2
import numpy as np

# Uncomment the model you want to use:
# model = tf.keras.models.load_model("../models/mobilenetv2_model.h5")
model = tf.keras.models.load_model("../models/tiny_cnn_model.h5")

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Predicting the emotion
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]

    # Display the prediction on the frame
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
