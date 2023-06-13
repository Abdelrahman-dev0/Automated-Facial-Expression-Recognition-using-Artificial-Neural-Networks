import os
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from keras.models import model_from_json

# For Ignore warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# dataset train path
train_path = "D:\\Projects\\AI\\dataset\\train"

# dataset test path
test_path = "D:\\Projects\\AI\\dataset\\test"

emotion_labels = sorted(os.listdir(train_path))
# print(emotion_labels)  # ----> ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Creating Batch size and Image inputs Dimensions of our Dataset
batch_size = 64
target_size = (48, 48)
input_shape = (48, 48, 1)  # img_rows, img_columns, color_channels
num_classes = 7

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Found 28709 images belonging to 7 classes.
# Found 7178 images belonging to 7 classes.
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=target_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True)

val_generator = val_datagen.flow_from_directory(
    test_path,
    target_size=target_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')


# Breakdown of all data class variable
def plot_images(img_dir, top=10):
    all_img_dirs = os.listdir(img_dir)
    img_files = [os.path.join(img_dir, file) for file in all_img_dirs][:5]

    plt.figure(figsize=(12, 12))

    for idx, img_path in enumerate(img_files):
        plt.subplot(5, 5, idx + 1)
        img = plt.imread(img_path)
        plt.tight_layout()
        plt.imshow(img, cmap='gray')


# Data Visualization
print("\n")
print("Train classes")
emotions = os.listdir(train_path)
for emotion in emotions:
    count = len(os.listdir(f'D:\\Projects\\AI\\dataset\\train\\{emotion}'))
    print(f'{emotion} faces={count}')

print("\n")
print("Test classes")
emotions = os.listdir(test_path)
for emotion in emotions:
    count = len(os.listdir(f'D:\\Projects\\AI\\dataset\\test\\{emotion}'))
    print(f'{emotion} faces={count}')

print("\n")
emotions = os.listdir(train_path)
values = [len(os.listdir(f'D:\\Projects\\AI\\dataset\\train\\{emotion}')) for emotion in emotions]
fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(emotions, values, color='grey',
        width=0.4)

plt.xlabel("Emotions")
plt.ylabel("No. of images")
plt.title("Train dataset overview")
plt.show()

emotions = os.listdir(test_path)
values = [len(os.listdir(f'D:\\Projects\\AI\\dataset\\test\\{emotion}')) for emotion in emotions]
fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(emotions, values, color='grey',
        width=0.4)

plt.xlabel("Emotions")
plt.ylabel("No. of images")
plt.title("Test dataset overview")
plt.show()

# Artificial Neural Networks (ANN) model
model = Sequential()
model.add(layers.Dense(16, input_shape=input_shape, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Flatten())
model.add(layers.Dense(7, activation='softmax'))

model.summary()

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 50
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

# Training the Model
history = model.fit(x=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=num_epochs, batch_size=batch_size,
                    validation_data=val_generator, validation_steps=STEP_SIZE_VAL)

# Save Model
models.save_model(model, 'ANN.h5')

# Evaluate Model
ann_score = model.evaluate_generator(val_generator, steps=STEP_SIZE_VAL)
print('Test loss: ', ann_score[0])
print('Test accuracy: ', ann_score[1])

# Show Training History
keys = history.history.keys()
print(keys)


def show_train_history(hisData, train, test):
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Training History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


show_train_history(history, 'loss', 'val_loss')
show_train_history(history, 'accuracy', 'val_accuracy')

model_json = model.to_json()
model.save_weights('model_weights.h5')
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# GUI
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return fr


def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow('Facial Expression Recognization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


try:
    gen(VideoCamera())
except:
    print("Finish")
