import fnmatch  # For Name Compairing
import os  # For dealing with paths
import warnings  # For ignoring warnings
import cv2  # For Read Images as array from Path
import matplotlib.pyplot as plt  # For Showing Images
import numpy as np  # For dealing with arrays
from keras.models import Sequential  # For ... model
from sklearn.model_selection import train_test_split  # For Splitting Data into Train and Test For Model
from sklearn.preprocessing import LabelEncoder  # For Preprocessing Label Encoding
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense  # For Model Building
from keras.optimizers import Adam  # Model compiler optemizer
from keras.preprocessing.image import ImageDataGenerator  # For Make From 1 image multi images
from keras.utils import to_categorical  # For preprocessing
from tqdm import tqdm  # For Showing the progress of Loading
from skimage import feature  # For Extracting Features From Image array


class CNN:
    # ## 1. initialize Our Environment

    # ### 1.1 Importing Libs We Will Use

    # ### 1.2 Customize Environment

    # For Ignore warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')

    # ### 1.3 Initialize Vars

    Data = []  # Array that carry data from data set
    Label = []  # Array that carry labels from data set
    characters = []  # Characters array
    IMG_SIZE = 100  # Img size = 100 px
    dataDir = 'D:\\Projects\\AI\\dataset\\train'  # dataset directory
    batch_size = 128

    def __init__(self):
        # Calling read_data Fun to Get Data and Label
        self.read_data()

        # ### 2.2 Preprocess The Data

        # Label Encoding our characters labels
        le = LabelEncoder()
        self.Labels = le.fit_transform(self.Label)
        x = self.Labels
        self.Labels = to_categorical(x, len(self.characters))
        self.Data = np.array(self.Data)
        self.Data = self.Data / 255.0

        # ### 2.3 Splites it into Training Data and Test Data

        # Split our data to 75% train and 25% test
        dataTrain, dataTest, LabelsTrain, LabelsTest = train_test_split(self.Data, self.Labels, test_size=0.25,
                                                                        random_state=None)

        # ### 2.4 Make Data 4D For Model
        #

        # Reshape dataTrain dataTest To be a vailed Input for model
        dataTrain = np.array(dataTrain).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        dataTest = np.array(dataTest).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

        # ## 3. Model Section

        # ### 3.1 Build Model

        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(
            100, 100, 1)))  # First Layer in Model is our input layer is convolution for extracting features .
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # Layer which downsamples the input along its spatial dimensions
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                              activation='relu'))  # convolution For extract feature
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2)))  # Layer which downsamples the input along its spatial dimensions
        self.model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same',
                              activation='relu'))  # convolution For extract feature
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2)))  # Layer which downsamples the input along its spatial dimensions
        self.model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same',
                              activation='relu'))  # convolution For extract feature
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2)))  # Layer which downsamples the input along its spatial dimensions
        self.model.add(Flatten())  # Hidden Layer converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(512))  # Hidden Layer in Model has 512 node
        self.model.add(Dropout(
            .5))  # This Layer in Model is Hidden layer which drops % from feature every epoch avoids locals
        self.model.add(Activation('relu'))  # Hidden Activation Layer relu
        self.model.add(Dense(len(self.characters), activation='softmax'))  # Last Layer in Model is our Output Layer

        # ### **3.2 Fit Learning Rate**

        # ### **3.3 Augment imgs while model training**

        # Set img generator for augmentation
        datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                     featurewise_std_normalization=False,
                                     samplewise_std_normalization=False,
                                     zca_whitening=False,
                                     rotation_range=5,
                                     zoom_range=.05,
                                     width_shift_range=.1,
                                     height_shift_range=.1,
                                     horizontal_flip=False,
                                     vertical_flip=False)

        # ### 3.4 Compile And  Fit Model

        self.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

        self.history = self.model.fit(datagen.flow(dataTrain, LabelsTrain,
                                                   batch_size=self.batch_size,
                                                   seed=27,
                                                   shuffle=False), epochs=150, validation_data=(dataTest, LabelsTest),
                                      verbose=1)

        self.model.evaluate(dataTest, LabelsTest, batch_size=self.batch_size)

        # ### 3.5 Save Model

        self.model.save("Last_CNN.h5")
        self.show_plots()

    # ###  plots
    def show_plots(self):
        plt.plot(self.history.history['loss'])  # loss curve
        plt.title('CNN Model Loss Curve')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['accuracy'])  # accuracy curve
        plt.title('CNN Model Accuracy Curve')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

    def read_data(self):
        self.characters.clear()
        for character in os.listdir(self.dataDir):
            self.characters.append(character)
            for img in tqdm(os.listdir(os.path.join(self.dataDir, character))):
                if fnmatch.fnmatch(img, '*jpg'):
                    label = character
                    path = os.path.join(self.dataDir, character, img)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.Data.append(np.array(img))
                    self.Label.append(str(label))

    # Function That View Image just send path
    def show_img(self, img):
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()

    # Function That prepare data to be rcognized
    def prepare(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        return new_img.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

    # Function That Recognize character from the img u sent
    def recognize(self, img):
        prediction = self.model.predict([self.prepare(img)])
        index = np.argmax(prediction)
        # print("CNN this image for : ", self.characters[index])
        return self.characters[index]

    def Validate(self):
        # model = models.load_model("Last.h5")
        validateDataDir = "D:\\Projects\\AI\\dataset\\test"
        TP = os.path.join(validateDataDir)
        for file in os.listdir(TP):
            P = os.path.join(TP, file)
            for img in os.listdir(P):
                try:
                    print(os.path.join(TP, file))
                    self.recognize(os.path.join(TP, file, img))
                except Exception as e:
                    print("Some Thing Went Wrong")


class ANN:
    # ### 1.2 Customize Environment

    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')

    # ### 1.3initialize Vars

    X = []
    Z = []
    characters = []
    IMG_SIZE = 100
    dataDir = "D:\\Projects\\AI\\dataset\\train"

    def __init__(self):

        self.read_data()

        # ### 2.2 Preprocess The Data

        le = LabelEncoder()
        self.Y = le.fit_transform(self.Z)
        self.Y = to_categorical(self.Y, len(self.characters))
        self.X = np.array(self.X)
        self.X = self.X / 255.0

        # ### 2.3 Splits it into Training Data and Test Data

        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=None)

        # ## 3. Model Section

        # ### 3.1 Build Model

        self.model = Sequential()

        self.model.add(Dense(100, input_shape=(900,), activation='relu'))  # First Layer in Model is our input layer
        self.model.add(Dense(60))
        self.model.add(Dense(60))
        self.model.add(Dense(80))

        self.model.add(Dropout(.52))
        self.model.add(Dense(80))
        self.model.add(Dense(100))

        self.model.add(Flatten())

        self.model.add(Dense(80))
        self.model.add(Dense(40))
        self.model.add(Dense(60))

        self.model.add(Activation('relu'))

        self.model.add(Dense(len(self.characters), activation='softmax'))

        # ### 3.2 Compile And  Fit Model

        # self.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
        # self.history = self.model.fit(x_train, y_train, epochs=600, validation_data=(x_test, y_test), verbose=1)
        # self.model.evaluate(x_test, y_test, batch_size=128)
        #
        # # ### 3.3 Save Model
        # self.model.save("Last_ANN.h5")
        # self.show_plots()

    # ###  plots
    def show_plots(self):
        plt.plot(self.history.history['loss'])  # loss curve
        plt.title('ANN Model Loss Curve')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['accuracy'])  # accuracy curve
        plt.title('ANN Model Accuracy Curve')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

    def read_data(self):
        self.characters.clear()
        for character in os.listdir(self.dataDir):
            self.characters.append(character)
            for img in tqdm(os.listdir(os.path.join(self.dataDir, character))):
                if fnmatch.fnmatch(img, '*jpg'):
                    label = character
                    path = os.path.join(self.dataDir, character, img)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    img = feature.hog(img, orientations=9, pixels_per_cell=(15, 15), cells_per_block=(2, 2),
                                      transform_sqrt=True, block_norm="L1")
                    self.X.append(np.array(img))
                    self.Z.append(str(label))

    # Function That View Image just send path
    def show_img(self, img):
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()

    # Function That prepare data to be recognized
    def prepare(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = feature.hog(img, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                          transform_sqrt=True, block_norm="L1")
        return img.reshape(2916, )

    # Function That Recognize character from the img u sent
    def recognize(self, img):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = feature.hog(img, orientations=9, pixels_per_cell=(15, 15), cells_per_block=(2, 2),
                          transform_sqrt=True, block_norm="L1")
        prediction = self.model.predict(np.array([img]) / 255)
        index = np.argmax(prediction)
        return 'ann',self.characters[index]

    def Validate(self):
        # model = models.load_model("Last.h5")
        validateDataDir = "D:\\Projects\\AI\\dataset\\test"
        TP = os.path.join(validateDataDir)
        for file in os.listdir(TP):
            P = os.path.join(TP, file)
            for img in os.listdir(P):
                try:
                    print(os.path.join(TP, file))
                    self.recognize(os.path.join(TP, file, img))
                except Exception as e:
                    print("Some Thing Went Wrong")
