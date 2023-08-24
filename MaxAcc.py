import ipykernel 	# 進度條格式更改
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from readDataset import *
from attention import *
from loss import CE
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import seaborn as sn

def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift)
    datagen.fit(x_train)
    return datagen


def define_model(input_shape=(100, 100, 1), classes=7):

	inputTensor = Input(input_shape)
	x = Conv2D(64, (3, 3), padding='same')(inputTensor)
	x = BatchNormalization()(x)
	a = x = Activation(gelu)(x)

	x = Conv2D(64, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	y1 = x = Activation(gelu)(x)
	x = Conv2D(64, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y1)
	y1 = x = concatenate([x, y1])

	y2 = x = Conv2D(128, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	# x = Activation("relu")(x)
	x = Activation(gelu)(x)
	x = Conv2D(128, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	y2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y2)
	x = concatenate([x, y2])

	y3 = x = Conv2D(256, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)
	x = Conv2D(256, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	y3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y3)
	x = concatenate([x, y3])

	y4 = x = Conv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)
	# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	# y4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y4)
	x = concatenate([x, y4])

	x = Conv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(gelu)(x)

	# 分支
	x2 = Conv2D(128, (3, 3), padding='same')(y1)
	x2 = BatchNormalization()(x2)
	x2 = Activation(gelu)(x2)
	x21 = x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
	x2 = Conv2D(128, (3, 3), padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(gelu)(x2)
	x2 = Add()([x2, x21])
	# x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
	x2 = Conv2D(512, (3, 3), padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(gelu)(x2)
	x22 = x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
	x2 = Add()([x2, x22])

	# 切割 feature map 成4塊
	a1, a2 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(a)
	a3, a4 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(a1)
	a5, a6 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(a2)
	#
	# # 左上
	# a31 = a3 = local_attention(a3)
	#
	# # 右上
	# a41 = a4 = local_attention(a4)
	#
	# # 右上
	# a51 = a5 = local_attention(a5)
	#
	# # 右上
	# a61 = a6 = local_attention(a6)

	# 切割部分結合
	a1 = a = concatenate([a3, a4, a5, a6])
	a1 = gACNN(a1)
	a = GlobalAveragePooling2D()(a)
	a = concatenate([a, a1])
	a = Dense(128, activation='relu')(a)
	a = Dropout(0.3)(a)

	x = Multiply()([x, x2])
	x1 = gACNN(x)
	x = GlobalAveragePooling2D()(x)
	x = concatenate([x, x1])
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.3)(x)
	x = concatenate([x, a])
	x = Dense(128, activation='relu')(x)
	x = Dense(7, activation='softmax')(x)

	return Model(inputs=inputTensor, outputs=x)


def run_model():
	fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

	x_train, x_test, y_train, y_test, x_val, y_val = readFERplus()
	# x_train, x_test, y_train, y_test= readRAFDB()
	datagen = data_augmentation(x_train)

	epochs = 300
	batch_size = 64

	# Training model from scratch

	black = define_model(input_shape=x_train[0].shape, classes=len(fer_classes))
	black.summary()
	black.compile(optimizer=Adam(learning_rate=0.0001), loss=CE, metrics=['accuracy'])
	history = black.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
						steps_per_epoch=len(x_train) // batch_size,
						validation_data=(x_val, y_val), verbose=2)
	test_loss, test_acc = black.evaluate(x_test, y_test, batch_size=batch_size)

	plot_acc_loss(history)
	save_model_and_weights(black, test_acc)
	# ==========confusion_matrix===========20230523=======start


	save_path = r"C:\Users\Black\Desktop\research\program\BlackBlack\Saved-Models/confusion_matrix_basic.png"

	prediction = black.predict(x_test)
	answer = np.argmax(prediction, axis=1)
	y_answer = np.argmax(y_test, axis=1)
	# len(y_answer),len(answer)
	pd.crosstab(y_answer, answer, rownames=['label'], colnames=['predict'])

	confusion_matrix(y_answer, answer)
	sum(confusion_matrix(y_answer, answer))

	rate = confusion_matrix(y_answer, answer) / sum(confusion_matrix(y_answer, answer))
	# print(np.round(new, 2))

	plt.figure(figsize=(12, 8))
	fx = sn.heatmap(np.round(rate, 2), annot=True, fmt='.2f', cmap='Blues', vmin=0.1, vmax=0.8)
	fx.xaxis.set_ticklabels(["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])
	fx.yaxis.set_ticklabels(["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])
	plt.xlabel('Predict Label')
	plt.ylabel('True Label')
	plt.title('Conusion matrix on RAFDB')
	plt.savefig(save_path)


# ==========confusion_matrix===========20230523=======end

def show_augmented_images(datagen, x_train, y_train):
    it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(it.next()[0][0], cmap='gray')
        # plt.xlabel(class_names[y_train[i]])
    plt.show()


def plot_acc_loss(history):
    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.show()

    # Plot loss graph
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 3.5])
    plt.legend(loc='upper right')
    plt.show()


def save_model_and_weights(model, test_acc):
    # Serialize and save model to JSON
    test_acc = int(test_acc * 10000)
    model_json = model.to_json()
    with open('Saved-Models\\model' + str(test_acc) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # Serialize and save weights to JSON
    model.save_weights('Saved-Models\\model' + str(test_acc) + '.h5')
    print('Model and weights are saved in separate files.')

#
# def load_model_and_weights(model_path, weights_path):
#     # Loading JSON model
#     json_file = open(model_path, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#
#     # Loading weights
#     model.load_weights(weights_path)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     print('Model and weights are loaded and compiled.')

def gelu(x):
	cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
	return x * cdf

if __name__ == '__main__':
	run_model()