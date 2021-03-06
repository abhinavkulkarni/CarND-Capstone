#!/usr/bin/env python

import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator


import os
import numpy as np
from PIL import Image

CLASSES = ['Red', 'Yellow', 'Green', 'NoTrafficLight']


class TLClassifier(object):
    def __init__(self, model_dir=''):
        self.graph = tf.Graph()

        # configuration for possible GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            graph_def = tf.GraphDef()


            with tf.gfile.GFile(os.path.join(model_dir, 'model_frozen.pb'), 'rb') as f:
                serialized_graph = f.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('conv1_input:0')
            self.output_tensor = self.graph.get_tensor_by_name('softmax/Softmax:0')

        print('*' * 50)
        print('Loaded CNN model')
        print('*' * 50)

    def get_classification(self, image, height=32, width=32):
        """Determines the color of the traffic light in the image

        Args:
            image: image containing the traffic light
            height: target height
            width: target width
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = image.resize((width, height), Image.ANTIALIAS)
        image = np.array(image)
        img_reshape = np.expand_dims(image, axis=0).astype('float32')

        # Normalization
        img_norm = img_reshape / 255.

        # Prediction
        # predict = self.model.predict(img_norm)
        predict = self.sess.run(self.output_tensor, feed_dict={self.image_tensor: img_norm})
        return CLASSES[np.argmax(predict)]

    @staticmethod
    def build_model(height=32, width=32, channels=3, n_classes=4):
        """Builds a CNN model to determines the color of the traffic light in the image

        Args:
            height: height of the image
            width: width of the image
            channels: number of channels in the image
            n_classes: number of classes in the output

        Returns:
            model: keras model

        """

        # initialize the model
        model = Sequential()
        input_shape = (height, width, channels)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', name='conv1'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # third set of CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes))
        model.add(Activation('softmax', name='softmax'))

        # return the constructed network architecture
        print("Model Summary")
        print(model.summary())
        return model

    def train_model(self, data, model_path, plot='plot.png',
                    lr=1e-3, height=32, width=32, batch_size=32, epochs=50):
        from data_preparation import prepare_data
        import matplotlib.pyplot as plt

        model = self.build_model()
        opt = Adam(lr=lr)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=[metrics.mae, metrics.categorical_accuracy])

        # Split training/validation 80/20
        # https://faroit.github.io/keras-docs/2.0.8/preprocessing/image/
        prepare_data(data_location=data,
                     train_data_path=data+'/train/',
                     test_data_path=data+'/test/')

        datagen = ImageDataGenerator(#validation_split=0.2,
                                     rescale=1. / 255,
                                     rotation_range=20.,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True
                                     )
        train_gen = datagen.flow_from_directory(
            data+'/train/',
            classes=CLASSES,
            target_size=(height, width),
            batch_size=batch_size
        )

        val_gen = datagen.flow_from_directory(
            data+'/test/',
            classes=CLASSES,
            target_size=(height, width)
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        info = model.fit_generator(train_gen,
                                   steps_per_epoch=3,
                                   validation_steps=2,
                                   validation_data=val_gen,
                                   epochs=epochs,
                                   callbacks=[early_stopping]
                                   )

        model.save(os.path.join(model_path, 'model.h5'))

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        num = len(info.history["loss"])
        plt.plot(np.arange(0, num), info.history["loss"], label="train_loss")
        plt.plot(np.arange(0, num), info.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, num), info.history["categorical_accuracy"], label="train_acc")
        plt.plot(np.arange(0, num), info.history["val_categorical_accuracy"], label="val_acc")
        plt.plot(np.arange(0, num), info.history["mean_absolute_error"], label="train_mae")
        plt.plot(np.arange(0, num), info.history["val_mean_absolute_error"], label="val_mae")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.savefig(plot)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument("-d", "--dataset", default='../data/tl_classifier_exceptsmall/simulator',
                        help="path to dataset")
    parser.add_argument("-m", "--model-dir", default='../models/classifier',
                        help="path to output model")
    parser.add_argument("-p", "--plot", default="plot.png",
                        help="path to output loss/accuracy plot")
    parser.add_argument("-i", "--image", default="../data/tl_classifier_exceptsmall/simulator/Green/001176.png",
                        help="path to output loss/accuracy plot")

    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument("--batch", type=int, default=32,
                        help="batch size")

    args = parser.parse_args()
    
    if args.command == 'train':
        tl_classifier = TLClassifier()
        tl_classifier.train_model(args.dataset, args.output_dir,
                                  lr=args.lr, batch_size=args.batch,
                                  plot=args.plot)

    elif args.command == 'test':
        tl_classifier = TLClassifier(args.model_dir)
        img = Image.open(args.image)
        print(tl_classifier.get_classification(img))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
