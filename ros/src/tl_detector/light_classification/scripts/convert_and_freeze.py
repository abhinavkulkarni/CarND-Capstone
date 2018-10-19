import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import keras
from keras import backend as K


def keras_to_tf(path, model_name="model"):
    with tf.Session() as sess:
        K.set_learning_phase(0)
        model = keras.models.load_model(os.path.join(path, '{}.h5'.format(model_name)))
        saver = tf.train.Saver()
        saver.save(sess, save_path=os.path.join(path, model_name))
        tf.train.write_graph(sess.graph_def, path, '{}.pbtxt'.format(model_name), as_text=True)


def freeze_model(path, model_name="model", output_name="model_frozen.pb"):
    path = os.path.realpath(path)
    freeze_graph.freeze_graph(os.path.join(path, '{}.pbtxt'.format(model_name)), "", False,
                              os.path.join(path, model_name), "softmax/Softmax", "save/restore_all", "",
                              os.path.join(path, output_name), True, "")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="../models/classifier", help="absolute path to input directory")

    args = parser.parse_args()

    model_dir = os.path.realpath(args.model_dir)

    keras_to_tf(model_dir)
    freeze_model(model_dir)
