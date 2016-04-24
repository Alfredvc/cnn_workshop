import theano
import facial_recognition.util as util
from lasagne import layers
from nolearn.lasagne import NeuralNet
from facial_recognition.util import AdjustVariable
from facial_recognition.util import EarlyStopping
from facial_recognition.util import FlipBatchIterator
from facial_recognition.util import float32

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


def get_net():
    return NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('conv1', Conv2DLayer),
                ('pool1', MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('conv2', Conv2DLayer),
                ('pool2', MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),
                ('conv3', Conv2DLayer),
                ('pool3', MaxPool2DLayer),
                ('dropout3', layers.DropoutLayer),
                ('hidden4', layers.DenseLayer),
                ('dropout4', layers.DropoutLayer),
                ('hidden5', layers.DenseLayer),
                ('output', layers.DenseLayer),
            ],
            input_shape=(None, 1, 96, 96),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
            dropout1_p=0.1,
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
            dropout2_p=0.2,
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
            dropout3_p=0.3,
            hidden4_num_units=1000,
            dropout4_p=0.5,
            hidden5_num_units=1000,
            output_num_units=30, output_nonlinearity=None,

            update_learning_rate=theano.shared(float32(0.03)),
            update_momentum=theano.shared(float32(0.9)),

            regression=True,
            batch_iterator_train=FlipBatchIterator(batch_size=128),
            on_epoch_finished=[
                AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                AdjustVariable('update_momentum', start=0.9, stop=0.999),
                EarlyStopping(patience=200),
            ],
            max_epochs=3000,
            verbose=1,
    )


def train_network(net, save_name=''):
    print("Loading data...")
    X, y = util.load2d(util.FTRAIN)
    print("Building network...")
    print("Started training...")
    net.fit(X, y)
    print("Finished training...")
    print("Saving network...")
    util.pickle_network(save_name + ".pkl", net)
    util.visualize_learning(net)


def load_and_visualize_network(file):
    print("Loading data...")
    X, y = util.load2d(util.FTEST)
    print("Loading model...")
    net = util.unpickle_network(file)
    print("Finished training...")
    # util.visualize_learning(net)
    util.visualize_predictions(net)

net = get_net()

train_network(net, "net")