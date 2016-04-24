import pickle as pickle
from nolearn.lasagne import BatchIterator

from datetime import datetime
from pandas import DataFrame
from pandas.io.parsers import read_csv
import numpy as np
from PIL import Image
from matplotlib import pyplot
from scipy.ndimage.filters import convolve
from math import ceil
import theano.tensor as T
import theano
from lasagne.layers import get_output
from scipy.ndimage import rotate
import sys
import os
import zipfile
if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
FLOOKUP = 'data/IdLookupTable.csv'


def float32(k):
    return np.cast['float32'](k)


class RotateBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(RotateBatchIterator, self).transform(Xb, yb)

        angle = np.random.randint(-10,11)
        Xb_rotated = rotate(Xb, angle, axes=(2, 3), reshape=False)

        return Xb_rotated, yb


class PreSplitTrainSplit(object):

    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def __call__(self, X, y, net):
        return self.X_train, self.X_valid, self.y_train, self.y_valid


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        if epoch >= nn.max_epochs:
            return
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def load_file(file):

    def url(file):
        if file is FTRAIN:
            return 'http://folk.ntnu.no/alfredvc/workshop/data/training.zip'
        if file is FTEST:
            return 'http://folk.ntnu.no/alfredvc/workshop/data/test.zip'
        if file is FLOOKUP:
            return 'http://folk.ntnu.no/alfredvc/workshop/data/test.zip'

    def zip(file):
        if file is FTRAIN:
            return 'data/training.zip'
        if file is FTEST:
            return 'data/test.zip'

    def download(file):
        print("Downloading %s" % file)
        urlretrieve(url(file), zip(file))
        print("Unzipping data %s" % file)
        if file is FTRAIN or file is FTEST:
            with zipfile.ZipFile(zip(file), "r") as z:
                z.extractall('data/')
        print("Deleting zip file " + zip(file))
        os.remove(zip(file))

    if not os.path.exists(file):
        download(file)

    return read_csv(file)


def load(file_path):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """

    df = load_file(file_path)

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if file_path is FTRAIN:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        y = y.astype(np.float32)
    else:
        y = None

    # print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    #         X.shape, X.min(), X.max()))
    # print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    #         y.shape, y.min(), y.max()))

    return X, y


def load2d(file_path):
    X, y = load(file_path)
    X = X.reshape(-1, 1, 96, 96)
    return X, y


def pickle_network(file_name, network):
    # in case the model is very big
    sys.setrecursionlimit(10000)
    with open(file_name, 'wb') as f:
        pickle.dump(network, f, -1)


def unpickle_network(file_name):
    with open(file_name, 'rb') as f:  # !
        return pickle.load(f)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
    ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def visualize_predictions(net):
    X, _ = load2d(FTEST)
    y_pred = net.predict(X)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    pyplot.show()


def load_and_plot_layer(layer):
    with open(layer, 'rb') as f:
        layer0 = np.load(f)
        fig = pyplot.figure()
        fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i in range(32):
            img = layer0[i, :, :]
            img -= np.min(img)
            img /= np.max(img) / 255.0
            ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
            ax.imshow(img, cmap='gray', interpolation='none')
        pyplot.show()

def create_submition(net):
    X = load2d(FTEST)[0]
    y_pred = net.predict(X)

    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)

    cols = ("left_eye_center_x","left_eye_center_y","right_eye_center_x","right_eye_center_y","left_eye_inner_corner_x","left_eye_inner_corner_y","left_eye_outer_corner_x","left_eye_outer_corner_y","right_eye_inner_corner_x","right_eye_inner_corner_y","right_eye_outer_corner_x","right_eye_outer_corner_y","left_eyebrow_inner_end_x","left_eyebrow_inner_end_y","left_eyebrow_outer_end_x","left_eyebrow_outer_end_y","right_eyebrow_inner_end_x","right_eyebrow_inner_end_y","right_eyebrow_outer_end_x","right_eyebrow_outer_end_y","nose_tip_x","nose_tip_y","mouth_left_corner_x","mouth_left_corner_y","mouth_right_corner_x","mouth_right_corner_y","mouth_center_top_lip_x","mouth_center_top_lip_y","mouth_center_bottom_lip_x","mouth_center_bottom_lip_y")

    df = DataFrame(y_pred2, columns=cols)

    lookup_table = load_file(FLOOKUP)
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
        ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))

def visualize_learning(net):
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    ymax = max(np.max(valid_loss), np.max(train_loss))
    ymin = min(np.min(valid_loss), np.min(train_loss))
    pyplot.ylim(ymin * 0.8, ymax * 1.2)
    pyplot.yscale("log")
    pyplot.show()

def conv(input, weights):
    return convolve(input, weights)


def show_kernels(kernels, cols=8):
    rows = ceil(len(kernels)*1.0/cols)
    fig = pyplot.figure(figsize=(cols+2, rows+1))

    fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(len(kernels)):
        img = np.copy(kernels[i])
        img -= np.min(img)
        img /= np.max(img)
        ax = fig.add_subplot(rows, cols, i + 1, xticks=[], yticks=[])
        ax.imshow(img, cmap='gray', interpolation='none')
    pyplot.axis('off')
    pyplot.show()


def get_activations(layer, x):
    # compile theano function
    xs = T.tensor4('xs').astype(theano.config.floatX)
    get_activity = theano.function([xs], get_output(layer, xs))

    return get_activity(x)


def show_images(list, cols=1):
    rows = ceil(len(list)*1.0/cols)
    fig = pyplot.figure(figsize=(cols+2, rows+1))
    fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(len(list)):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        ax.imshow(list[i], cmap='gray')
    pyplot.axis('off')
    pyplot.show()


def get_conv_weights(net):
    layers = net.get_all_layers()
    layercounter = 0
    w = []
    b = []
    for l in layers:
        if('Conv2DLayer' in str(type(l))):
            weights = l.W.get_value()
            biases = l.b.get_value()
            b.append(biases)
            weights = weights.reshape(weights.shape[0]*weights.shape[1],weights.shape[2],weights.shape[3])
            w.append(weights)
            layercounter += 1
    return w, b


def load_image(file):
    x=Image.open(file,'r')
    x=x.convert('L')
    y=np.asarray(x.getdata(),dtype=np.float32).reshape((x.size[1],x.size[0]))
    return y
