import pickle
import os.path
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from sample_solution import sample_cnn

from matplotlib import pyplot as plt
if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


def pkl(file_name, object):
    with open(file_name, 'wb') as f:
        pickle.dump(object, f, -1)


def un_pkl_l(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def un_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def make_image(X):
    im = np.swapaxes(X.T, 0, 1)
    im = im - im.min()
    im = im * 1.0 / im.max()
    return im


def show_images(data, predicted, labels, classes):
    plt.figure(figsize=(16, 5))
    for i in range(0, 10):
        plt.subplot(1, 10, i+1)
        plt.imshow(make_image(data[i]), interpolation='nearest')
        true = classes[labels[i]]
        pred = classes[predicted[i]]
        color = 'green' if true == pred else 'red'
        plt.text(0, 0, true, color='black', bbox=dict(facecolor='white', alpha=1))
        plt.text(0, 32, pred, color=color, bbox=dict(facecolor='white', alpha=1))

        plt.axis('off')

DATA = 'data.pkl'


def load_file(file):
    def url(file):
        if file is DATA:
            return 'http://folk.ntnu.no/alfredvc/workshop/data/data.pkl'

    def download(file):
        print("Downloading %s" % file)
        urlretrieve(url(file), file)

    if not os.path.exists(file):
        download(file)
    return un_pkl_l(file)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='same')

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            network,
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def main(model='cnn', num_epochs=10):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_test, y_test, classes = load_file(DATA)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'cnn':
        network = build_cnn(input_var)
    elif model == 'suggested_cnn':
        network = sample_cnn.build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Compile a third function computing a prediction
    eval_fn = theano.function([input_var], [T.argmax(test_prediction, axis=1)])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    training_error = []
    test_error = []
    test_accuracy = []
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 64, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        training_error.append(train_err / train_batches)
        test_error.append(val_err / val_batches)
        test_accuracy.append(val_acc / val_batches)

    data = X_test[123:133]
    labels = y_test[123:133]
    predicted = eval_fn(data)[0]
    show_images(data, predicted, labels, classes)
    fig, ax1 = plt.subplots()
    ax1.plot(training_error, color='b', label='Training error')
    ax1.plot(test_error, color='g', label='Test error')
    ax2 = ax1.twinx()
    ax2.plot(test_accuracy, color='r', label='Test accuracy')
    ax1.legend(loc='upper left', numpoints=1)
    ax2.legend(loc='upper right', numpoints=1)
    plt.xlabel("Epoch")

    plt.show()



    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
main(num_epochs=15)
