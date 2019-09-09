import numpy

def retrieve_data():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = numpy.expand_dims(x_train, -1)
    x_test = numpy.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)
