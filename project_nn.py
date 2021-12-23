import math
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

categories = {"elefante": "elephant", "farfalla": "butterfly", "gatto": "cat"}
animals = ["elephant", "butterfly", "cat"]
img_size = 100

n = img_size * img_size  # features
# m = 36  # observations
c = 3  # class

hidden_layer_count = 1
learning_rate = 0.005
batch_size = 12
neuron_numbers = [n, 4, 3]  # hl1, hl2

x_train = np.zeros((n, batch_size))  # 10000, 12
y_train = np.zeros((c, batch_size))  # 10, 12


def create_data(data, url_category_data, start_index, finish_index):
    for i in range(start_index, finish_index):
        path = url_category_data[i][0]
        target = url_category_data[i][1]
        try:
            img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            new_img_array = cv2.resize(img_array, (img_size, img_size))
            data.append([new_img_array, target])
        except Exception as e:
            pass


def flatten_and_normalize_data(data, start_index, end_index):
    for i in range(start_index, end_index):
        arr = np.asarray(data[i][0])
        data[i][0] = arr.reshape((img_size * img_size,)).astype(np.float32)
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] / 255)
        # print(data[i][0])
        # plt.imshow(data[i][0].reshape((img_size, img_size)), cmap='gray', vmin=0, vmax=1)
        # plt.show()


def sigmoid(x):
    a = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a[i][j] = 1 / (1 + math.pow(math.e, -x[i][j]))
    return a


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    expX = np.exp(x)
    return expX / np.sum(expX, axis=0)


def derivative_sigmoid(x):
    d = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            d[i][j] = (x[i][j])*(1-x[i][j])
    return d


def derivative_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)


def forward_propagation(x, parameters):
    w = []
    b = []
    for i in range(hidden_layer_count+1):
        w.append(parameters['w'][i])
        b.append(parameters['b'][i])

    a = []
    z = []
    for i in range(hidden_layer_count+1):
        if i == 0:  # first
            zi = np.dot(w[i], x) + b[i]
            z.append(zi)
        else:
            zi = np.dot(w[i], a[i-1]) + b[i]
            z.append(zi)

        if i == hidden_layer_count:  # last element
            ai = softmax(zi)
            a.append(ai)
        else:
            ai = sigmoid(zi)
            a.append(ai)

    return a


def cost_function(softmax_layer, y):
    mx = y.shape[1]
    loss = 0
    for j in range(softmax_layer.shape[1]):
        for i in range(softmax_layer.shape[0]):
            if y[i][j] == 1:
                probability = softmax_layer[i][j]
                loss += (-(math.log(probability)))
    cost = (1/mx) * loss
    return cost


def backward_prop(x, y, parameters, forward_cache):
    w1 = parameters['w'][0]
    b1 = parameters['b'][0]
    w2 = parameters['w'][1]
    b2 = parameters['b'][1]

    a1 = forward_cache[0]
    a2 = forward_cache[1]

    dz2 = (a2 - y)
    dw2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * derivative_sigmoid(a1)
    dw1 = np.dot(dz1, x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    parameters['w'][0] = parameters['w'][0] - learning_rate * dw1
    parameters['b'][0] = parameters['b'][0] - learning_rate * db1
    parameters['w'][1] = parameters['w'][1] - learning_rate * dw2
    parameters['b'][1] = parameters['b'][1] - learning_rate * db2

    return parameters


def init_parameters(parameters):
    np.random.seed(101)

    weight = []
    bias = []

    for i in range(hidden_layer_count):
        wei = np.random.randn(neuron_numbers[i + 1], neuron_numbers[i]) * 0.01
        bia = np.zeros((neuron_numbers[i + 1], 1))
        weight.append(wei)
        bias.append(bia)

    wei = np.random.randn(c, neuron_numbers[hidden_layer_count]) * 0.01
    bia = np.zeros((c, 1))
    weight.append(wei)
    bias.append(bia)

    parameters["w"] = weight
    parameters["b"] = bias
    return parameters


def initialization(data):
    for i in range(x_train.shape[0]):  # 10000
        for j in range(x_train.shape[1]):  # 36
            x_train[i][j] = data[j][0][i]

    for i in range(y_train.shape[1]):  # 36
        class_index = data[i][1]
        y_train[class_index][i] = 1


def all_image_urls_to_csv():
    folder_names = os.listdir('../Project/sample-img')
    category = []
    files = []
    i = 0
    for k, folder in enumerate(folder_names):
        filenames = os.listdir("../Project/sample-img/" + folder)
        for file in filenames:
            files.append("../Project/sample-img/" + folder + "/" + file)
            category.append(k)

    df = pd.DataFrame({
        'filename': files,
        'category': category
    })
    train_df = pd.DataFrame(columns=['filename', 'category'])
    for i in range(10):
        train_df = train_df.append(df[df.category == i].iloc[:500, :])

    df.to_csv('out.csv', index=False)
    return train_df


def main():
    train_df = all_image_urls_to_csv()
    url_category_data = np.array(train_df.iloc[:, :])
    np.random.seed(101)
    np.random.shuffle(url_category_data)
    print(url_category_data)

    parameters = {}
    parameters = init_parameters(parameters)

    cost_list = []

    for i in range(10):  # epoch
        data = []
        for j in range(0, 36, batch_size):  # batch
            create_data(data, url_category_data, j, j + batch_size)
            flatten_and_normalize_data(data, j, j + batch_size)
            initialization(data)

            activation_funcs = forward_propagation(x_train, parameters)
            cost = cost_function(activation_funcs[-1], y_train)  # a2 son layer olmalı
            parameters = backward_prop(x_train, y_train, parameters, activation_funcs)
            cost_list.append(cost)

            print("Cost after", j // 12 + 1, "batch is :", cost)

        if i % (1 / 1) == 0:
            print("Cost after", i, "iterations is :", cost)
            print()

    t = np.arange(0, 10*3)
    plt.plot(t, cost_list)
    plt.show()


if __name__ == "__main__":
    main()
