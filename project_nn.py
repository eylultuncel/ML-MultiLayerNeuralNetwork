import math
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

img_size = 100
n = img_size * img_size  # features
c = 10  # class

hidden_layer_count = 1
learning_rate = 0.002
epoch = 3
batch_size = 16
total_train_image_count = 800
neuron_numbers = [n, 10, 10, 3]  # hl1, hl2

total_validation_image_count = 80


def all_image_urls_to_csv():
    folder_names = os.listdir('../Project/raw-img')
    category = []
    files = []
    i = 0
    for k, folder in enumerate(folder_names):
        filenames = os.listdir("../Project/raw-img/" + folder)
        for file in filenames:
            files.append("../Project/raw-img/" + folder + "/" + file)
            category.append(k)

    df = pd.DataFrame({
        'filename': files,
        'category': category
    })
    train_df = pd.DataFrame(columns=['filename', 'category'])
    for i in range(10):
        train_df = train_df.append(df[df.category == i].iloc[:, :])

    df.to_csv('out.csv', index=False)
    return train_df


def init_parameters():
    parameters = {}
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
    return data


def flatten_and_normalize_data(data, start_index, end_index):
    for i in range(start_index, end_index):
        arr = np.asarray(data[i][0])
        data[i][0] = arr.reshape((img_size * img_size,)).astype(np.float32)
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] / 255)
        # plt.imshow(data[i][0].reshape((img_size, img_size)), cmap='gray', vmin=0, vmax=1)
        # plt.show()
    return data


def init_train(data, x_train, y_train):
    for i in range(x_train.shape[0]):  # 10000
        for j in range(x_train.shape[1]):  # 36
            x_train[i][j] = data[j][0][i]

    for i in range(y_train.shape[1]):  # 36
        class_index = data[i][1]
        y_train[class_index][i] = 1

    return x_train, y_train


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
            d[i][j] = (x[i][j]) * (1 - x[i][j])
    return d


def derivative_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)


def forward_propagation(x, parameters):
    w = []
    b = []
    for i in range(hidden_layer_count + 1):
        w.append(parameters['w'][i])
        b.append(parameters['b'][i])

    a = []
    z = []
    for i in range(hidden_layer_count + 1):
        if i == 0:  # first
            zi = np.dot(w[i], x) + b[i]
            z.append(zi)
        else:
            zi = np.dot(w[i], a[i - 1]) + b[i]
            z.append(zi)

        if i == hidden_layer_count:  # last element
            ai = softmax(zi)
            a.append(ai)
        else:
            ai = relu(zi)
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
    cost = (1 / mx) * loss
    return cost


def backward_prop(x, y, parameters, forward_cache):
    w = parameters["w"].copy()
    w.reverse()
    a = forward_cache
    a.reverse()
    a.append(x)

    dz = []
    dw = []
    db = []

    dz_output = (a[0] - y)
    dw_output = np.dot(dz_output, a[1].T)
    db_output = np.sum(dz_output, axis=1, keepdims=True)

    dz.append(dz_output)
    dw.append(dw_output)
    db.append(db_output)

    for i in range(hidden_layer_count):
        dz_i = np.dot(w[i].T, dz[i]) * derivative_relu(a[i + 1])
        dw_i = np.dot(dz_i, a[i + 2].T)
        db_i = np.sum(dz_i, axis=1, keepdims=True)
        dz.append(dz_i)
        dw.append(dw_i)
        db.append(db_i)

    dz.reverse()
    dw.reverse()
    db.reverse()

    for i in range(hidden_layer_count + 1):
        parameters['w'][i] = parameters['w'][i] - (learning_rate * dw[i])
        parameters['b'][i] = parameters['b'][i] - (learning_rate * db[i])

    return parameters


def all_image_urls_to_csv():
    folder_names = os.listdir('../Project/raw-img')
    category = []
    files = []
    i = 0
    for k, folder in enumerate(folder_names):
        filenames = os.listdir("../Project/raw-img/" + folder)
        for file in filenames:
            files.append("../Project/raw-img/" + folder + "/" + file)
            category.append(k)

    df = pd.DataFrame({
        'filename': files,
        'category': category
    })
    train_df = pd.DataFrame(columns=['filename', 'category'])
    for i in range(10):
        train_df = train_df.append(df[df.category == i].iloc[:, :])

    df.to_csv('out.csv', index=False)
    return train_df


def performance(softmax_layer, y):
    mx = y.shape[1]
    true_count = 0

    # print(softmax_layer.shape)
    # print(softmax_layer)
    # print(y)
    for j in range(softmax_layer.shape[1]):
        max_prob = -1
        for i in range(softmax_layer.shape[0]):
            if max_prob < softmax_layer[i][j]:
                max_prob = softmax_layer[i][j]

        # print(max_prob)

        for i in range(softmax_layer.shape[0]):
            if y[i][j] == 1 and softmax_layer[i][j] == max_prob:
                true_count += 1

    print("true prediction count = ", true_count)
    print(true_count / batch_size)


def main(learning_rate=learning_rate):
    train_df = all_image_urls_to_csv()
    url_category_data = np.array(train_df.iloc[:, :])
    np.random.seed(101)
    np.random.shuffle(url_category_data)

    x_train = np.zeros((n, batch_size))  # 10000, 12
    y_train = np.zeros((c, batch_size))  # 10, 12

    x_test = np.zeros((n, batch_size))  # 10000, 12
    y_test = np.zeros((c, batch_size))  # 10, 12

    parameters = init_parameters()
    cost_list = []

    for i in range(epoch):  # epoch
        data = []
        for j in range(0, total_train_image_count, batch_size):  # batch

            data = create_data(data, url_category_data, j, j + batch_size)
            data = flatten_and_normalize_data(data, j, j + batch_size)
            x_train, y_train = init_train(data, x_train, y_train)

            activation_funcs = forward_propagation(x_train, parameters)
            cost = cost_function(activation_funcs[-1], y_train)  # a2 son layer olmalı
            performance(activation_funcs[-1], y_train)

            parameters = backward_prop(x_train, y_train, parameters, activation_funcs)
            cost_list.append(cost)

            print("Cost after", j // batch_size + 1, "batch is :", cost)

        print("Cost after", i, "epoch is :", cost)
        print()
        learning_rate *= 0.5

        print("------------------------------------------------")
        validation_data = []
        for j in range(total_train_image_count, total_train_image_count + total_validation_image_count, batch_size):
            validation_data = create_data(validation_data, url_category_data, j, j + batch_size)
            validation_data = flatten_and_normalize_data(validation_data, j - total_train_image_count,
                                                         j - total_train_image_count + batch_size)
            x_test, y_test = init_train(data, x_test, y_test)

            activation_funcs_v = forward_propagation(x_test, parameters)
            cost = cost_function(activation_funcs_v[-1], y_test)  # a2 son layer olmalı
            performance(activation_funcs_v[-1], y_test)

    t = np.arange(0, len(cost_list))
    plt.plot(t, cost_list)
    plt.show()


if __name__ == "__main__":
    main()
