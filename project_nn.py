import math
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

img_size = 50
n = img_size * img_size  # features
c = 10  # class

hidden_layer_count = 0
learning_rate = 0.01
epoch = 30
batch_size = 16
# activation_func = "sigmoid"
# activation_func = "tan"
activation_func = "relu"

total_train_image_count = 20480
neuron_numbers = [n]  # as a start only input neurons

total_validation_image_count = 2816
total_test_image_count = 2816


# this function takes all the images urls and their class (which animal)
# creates a new csv file which has urls
def all_image_urls_to_csv():
    folder_names = os.listdir('../Project/raw-img')
    category = []
    files = []
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
    # parameters
    parameters = {}
    np.random.seed(101)

    weight = []
    bias = []

    # initializing neuron numbers for each hidden layers
    for i in range(hidden_layer_count):
        x = round(math.sqrt(neuron_numbers[i] + c))
        neuron_numbers.append(x)

    for i in range(hidden_layer_count):
        wei = np.random.randn(neuron_numbers[i + 1], neuron_numbers[i]) * 0.0001
        bia = np.zeros((neuron_numbers[i + 1], 1))
        weight.append(wei)
        bias.append(bia)

    wei = np.random.randn(c, neuron_numbers[hidden_layer_count]) * 0.0001
    bia = np.zeros((c, 1))
    weight.append(wei)
    bias.append(bia)

    parameters["w"] = weight
    parameters["b"] = bias
    return parameters


def create_data(data, url_category_data, start_index, finish_index):
    data = []
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
        data[i][0] = (data[i][0] / 255)
    return data


def init_train(data, x_train, y_train):
    x_train = np.zeros((n, batch_size))
    y_train = np.zeros((c, batch_size))

    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            x_train[i][j] = data[j][0][i]

    for i in range(y_train.shape[1]):
        class_index = data[i][1]
        y_train[class_index][i] = 1

    return x_train, y_train


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    expX = np.exp(x)
    return expX / np.sum(expX, axis=0)


def derivative_sigmoid(x):
    return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))


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
            if activation_func == "sigmoid":
                ai = sigmoid(zi)
            elif activation_func == "tan":
                ai = tanh(zi)
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

    m = batch_size

    dz_output = (a[0] - y)
    dw_output = (1 / m) * np.dot(dz_output, a[1].T)
    db_output = (1 / m) * np.sum(dz_output, axis=1, keepdims=True)

    dz.append(dz_output)
    dw.append(dw_output)
    db.append(db_output)

    for i in range(hidden_layer_count):
        if activation_func == "sigmoid":
            dz_i = np.dot(w[i].T, dz[i]) * derivative_sigmoid(a[i + 1])
        elif activation_func == "tan":
            dz_i = np.dot(w[i].T, dz[i]) * derivative_tanh(a[i + 1])
        else:
            dz_i = np.dot(w[i].T, dz[i]) * derivative_relu(a[i + 1])
        dw_i = (1 / m) * np.dot(dz_i, a[i + 2].T)
        db_i = (1 / m) * np.sum(dz_i, axis=1, keepdims=True)
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


def performance(softmax_layer, y):
    true_count = 0
    for j in range(softmax_layer.shape[1]):
        max_prob = -1
        index = -1
        for i in range(softmax_layer.shape[0]):
            if max_prob < softmax_layer[i][j]:
                max_prob = softmax_layer[i][j]
                index = i
        if y[index][j] == 1:
            true_count += 1
    return true_count / softmax_layer.shape[1]


def visualize_weights(parameters):
    for i in range(c):
        weight_best = parameters["w"][0][i].reshape((img_size, img_size))
        plt.imshow(weight_best, cmap='gray', vmin=np.amin(weight_best), vmax=np.amax(weight_best))
        plt.savefig(str(i) + ".png")
        plt.show()


def train_amd_validation(x_train, y_train, x_validation, y_validation, url_category_data, learning_rate):
    parameters = init_parameters()
    cost_list = []
    validation_cost_list = []
    start_learning_rate = learning_rate

    best_parameters_from_validation = parameters
    best_performance_from_validation = 0
    train_performance_list = []
    validation_performance_list = []

    for i in range(epoch):  # epoch
        data = []
        performance_val_train = 0
        for j in range(0, total_train_image_count, batch_size):  # batch
            data = create_data(data, url_category_data, j, j + batch_size)
            data = flatten_and_normalize_data(data, 0, batch_size)
            x_train, y_train = init_train(data, x_train, y_train)

            activation_funcs = forward_propagation(x_train, parameters)
            cost = cost_function(activation_funcs[-1], y_train)
            perf_t = performance(activation_funcs[-1], y_train)
            performance_val_train += perf_t
            parameters = backward_prop(x_train, y_train, parameters, activation_funcs)

        cost_list.append(cost)
        performance_val_train = performance_val_train / (total_train_image_count / batch_size)
        train_performance_list.append(performance_val_train)
        print("Cost after", i, "epoch is :", cost)
        print("------------------------------------------------")

        performance_val = 0
        for k in range(total_train_image_count, total_train_image_count + total_validation_image_count, batch_size):
            validation_data = []
            validation_data = create_data(validation_data, url_category_data, k, k + batch_size)
            validation_data = flatten_and_normalize_data(validation_data, 0, batch_size)
            x_validation, y_validation = init_train(validation_data, x_validation, y_validation)
            activation_funcs_v = forward_propagation(x_validation, parameters)
            cost_v = cost_function(activation_funcs_v[-1], y_validation)
            perf = performance(activation_funcs_v[-1], y_validation)
            performance_val += perf
        validation_cost_list.append(cost_v)
        print("Validation Cost :", cost_v)
        performance_val = performance_val / (total_validation_image_count / batch_size)
        validation_performance_list.append(performance_val)
        print("PERFORMANCE : ", performance_val)
        if performance_val >= best_performance_from_validation:
            best_performance_from_validation = performance_val
            print("BEST PERFORMANCE FOR NOW : ", best_performance_from_validation)
            best_parameters_from_validation = parameters
        print("--------------------------------------------------------------------")

        learning_rate *= 0.95

    plt.title(
        "TRAIN DATA \n" + "Hidden Layer Count:" + str(hidden_layer_count) + "-- Learning Rate:" + str(
            start_learning_rate) + "-- Batch:" + str(batch_size))
    plt.plot(np.arange(0, len(cost_list)), cost_list, label="train")
    plt.plot(np.arange(0, len(validation_cost_list)), validation_cost_list, label="validation")
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("cost.png")
    plt.show()

    plt.title("VALIDATION DATA \n" + "Hidden Layer Count:" + str(hidden_layer_count) + "-- Learning Rate:" + str(
        start_learning_rate) + "-- Batch:" + str(batch_size))
    plt.plot(np.arange(0, len(train_performance_list)), train_performance_list, label="train")
    plt.plot(np.arange(0, len(validation_performance_list)), validation_performance_list, label="validation")
    plt.ylabel("Performance")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("performance.png")
    plt.show()
    return best_parameters_from_validation


def main(learning_rate=learning_rate):
    train_df = all_image_urls_to_csv()
    url_category_data = np.array(train_df.iloc[:, :])
    np.random.seed(101)
    np.random.shuffle(url_category_data)

    x_train = np.zeros((n, batch_size))
    y_train = np.zeros((c, batch_size))

    x_validation = np.zeros((n, batch_size))
    y_validation = np.zeros((c, batch_size))

    x_test = np.zeros((n, batch_size))
    y_test = np.zeros((c, batch_size))

    parameters = train_amd_validation(x_train, y_train, x_validation, y_validation, url_category_data, learning_rate)
    np.save("parameters.npy", parameters)

    test_data = []
    performance_test = 0
    start = total_train_image_count + total_validation_image_count
    for j in range(start, start + total_test_image_count, batch_size):
        test_data = create_data(test_data, url_category_data, j, j + batch_size)
        test_data = flatten_and_normalize_data(test_data, 0, batch_size)
        x_test, y_test = init_train(test_data, x_test, y_test)

        activation_funcs_t = forward_propagation(x_test, parameters)
        perf = performance(activation_funcs_t[-1], y_test)
        performance_test += perf
    performance_test = performance_test / (total_test_image_count / batch_size)
    print("TEST PERFORMANCE", performance_test)

    if hidden_layer_count == 0:
        visualize_weights(parameters)


if __name__ == "__main__":
    main()
