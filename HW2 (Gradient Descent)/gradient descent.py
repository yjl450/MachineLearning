import random
import time
# Normalization


def mean_std(l):
    mean = sum(l) / len(l)
    square = 0
    for i in l:
        square += (i - mean) ** 2
    vari = square / len(l)
    # std = math.sqrt(vari)
    std = vari ** 0.5
    return mean, std


source = open('housing.txt', 'r')
size = []
bedroom = []
price = []
for i in source.readlines():
    entry = i.split(',')
    for j in range(3):
        if j == 0 and entry[j].isdigit():
            size.append(int(entry[j]))
        elif j == 1 and entry[j].isdigit():
            bedroom.append(int(entry[j]))
        elif j == 2 and entry[j][:-2].isdigit():
            price.append(int(entry[j]))
source.close()
m1, std1 = mean_std(size)
# print(m1,std1)
m2, std2 = mean_std(bedroom)
# print(m2,std2)/
for i in range(len(size)):
    size[i] = (size[i] - m1) / std1
for i in range(len(bedroom)):
    bedroom[i] = (bedroom[i] - m2) / std2
target = open('normalized.txt', 'w')
for i in range(len(size)):
    target.write(str(size[i]) + ',' + str(bedroom[i]) +
                 ',' + str(price[i]) + '\n')
target.close()

# Gradient Descent


def derivative(data, w):
    l = [0, 0, 0]
    for i in data:
        y = i[-1]
        estimate = w[0] + w[1] * i[1] + w[2] * i[2]
        l[0] += ((estimate - y) * i[0])
        l[1] += ((estimate - y) * i[1])
        l[2] += ((estimate - y) * i[2])
    for i in range(3):
        l[i] = l[i] / len(data)
    return l


def error(data, w):
    m = len(data)
    total = 0
    for x in data:
        total += (w[0] + w[1] * x[1] + w[2] * x[2] - x[3])**2
    total = total / (2 * m)
    return total


def gradient_descent(data, w, a):
    for i in range(10):
        l = derivative(data, w)
        w[0] = w[0] - a * l[0]
        w[1] = w[1] - a * l[1]
        w[2] = w[2] - a * l[2]
        print(error(data, w))
    return w


def predict(w, size, bedroom, m1, std1, m2, std2):
    size = (size - m1) / std1
    bedroom = (bedroom - m2) / std2
    price = w[0] + w[1] * size + w[2] * bedroom
    return price


def test_GD():
    source = open('normalized.txt', 'r')
    data = []
    for i in source.readlines():
        stats = i[:-1].split(',')
        for i in range(len(stats)):
            stats[i] = float(stats[i])
        data.append([1] + stats)
    source.close()
    w = [0, 0, 0]
    a = 0.01
    # GD_start = time.time()
    print(gradient_descent(data, w, a))
    # GD_end = time.time()
    # print('Gradient Descent takes:', GD_end - GD_start)
    # print(predict(w, 2650, 4, m1, std1, m2, std2))


test_GD()


# SGD
def derivative_SGD(entry, w):
    l = [0, 0, 0]
    y = entry[-1]
    estimate = w[0] + w[1] * entry[1] + w[2] * entry[2]
    l[0] += ((estimate - y) * entry[0])
    l[1] += ((estimate - y) * entry[1])
    l[2] += ((estimate - y) * entry[2])
    return l


def stochastic_gradient_descent(data, w, a):
    for j in range(3):
        for i in data:
            l = derivative_SGD(i, w)
            w[0] = w[0] - a * l[0]
            w[1] = w[1] - a * l[1]
            w[2] = w[2] - a * l[2]
        # print(error(data, w))
        random.shuffle(data)
    return w


def predict_SGD(w, size, bedroom, m1, std1, m2, std2):
    size = (size - m1) / std1
    bedroom = (bedroom - m2) / std2
    price = w[0] + w[1] * size + w[2] * bedroom
    return price


def test_SGD():
    source = open('normalized.txt', 'r')
    data = []
    for i in source.readlines():
        stats = i[:-1].split(',')
        for i in range(len(stats)):
            stats[i] = float(stats[i])
        data.append([1] + stats)
    source.close()
    w = [0, 0, 0]
    a = 0.05
    SGD_start = time.time()
    print(stochastic_gradient_descent(data, w, a))
    SGD_end = time.time()
    print('SGD takes:', SGD_end - SGD_start)
    # print(predict_SGD(w, 2650, 4, m1, std1, m2, std2))


# test_SGD()
