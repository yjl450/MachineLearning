def word(data, X, N = 4000): #data: training file X: frequency
    all_word = {}
    source = open(data, 'r')
    i = 1
    for l in source.readlines():
        if i > N:
            break
        for w in set(l.split()):
            if w.isalpha():
                if w in all_word.keys():
                    all_word[w] += 1
                else:
                    all_word[w] = 1
        i += 1
    vocabulary = []
    for i in all_word.keys():
        if all_word[i] >= X:
            vocabulary.append(i)
    source.close()
    return vocabulary


def feature_vector(email, vocabulary): #one email (One line)
    dim = len(vocabulary)
    vector = [0] * dim
    email_content = set(email[2:].split())
    for i in range(dim):
        if vocabulary[i] in email_content:
            vector[i] = 1
    return vector

def mul(a, b): #calculating the dot product
    product = 0
    for i in range(len(a)):
        product += a[i]*b[i]
    return product

def scale(a,b):
    c = [0] * len(b)
    for i in range(len(b)):
        b[i] = a*b[i]
    return b

def plus(a, b):
    summation = []
    for i in range(len(a)):
        summation.append(a[i] + b[i])
    return summation

def perceptron_train(data, vocabulary, N = 4000):
    dim = len(vocabulary)
    w = [0] * dim # classifier vector
    k = 0 # number of updates
    iter = 0 # number of passes
    source = open(data, 'r')
    update = True
    emails = []
    for l in source.readlines():
        emails.append(l)
    while update:
        update = False
        iter += 1
        for l in emails[: N]:
            y = int(l[0])
            if y == 0:
                y = -1
            x = feature_vector(l, vocabulary)
            product = mul(w, x)
            if y * product <= 0:
                x = scale(y, x)
                w = plus(w, x)
                k += 1
                update = True
    source.close()
    return w, k, iter

def perceptron_train_limited(data, vocabulary, limit):
    dim = len(vocabulary)
    w = [0] * dim # classifier vector
    k = 0 # number of updates
    iter = 0 # number of passes
    source = open(data, 'r')
    update = True
    emails = []
    for l in source.readlines():
        emails.append(l)
    while update:
        update = False
        if iter >= limit:
            break
        iter += 1
        for l in emails:
            y = int(l[0])
            if y == 0:
                y = -1
            x = feature_vector(l, vocabulary)
            product = mul(w, x)
            if y * product <= 0:
                x = scale(y, x)
                w = plus(w, x)
                k += 1
                update = True
    source.close()
    return w, k, iter

def perceptron_error(w, data, vocabulary):
    source = open(data, 'r')
    count = 0
    error = 0
    for i in source.readlines():
        count += 1
        spam = int(i[0])
        if spam == 0:
            spam = -1
        feature = feature_vector(i, vocabulary)
        detect = mul(w, feature)
        if detect * spam < 0:
            error += 1
    return error/count

if __name__ == "__main__":
    # data = 'train.txt'
    # vocabulary = word(data, 26, 4000)
    # w, k ,iter = perceptron_train(data,vocabulary, 4000)
    # print('Updates:', k)
    # print('Iterations:', iter)
    # print(perceptron_error(w, data, vocabulary)*100,'%')
    # print(perceptron_error(w, 'validate.txt', vocabulary)*100,'%')

    # plot data
    # data = 'train.txt'
    # for N in [200,600,1200,2400,4000]:
    #     vocab = word(data, 26, N)
    #     w, k, iter = perceptron_train(data, vocab, N)
    #     print('When N =', N, end=': ')
    #     print(iter, 'iterations',end=', ')
    #     print('error rate: ', perceptron_error(w, 'validate.txt', vocab)*100,'%')

    # test
    data = 'spam_train.txt'
    X = 22
    limit = 8
    vocabulary = word(data, X)
    w, k, iter = perceptron_train_limited(data, vocabulary, limit)
    print('---->', limit, iter, X)
    print('Error for the training data:', perceptron_error(w, data, vocabulary) * 100, '%')
    print('Error for validation data:', perceptron_error(w, 'spam_test.txt', vocabulary)*100, '%\n')

    #>1500 emails
    # data = 'train.txt'
    # vocabulary = word(data, 1500, 4000)
    # print(len(vocabulary))
    # print(vocabulary)
    # w, k ,iter = perceptron_train(data,vocabulary, 4000)
    # print('Updates:', k)
    # print('Iterations:', iter)
    # print(perceptron_error(w, data, vocabulary)*100,'%')
    # print(perceptron_error(w, 'validate.txt', vocabulary)*100,'%')