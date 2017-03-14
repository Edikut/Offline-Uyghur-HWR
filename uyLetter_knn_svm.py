from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np

class Bunch(dict):
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes.

    # >>> b = Bunch(a=1, b=2)
    # >>> b['b']
    # 2
    # >>> b.b
    # 2
    # >>> b.a = 3
    # >>> b['a']
    # 3
    # >>> b.c = 6
    # >>> b['c']
    # 6

    """

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass

def open_file(m_file, wr="r"):
    try:
        file = open(m_file, wr)
        return file
    except:
        print("error! opennnig file: %s\n" % m_file)
        return False


def close_file(m_file):
    try:
        file = m_file.close()
        return file
    except:
        print("error! closing file: %s\n" % m_file)
        return False


def load_svm_format_ftr(svm_file):
    data = [];    labels = []
    i_file = open_file(svm_file)
    line = i_file.readline()
    while line:
            tokens = line.strip().split(' ')
            labels.append(int(tokens[0]))
            xx = []
            for tk in tokens[1:]:
                (l, d) = tk.split(':')
                # print(d)
                xx.append(float(d))
            data.append(xx)
            line = i_file.readline()
    data = np.array(data)
    target = np.array(labels)
    # return data, target
    return Bunch(data=data, target=target)


def load_csv_format_ftr(csv_file):

    data = np.loadtxt(csv_file, delimiter=',')
    target = data[:, -1].astype(np.int)
    flat_data = data[:, :-1]

    # if return_X_y:
    # return flat_data, target

    return Bunch(data=flat_data, target=target)


def load_chars(csv_file, n_class=128, return_X_y=False):
    # module_path = dirname(__file__)
    data = np.loadtxt(csv_file, delimiter=',')
    # with open(join(module_path, 'descr', 'digits.rst')) as f:
    #     descr = f.read()
    target = data[:, -1].astype(np.int)
    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 8, 8)

    if n_class < 128:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    if return_X_y:
        return flat_data, target

    return Bunch(data=flat_data,
                 target=target,
                 target_names=np.arange(10),
                 images=images)

if __name__ == "__main__":
    ftrs = load_csv_format_ftr("E:/HWRProj/Data/FeaturesData/smlImg/wt0404.csv")   # for 28*28 image
    X = ftrs.data
    y = ftrs.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    # knn = KNeighborsClassifier(n_neighbors=5)
    # print("Training started!")
    # knn.fit(X_train, y_train)
    # print("Evaluation started!")
    # print(knn.score(X_test, y_test))
    svm = SVC()
    print("Training started!")
    svm.fit(X_train, y_train)
    print("Evaluation started!")
    print(svm.score(X_test, y_test))
