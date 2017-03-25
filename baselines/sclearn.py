import evaluation
from sklearn.feature_selection import RFE,SelectKBest
from collections import defaultdict
import printlog
#import ucr_load_data
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from numpy import *
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import loaddata
import os
import visualize
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2,f_classif
import sys
import collections
import itertools
from scipy.stats import mode
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
flags = tf.app.flags
FLAGS = flags.FLAGS

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False
def pprint(msg,method=''):
    if not 'Warning' in msg:
        sys.stdout = printlog.PyLogger('',method)
        print(msg)
        try:
            sys.stderr.write(msg+'\n')
        except:
            pass

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """

    def __init__(self, n_neighbors=1, max_warping_window=10, subsample_step=10):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = 65535.0 * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if (np.array_equal(x, y)):
            x_s = shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            p = ProgressBar(shape(dm)[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])

                    dm_count += 1
                    p.animate(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]

            p = ProgressBar(dm_size)

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)

            return dm

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """

        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        #return mode_label.ravel(), mode_proba.ravel()
        return mode_label.ravel()

class ProgressBar:
    """This progress bar was taken from PYMC
    """

    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print
        '\r', self,
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

def knn(train,test,w):
    preds=[]
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        #print ind
        for j in train:
            if LB_Keogh(i[:-1],j[:-1],5)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j
        preds.append(closest_seq[-1])
    return classification_report(test[:,-1],preds)


def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return sqrt(LB_sum)

def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])




def Basemodel(_model,filename,trigger_flag,evalua_flag,is_binary_class,evaluation_list):

    filepath = FLAGS.data_dir
    sequence_window = FLAGS.sequence_window
    is_multi_scale = FLAGS.is_multi_scale
    training_level = FLAGS.scale_levels
    is_add_noise = FLAGS.is_add_noise
    noise_ratio = FLAGS.noise_ratio

    result_list_dict = defaultdict(list)

    for each in evaluation_list:
        result_list_dict[each] = []
    np.random.seed(1337)  # for reproducibility
    # num_selected_features = 25#AS leak tab=0
    # num_selected_features = 32#Slammer tab=0
    #num_selected_features = 20  # Nimda tab=1
    x_train, y_train, x_val, y_val, x_test, y_test = loaddata.get_data_withoutS(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, FLAGS.data_dir,
                                            filename, FLAGS.sequence_window, trigger_flag,
                                            multiScale=False, waveScale=FLAGS.scale_levels,
                                            waveType=FLAGS.wave_type)
    for tab_selected_features in range(2,34):
    #for tab_selected_features in range(33):

        if _model == '1NN':
            #x_train, y_train, x_test, y_test = ucr_load_data.load_ucr_data()
            print(_model + " is running..............................................")
            clf = KNeighborsClassifier(n_neighbors=1)
        elif _model == '1NN-DTW':
            x_train, y_train, x_val, y_val, x_test, y_test = loaddata.get_data(FLAGS.pooling_type, FLAGS.is_add_noise,
                                                                               FLAGS.noise_ratio, FLAGS.data_dir,
                                                                               filename, FLAGS.sequence_window,
                                                                               trigger_flag,
                                                                               multiScale=False,
                                                                               waveScale=FLAGS.scale_levels,
                                                                               waveType=FLAGS.wave_type)
            x_train = np.transpose(x_train,[0,2,1])
            x_test = np.transpose(x_test,[0,2,1])
            print(_model + " is running..............................................")

            clf = KnnDtw(n_neighbors=1)
            x_train = x_train[:,tab_selected_features]
            x_test = x_test[:,tab_selected_features]

        elif _model == 'RF':
            clf = RandomForestClassifier(n_estimators=50)
        elif _model == "SVM":
            print(_model + " is running..............................................")
            clf = svm.SVC(kernel="rbf", gamma=0.0001, C=100000, probability=True)

        elif _model == "SVMF":
            x_train, y_train, x_val, y_val, x_test, y_test = loaddata.get_data_withoutS(FLAGS.pooling_type,
                                                                                        FLAGS.is_add_noise,
                                                                                        FLAGS.noise_ratio,
                                                                                        FLAGS.data_dir,
                                                                                        filename, FLAGS.sequence_window,
                                                                                        trigger_flag,
                                                                                        multiScale=False,
                                                                                        waveScale=FLAGS.scale_levels,
                                                                                        waveType=FLAGS.wave_type)
            print(_model + " is running.............................................."+str(tab_selected_features))
            clf = svm.SVC(kernel="rbf", gamma=0.0001, C=100000, probability=False)
            estimator = SelectKBest(chi2, k=tab_selected_features)

            x_train_new = estimator.fit_transform(x_train, y_train)
            x_test_new = estimator.fit_transform(x_test, y_test)

            x_train = x_train_new
            x_test = x_test_new



        elif _model == "SVMW":
            print(_model + " is running.............................................."+str(tab_selected_features))
            #estimator = svm.SVC(kernel="rbf", gamma=0.00001, C=1000, probability=False)
            estimator = svm.SVC(kernel="linear")
            clf = RFE(estimator, tab_selected_features, step=1)


        elif _model == "NB":
            print(_model + " is running..............................................")
            clf = BernoulliNB()

        elif _model == "DT":
            print(_model + " is running.............................................." + str(x_train.shape))
            y_train = y_train
            clf = tree.DecisionTreeClassifier()

        elif _model == "Ada.Boost":
            print(_model + " is running.............................................." + str(x_train.shape))
            y_train = y_train
            clf = AdaBoostClassifier()

        visualize.curve_plotting(x_test,y_test,filename,_model)
        clf.fit(x_train, y_train)
        if evalua_flag == True:
            result = clf.predict(x_test)

        else:
            y_test = loaddata.one_hot(y_test)
            result = clf.predict_proba(x_test)
            print(y_test.shape)
            print(result.shape)

        if is_binary_class == True:
            #print(y_test.shape)
            #print(result.shape)
            #print(evalua_flag)
            results = evaluation.evaluation(y_test, result,trigger_flag,evalua_flag)  # Computing ACCURACY,F1-score,..,etc
        else:
            accuracy = sklearn.metrics.accuracy_score(y_test,result)
            symbol_list = [0,1,2,3,4]
            confmat = confusion_matrix(y_test, result, labels=symbol_list)
            visualize.plotConfusionMatrix(confmat)

            #for symbol in symbol_list:
             #   for tab in range(len(y_test)):
              #      if y_test[tab] == symbol and y_test[tab] == result[tab]:
               #         y_.append(symbol)
                #print(y_test[0:10])
                #rint(result[0:10])
                #print("Accuracy is :"+str(accuracy))
                #print("Accuracy of "+str(symbol)+" is :"+str(float(len(y_))/(list(result).count(symbol))))
            f1_score = sklearn.metrics.f1_score(y_test,result)
            print("F-score is :"+str(f1_score))
            results = {'ACCURACY':accuracy,'F1_SCORE':f1_score,'AUC':9999,'G_MEAN':9999}
        try:
            y_test2 = np.array(evaluation.ReverseEncoder(y_test))
            result2 = np.array(evaluation.ReverseEncoder(result))
        except:
            y_test2 = y_test
            result2 = result
        # Statistics False Alarm Rate
        with open(os.path.join("./stat/","StatFalseAlarm_" + filename + "_True"), "w") as fout:
            for tab in range(len(y_test2)):
                fout.write(str(int(y_test2[tab])) + '\n')
        with open(os.path.join("./stat/","StatFalseAlarm_" + filename + "_" + _model + "_" + "_Predict"), "w") as fout:
            for tab in range(len(result2)):
                fout.write(str(int(result2[tab])) + '\n')

        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)

        for each_eval in evaluation_list:
            result_list_dict[each_eval].append(results[each_eval])
        #for eachk, eachv in result_list_dict.items():
            #result_list_dict[eachk] = np.average(eachv)
        if evalua_flag:
            with open(os.path.join(FLAGS.output, "Comparison_Log_" + filename), "a")as fout:
                outfileline = _model + ":__"+str(tab_selected_features)
                fout.write(outfileline)
                for each_eval in evaluation_list:
                    fout.write(each_eval + ": " + str(round(np.average(result_list_dict[each_eval]), 3)) + ",\t")
                #for eachk, eachv in result_list_dict.items():
                    #fout.write(eachk + ": " + str(round(eachv, 3)) + ",\t")
                fout.write('\n')
        #if '-' in _model:break
        if 'MW' in _model or 'MF' in _model:
            continue
        else:
            break
    if evalua_flag:
        pass
        #return epoch_training_loss_list,epoch_val_loss_list
    else:return results



def epoch_loss_plotting(train_loss_list, val_loss_list):
    epoch = len(train_loss_list[0])
    x = [i + 1 for i in range(epoch)]
    plt.plot(x, train_loss_list[1], 'r-', label='multi-scale training loss')
    plt.plot(x, val_loss_list[1], 'b-', label='multi-scale val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
    plt.plot(x, val_loss_list[0], 'r-', label='original val loss')
    plt.plot(x, val_loss_list[1], 'g-', label='multi-scale val loss')
    plt.plot(x, val_loss_list[2], 'b-', label='multi-original val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()


