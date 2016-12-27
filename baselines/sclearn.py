import evaluation
from sklearn.feature_selection import RFE
from collections import defaultdict
import numpy as np
from numpy import *
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import loaddata
import os
flags = tf.app.flags
FLAGS = flags.FLAGS

def Basemodel(_model,filename,cross_cv,tab_crosscv):
    filepath = FLAGS.data_dir
    sequence_window = FLAGS.sequence_window
    is_multi_scale = FLAGS.is_multi_scale
    training_level = FLAGS.scale_levels
    is_add_noise = FLAGS.is_add_noise
    noise_ratio = FLAGS.noise_ratio

    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY", "F1_SCORE", "AUC", "G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []
    np.random.seed(1337)  # for reproducibility
    # num_selected_features = 25#AS leak tab=0
    # num_selected_features = 32#Slammer tab=0
    num_selected_features = 33  # Nimda tab=1
    for tab_cv in range(cross_cv):
        if tab_crosscv == tab_cv:continue
        if _model == "SVM":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_crosscv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(_model + " is running..............................................")
            y_train = y_train0
            clf = svm.SVC(kernel="rbf", gamma=0.001, C=5000, probability=True)
            print(x_train.shape)
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

        elif _model == "SVMF":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=5)

            print(_model + " is running..............................................")
            clf = svm.SVC(kernel="rbf", gamma=0.00001, C=100000, probability=True)
            print(x_train.shape)
            # x_train_new = SelectKBest(f_classif, k=num_selected_features).fit_transform(x_train, y_train0)
            # x_test_new = SelectKBest(f_classif, k=num_selected_features).fit_transform(x_test, y_test0)

            clf.fit(x_train, y_train0)
            result = clf.predict_proba(x_test)

        elif _model == "SVMW":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=6)

            print(_model + " is running..............................................")
            # SVR(kernel="linear") = svm.SVC(kernel="rbf", gamma=0.00001, C=100000, probability=True)
            estimator = svm.SVC(kernel="linear", probability=True)
            selector = RFE(estimator, num_selected_features, step=1)
            selector = selector.fit(x_train, y_train0)

            result = selector.predict_proba(x_test)
        elif _model == "NBF":

            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=10)

            print(_model + " is running..............................................")
            clf = MultinomialNB()
            clf.fit(x_train, y_train0)
            result = clf.predict_proba(x_test)

        elif _model == "NBW":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=11)

            print(_model + " is running..............................................")
            # SVR(kernel="linear") = svm.SVC(kernel="rbf", gamma=0.00001, C=100000, probability=True)
            estimator = MultinomialNB()
            selector = RFE(estimator, num_selected_features, step=1)
            selector = selector.fit(x_train, y_train0)
            result = selector.predict_proba(x_test)

        elif _model == "NB":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=1)

            print(_model + " is running..............................................")
            y_train = y_train0
            clf = MultinomialNB()
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

        elif _model == "DT":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=2)

            print(_model + " is running.............................................." + str(x_train.shape))
            y_train = y_train0
            clf = tree.DecisionTreeClassifier()
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

        elif _model == "Ada.Boost":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(_model + " is running.............................................." + str(x_train.shape))
            y_train = y_train0
            # clf = AdaBoostClassifier(n_estimators=10) #Nimda tab=1
            clf = AdaBoostClassifier(n_estimators=10)

            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)


        results = evaluation.evaluation(y_test, result)  # Computing ACCURACY,F1-score,..,etc
        y_test2 = np.array(evaluation.ReverseEncoder(y_test))
        result2 = np.array(evaluation.ReverseEncoder(result))
        # Statistics False Alarm Rate
        if tab_cv == 2:
            with open(os.path.join(FLAGS.output,"StatFalseAlarm_" + filename + "_True"), "w") as fout:
                for tab in range(len(y_test2)):
                    fout.write(str(int(y_test2[tab])) + '\n')
            with open(os.path.join(FLAGS.output,"StatFalseAlarm_" + filename + "_" + _model + "_" + "_Predict"), "w") as fout:
                for tab in range(len(result2)):
                    fout.write(str(int(result2[tab])) + '\n')

        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)


    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    if is_add_noise == False:
        with open(os.path.join(FLAGS.output, "Comparison_Log_" + filename), "a")as fout:
            outfileline = _model + ":__"
            fout.write(outfileline)
            for eachk, eachv in result_list_dict.items():
                fout.write(eachk + ": " + str(round(eachv, 3)) + ",\t")
            fout.write('\n')
    print(results)
    return results
    # return epoch_training_loss_list,epoch_val_loss_list


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


