import c
import os
import da
import db2
import matplotlib
import matplotlib.pyplot as plt
def set_style():
    plt.style.use(['classic'])
    matplotlib.rc("font", family="serif")
set_style()
import tensorflow as tf
flags = tf.app.flags

filename_list = ["HB_AS_Leak.txt","HB_Slammer.txt","HB_Nimda.txt","HB_Code_Red_I.txt"]
#filename_list = ["HB_AS_Leak.txt"]
#filename_list = ["HB_Slammer.txt"]
#filename_list = ["HB_Nimda.txt"]
#filename_list = ["HB_Code_Red_I.txt"]

#method_list = ["SVM","NB","DT","Ada.Boost","MLP","RNN","LSTM","MS-LSTM"]
#fpr = [0 for i in range(len(method_list))]
#tpr = [0 for i in range(len(method_list))]
#auc = [0 for i in range(len(method_list))]


wave_type_list =['db1','db2','haar','coif1','db1','db2','haar','coif1','db1','db2']
multi_scale_value_list = [2,2,2,2,3,3,3,3,4,4]
for filename in filename_list:

    #Parameters["filename"] = filename
    for wave_type_tab in range(len(wave_type_list)):
        #Parameters["wave_type"] = wave_type_list[wave_type_tab]
        #Parameters["training_level"] = multi_scale_value_list[wave_type_tab]
        #da.Model("MS-LSTM",Parameters)
        pass
