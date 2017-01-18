import os
import visualize
global tempstdout
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
sequence_window = 20
learning_rate = 0.04
def load_log(filename,file_folder):
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    with open(os.path.join(file_folder,filename))as fin:
        for eachline in fin.readlines():
            lin1 = eachline.split('=')
            lin2 = lin1[-1].split(',',4)
            train_accuracy.append(lin2[0].split(':')[-1].strip())
            train_loss.append(lin2[1].split(':')[-1].strip())
            val_accuracy.append(lin2[2].split(':')[-1].strip())
            val_loss.append(lin2[3].split(':')[-1].strip())
    return train_accuracy,train_loss,val_accuracy,val_loss


# main function
# filename_list = ["HB_AS_Leak.txt", "HB_Slammer.txt", "HB_Nimda.txt", "HB_Code_Red_I.txt"]
filename = "HB_AS_Leak.txt"

# wave_type_list =['db1','db2','haar','coif1','db1','db2','haar','coif1','db1','db2']
wave_type_list = ['db1']

multi_scale_value_list = [2, 3, 4, 5, 6, 10]
# multi_scale_value_list = [2,2,2,2,3,3,3,3,4,4]
#case_ = ['1L','HAL','AL','HL']
#case_ = ['1L','2L','AL','HL','HAL']
case_ = ['AL','HL','HAL']
#case_ = ['1L','2L','HL']
#case_ = ['AL','HL','HAL']

# case = ["SVM","SVMF","SVMW","NB","NBF","NBW","DT","Ada.Boost"]
# method_list2 = ["MLP","RNN","LSTM"]

case_label = {'1L': 'LSTM', '2L': '2-LSTM', 'AL': 'ALSTM', 'HL': 'HLSTM', 'HAL': 'HALSTM'}
case_list = [case_label[k] for k in case_]
cross_cv = 2
tab_cross_cv = 1
wave_type = wave_type_list[0]
hidden_units = 64
file_list = [('server_'+i+'_'+filename+'.log') for i in case_]
folder = os.path.join(os.getcwd(),'tmp')
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []
for each_file in file_list:
    train_acc, train_loss, val_acc, val_loss = load_log(each_file, folder)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

visualize.epoch_acc_plotting(filename, case_list, sequence_window, tab_cross_cv, learning_rate,
                             train_acc_list, val_acc_list)
visualize.epoch_loss_plotting(filename, case_list, sequence_window, tab_cross_cv, learning_rate,
                              train_loss_list, val_loss_list)


visualize.plotStat(filename,case_)
