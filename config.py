import tensorflow as tf
import os
flags = tf.app.flags
flags.DEFINE_string('data_dir',os.path.join(os.getcwd(),'BGP_Data'),"""Directory for storing BGP_Data set""")
flags.DEFINE_string('is_multi_scale',False,"""Run with multi-scale or not""")
flags.DEFINE_string('input_dim',33,"""Input dimension size""")
flags.DEFINE_string('num_neurons1',32,"""Number of hidden units""")#HAL(hn1=32,hn2=16)
flags.DEFINE_string('num_neurons2',8,"""Number of hidden units""")#16,32
flags.DEFINE_string('sequence_window',30,"""Sequence window size""")
flags.DEFINE_string('attention_size',10,"""attention size""")
flags.DEFINE_string('scale_levels',10,"""Scale level value""")
flags.DEFINE_string('number_class',2,"""Number of output nodes""")
flags.DEFINE_string('max_grad_norm',5,"""Maximum gradient norm during training""")
flags.DEFINE_string('wave_type','haar',"""Type of wavelet""")
flags.DEFINE_string('pooling_type','max pooling',"""Type of wavelet""")
flags.DEFINE_string('batch_size',1000,"""Batch size""")
flags.DEFINE_string('max_epochs',100,"""Number of epochs to run""")
flags.DEFINE_string('learning_rate',0.01,"""Learning rate""")
flags.DEFINE_string('is_add_noise',False,"""Whether add noise""")
flags.DEFINE_string('noise_ratio',0,"""Noise ratio""")
flags.DEFINE_string('option','AL',"""Operation[1L:one-layer lstm;2L:two layer-lstm;HL:hierarchy lstm;HAL:hierarchy attention lstm]""")
flags.DEFINE_string('log_dir','./log/',"""Directory where to write the event logs""")
flags.DEFINE_string('output','./output/',"""Directory where to write the results""")

