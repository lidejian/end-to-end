# -*- coding:utf-8 -*-
import os
import config

# # ''' base line '''
# train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"

# # model = "CNN"
# model = "RNN"
# # model = "Attention_RNN1"
# # model = "Attention_RNN2" # mine
# # model = "Attention_RNN3"
# # model = "Attention_RNN4"
# # model = "Attention_RNN5"
# share_rep_weights = True
# bidirectional = True
# cell_type = "BasicLSTM"
# hidden_size = 50
# num_layers = 1
# dropout_keep_prob = 0.5
# l2_reg_lambda = 0.0
# learning_rate = 0.005
# batch_size = 64
# num_epochs = 20
# evaluate_every = 10


cmd = "python parser.py" \
      + " --data_path %s" % config.data_path


print(cmd)
os.system(cmd)