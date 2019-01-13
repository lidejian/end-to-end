# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import read_data
from record import do_record
import config

flags = tf.flags


logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 10)")
FLAGS = flags.FLAGS

record_file = config.RECORD_PATH + '/first_class.csv'

# 加载数据
raw = read_data.raw_data(FLAGS.data_path)
train_sentence, dev_sentence, test_sentence, train_labels, dev_labels, test_labels, vocabulary_len = raw

# print(train_sentence)
# print(train_labels)


max_len = 320  # 句子最大长度

# 使用 pad_sequences 函数将长度标准化，统一为max_len长度
train_sentence = keras.preprocessing.sequence.pad_sequences(train_sentence, padding='post', maxlen=max_len)

dev_sentence = keras.preprocessing.sequence.pad_sequences(dev_sentence, padding='post', maxlen=max_len)

test_sentence = keras.preprocessing.sequence.pad_sequences(test_sentence, padding='post', maxlen=max_len)


print(train_sentence.shape)
print(dev_sentence.shape)
print(test_sentence.shape)

# 构建模型
vocab_size = vocabulary_len  # 所有单词数

# model = keras.Sequential()
#
#
# # 第一层是 Embedding 层。该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。模型在接受训练时会学习这些向量。
# # 这些向量会向输出数组添加一个维度。生成的维度为：(batch, sequence, embedding)
# model.add(keras.layers.Embedding(vocab_size, 16,input_shape=(None,max_len)))
#
# # 接下来，一个 GlobalAveragePooling1D 层通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。
# # 这样，模型便能够以尽可能简单的方式处理各种长度的输入
# model.add(keras.layers.GlobalAveragePooling1D())
#
# # model.add(keras.layers.LSTM(32), input_shape=(16,))
# model.add(keras.layers.LSTM(16))
#
# # # 该长度固定的输出向量会传入一个全连接 (Dense) 层（包含 16 个隐藏单元）
# # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#
# # 最后一层与单个输出节点密集连接。应用 softmax 函数
# model.add(keras.layers.Dense(4, activation='softmax'))
# print(model.summary())

inputs = keras.layers.Input(name='inputs',shape=[max_len])
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = keras.layers.Embedding(vocab_size+1,128,input_length=max_len)(inputs)
layer = keras.layers.LSTM(128)(layer)
layer = keras.layers.Dense(128,activation="relu",name="FC1")(layer)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(4, activation="softmax",name="FC2")(layer)
model = keras.models.Model(inputs=inputs,outputs=layer)
model.summary()




# 配置模型
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_sentence,
                    train_labels,
                    epochs=FLAGS.num_epochs,
                    batch_size=FLAGS.batch_size,
                    validation_data=(dev_sentence, dev_labels),
                    verbose=1)

# 评估模型
results = model.evaluate(test_sentence, test_labels)
print('The results is :', results)

# predictions = model.predict(test_sentence)
# print(predictions)

#
# # 保存到文件
# configuration = {
#     "data_path": FLAGS.data_path,
#     # "model": FLAGS.model,
#     # "share_rep_weights": FLAGS.share_rep_weights,
#     # "bidirectional": FLAGS.bidirectional,
#     #
#     # "cell_type": FLAGS.cell_type,
#     # "hidden_size": FLAGS.hidden_size,
#     # "num_layers": FLAGS.num_layers,
#     #
#     # "dropout_keep_prob": FLAGS.dropout_keep_prob,
#     # "l2_reg_lambda": FLAGS.l2_reg_lambda,
#     # "Optimizer": "AdaOptimizer",
#     # "learning_rate": FLAGS.learning_rate,
#     #
#     "batch_size": FLAGS.batch_size,
#     "num_epochs": FLAGS.num_epochs,
# }
#
# evaluation_result={
#     'loss':history.history['loss'][-1],
#     'acc': history.history['acc'][-1],
#     'val_loss': history.history['val_loss'][-1],
#     'val_acc': history.history['val_acc'][-1],
#     'test_loss': results[0],
#     'test_acc': results[1]
# }
#
# # fieldnames = ["f1", "p", "r", "acc", "train_data_dir", "model", "share_rep_weights",
# #                    "bidirectional", "cell_type",  "hidden_size", "num_layers",
# #                   "dropout_keep_prob", "l2_reg_lambda", "Optimizer", "learning_rate", "batch_size", "num_epochs", "w2v_type",
# #                   "additional_conf"
# # ]
# fieldnames = ['data_path','batch_size','num_epochs','loss','acc','val_loss','val_acc','test_loss','test_acc']
# do_record(fieldnames, configuration, evaluation_result, record_file)

import pickle

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
history_dict = history.history

with open('mydata.pickle', 'wb') as f:
    pickle.dump((FLAGS.batch_size, FLAGS.num_epochs, acc, val_acc, loss, val_loss, history_dict), f)
