# 依赖
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# 初始化深度学习模型
# Use Keras
from keras import models
from keras import layers
from keras import regularizers
import tensorflow as tf


from tqdm import tqdm
from numpy import genfromtxt
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats

# 加载特征
submission_files = pd.read_csv('../Kaggle_Earthquake_challenge/sample_submission.csv',index_col='seg_id')
#X_test_scaled = pd.read_csv('test_80k_kernel.csv').set_index('seg_id')
#X_train_scaled = pd.read_csv('train_80k_kernel.csv').drop(columns = ['Unnamed: 0'])
#y_train = pd.read_csv('train_80k_y_kernel.csv').drop(columns = ['Unnamed: 0'])


# 加载数据集
Data = genfromtxt('train_feature_data.csv', delimiter=',')
X_train = Data

Data = genfromtxt('train_label_data.csv', delimiter = ',')
y_train = Data

# 测试集
X_test = genfromtxt('test_feature_data.csv', delimiter=',')
X_test = X_test.transpose()


# 进行特征归一化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

#scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)


def build_model():
    model = models.Sequential()
    # model.add(layers.Dense(256, activation = 'relu', input_shape = (X_train_scaled.shape[1],)))
    model.add(layers.Dense(1024, kernel_regularizer = regularizers.l1(0.001), activation = 'relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization(axis = -1, momentum=0.99, epsilon = 0.001,
                                        center=True, scale = True, beta_initializer= 'zeros',
                                        gamma_initializer='ones', moving_mean_initializer='zeros',
                                        moving_variance_initializer='ones', beta_regularizer=None,
                                        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        ))

    model.add(layers.Dense(2048, kernel_regularizer = regularizers.l1(0.0005), activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization(axis = -1, momentum=0.99, epsilon = 0.001,
                                        center=True, scale = True, beta_initializer= 'zeros',
                                        gamma_initializer='ones', moving_mean_initializer='zeros',
                                        moving_variance_initializer='ones', beta_regularizer=None,
                                        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        ))

    model.add(layers.Dense(1024, kernel_regularizer = regularizers.l1(0.001), activation = 'relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization(axis = -1, momentum=0.99, epsilon = 0.001,
                                        center=True, scale = True, beta_initializer= 'zeros',
                                        gamma_initializer='ones', moving_mean_initializer='zeros',
                                        moving_variance_initializer='ones', beta_regularizer=None,
                                        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        ))

    model.add(layers.Dense(512, kernel_regularizer = regularizers.l1(0.001), activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization(axis = -1, momentum=0.99, epsilon = 0.001,
                                        center=True, scale = True, beta_initializer= 'zeros',
                                        gamma_initializer='ones', moving_mean_initializer='zeros',
                                        moving_variance_initializer='ones', beta_regularizer=None,
                                        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        ))

    model.add(layers.Dense(256, kernel_regularizer = regularizers.l1(0.001), activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization(axis = -1, momentum=0.99, epsilon = 0.001,
                                        center=True, scale = True, beta_initializer= 'zeros',
                                        gamma_initializer='ones', moving_mean_initializer='zeros',
                                        moving_variance_initializer='ones', beta_regularizer=None,
                                        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        ))

    model.add(layers.Dense(128, kernel_regularizer = regularizers.l2(0.001)  , activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization(axis = -1, momentum=0.99, epsilon = 0.001,
                                        center=True, scale = True, beta_initializer= 'zeros',
                                        gamma_initializer='ones', moving_mean_initializer='zeros',
                                        moving_variance_initializer='ones', beta_regularizer=None,
                                        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        ))
    # model.
    model.add(layers.Dense(64, kernel_regularizer = regularizers.l2(0.01)  , activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization(axis = -1, momentum=0.99, epsilon = 0.001,
                                        center=True, scale = True, beta_initializer= 'zeros',
                                        gamma_initializer='ones', moving_mean_initializer='zeros',
                                        moving_variance_initializer='ones', beta_regularizer=None,
                                        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        ))
    model.add(layers.Dense(1))

    model.compile(optimizer = 'Adam',
                  loss = 'mae',
                  metrics = ['mae'])
    return model




# K 折验证, 生成三个模型
k = 3

num_val_samples = len(X_train_scaled) // k
num_epochs = 1000
all_scores = []
all_mae_histories = []
model_list = []


for i in range(k):
    print('processing fold #', i)
    val_data = X_train_scaled[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(

        [X_train_scaled[:i * num_val_samples], X_train_scaled[(i + 1) * num_val_samples:]], axis=0

    )
    partial_train_targets = np.concatenate(

        [y_train[:i * num_val_samples], y_train[(i + 1) * num_val_samples:]], axis=0
    )
    model = build_model()
    # 训练模型

    history = model.fit(partial_train_data, partial_train_targets,
            validation_data= (val_data, val_targets),
            epochs = num_epochs,
            verbose= 2,
            batch_size=64
            )

    #val_mse, val_mae = model.evaluate(val_data, val_targets)
    #all_scores.append(val_mae)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    model_list.append(model)


# 计算所有轮次K折验证平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print('the mean validation loss is'.format(average_mae_history))



# 进行模型集成预测
results = pd.Series([0 for i in range(len(submission_files))], dtype=np.float64)
for k,i in enumerate(model_list):
    y_test = i.predict(X_test_scaled)
    results += y_test.flatten()
results = results / (k + 1)
submission_files['time_to_failure'] = list(results)
submission_files.to_csv('ensemble_submission.csv')




'''
# 之后就可以进行预测了，循环预测三次，最终结果保存三次预测的均值
for k,i in enumerate(model_list):
    y_test = i.predict(X_test_scaled)
    submission_files['time_to_failure'] = y_test
    submission_files.to_csv('solutions/'+ str(k)+ 'th_3_layer_MLP_new.csv')

files = os.listdir('solutions')

for i in files:
        temp = pd.read_csv('solutions/'+ i)
        results += temp['time_to_failure']
results = results / len(files)
submission_files['time_to_failure'] = list(results)
submission_files.to_csv('ensemble_submission.csv')



# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
'''







'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)
'''


'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))
'''
'''
def build_model():
    model = models.Sequential()
    # model.add(layers.Dense(256, activation = 'relu', input_shape = (X_train_scaled.shape[1],)))
    model.add(layers.Dense(128, kernel_regularizer = regularizers.l2(0.001)  , activation = 'relu'))
    model.add(layers.Dropout(0.2))
    # model.
    model.add(layers.Dense(64, kernel_regularizer = regularizers.l2(0.01)  , activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, kernel_regularizer=  regularizers.l2(0.1),   activation = 'relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer = 'rmsprop',
                  loss = 'mae',
                  metrics = ['mae'])
    return model
'''








'''

# 绘制验证分数
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 绘制验证分数

def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
     if smoothed_points:
        previous = smoothed_points[-1]
        smoothed_points.append(previous * factor + point * (1-factor))
     else:
         smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history)+ 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

'''



'''
# serialize model to JSON

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

'''





'''
model = build_model()

model.fit(X_train_scaled, y_train, epochs = 1000, batch_size = 64,verbose= 2)

y_test = model.predict(X_test_scaled)

submission_files = pd.read_csv('../Kaggle_Earthquake_challenge/sample_submission.csv',index_col='seg_id')

submission_files['time_to_failure'] = y_test
submission_files.to_csv('solutions/3_layer_MLP.csv')

files = os.listdir('solutions')
print(files)
submission_files = pd.read_csv('../Kaggle_Earthquake_challenge/sample_submission.csv',index_col='seg_id')
results = pd.Series([0 for i in range(len(submission_files))],dtype=np.float64)
for i in files:
        temp = pd.read_csv('solutions/'+ i)
        results += temp['time_to_failure']
results = results / len(files)
submission_files['time_to_failure'] = list(results)
submission_files.to_csv('ensemble_submission.csv')
'''

'''
for k,i in enumerate(model_list):
    with open(str(k)+',json','w') as json_file:
        json_file.write(i.to_json())
        # serialize weights to H5
        i.save_weights('model'+str(k)+'.h5')
        print('saved model' + str(i))
'''



