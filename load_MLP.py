'''

from keras.models import load_model
from keras.models import model_from_json
models = []
for i in range(3):
    json_file = open(str(i)+ '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('model'+ str(i)+ '.h5')
    models.append(model)

# 徐新的特征要先归一化一下
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
# 训练集
Data = genfromtxt('2w_train.csv', delimiter=',')
#y_train = Data
X_train = Data
# 测试集
X_test = genfromtxt('ans.csv', delimiter=',')

# 做一下归一化，把变量名也改一下
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

#scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)
'''
import pandas as pd
import numpy as np
import os

submission_files = pd.read_csv('../Kaggle_Earthquake_challenge/sample_submission.csv',index_col='seg_id')
results = pd.Series([0 for i in range(len(submission_files))], dtype=np.float64)
# 之后就可以进行预测了，循环预测三次，最终结果保存三次预测的均值
'''
for k,i in enumerate(models):
    y_test = i.predict(X_test_scaled)
    submission_files['time_to_failure'] = y_test
    submission_files.to_csv('solutions/'+ str(k)+ 'th_3_layer_MLP_new.csv')
'''
files = os.listdir('solutions')

for i in files:
        temp = pd.read_csv('solutions/'+ i)
        results += temp['time_to_failure']
results = results / len(files)
submission_files['time_to_failure'] = list(results)
submission_files.to_csv('ensemble_submission.csv')

