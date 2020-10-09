import cv2
import os
import numpy as np
from keras.utils import np_utils


data_path_train='Train path goes here'
data_path_test='Test path goes here'
data_path_validation='Validation path goes here'

cate = ["WithMask","WithoutMask"]
label_value = {"WithMask":0,"WithoutMask":1}
img_size=50

train_X = []
train_Y = []

test_X = []
test_Y = []

valid_X = []
valid_Y = []

for category in cate:
    folder_path=os.path.join(data_path_train,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            
            resized=cv2.resize(gray,(img_size,img_size))

            train_X.append(resized)
            train_Y.append(label_value[category])
            
        except Exception as e:
            print('Exception:',e)
            

for category in cate:
    folder_path=os.path.join(data_path_test,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
           
            resized=cv2.resize(gray,(img_size,img_size))
            
            test_X.append(resized)
            test_Y.append(label_value[category])
            

        except Exception as e:
            print('Exception:',e)
          


for category in cate:
    folder_path=os.path.join(data_path_validation,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            
            resized=cv2.resize(gray,(img_size,img_size))
            
            valid_X.append(resized)
            valid_Y.append(label_value[category])
           
        except Exception as e:
            print('Exception:',e)
           



train_X=np.array(train_X)/255.0
train_X=np.reshape(train_X,(train_X.shape[0],img_size,img_size,1))
train_Y=np.array(train_Y)

test_X=np.array(test_X)/255.0
test_X=np.reshape(test_X,(test_X.shape[0],img_size,img_size,1))
test_Y=np.array(test_Y)

valid_X=np.array(valid_X)/255.0
valid_X=np.reshape(valid_X,(valid_X.shape[0],img_size,img_size,1))
valid_Y=np.array(valid_Y)

new_train_Y=np_utils.to_categorical(train_Y)
new_test_Y=np_utils.to_categorical(test_Y)
new_valid_Y=np_utils.to_categorical(valid_Y)

np.save('X_train',train_X)
np.save('y_train',new_train_Y)

np.save('X_test',test_X)
np.save('y_test',new_test_Y)

np.save('X_valid',valid_X)
np.save('y_valid',new_valid_Y)