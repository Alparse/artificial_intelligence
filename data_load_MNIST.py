import numpy as np
import h5py


def load_data():

    hdf5_file_name='C:\\Users\\aslan\\Desktop\\dataSets\\MNIST_Digits\\MNIST_Digits.h5'

    data=h5py.File(hdf5_file_name,'r')
    test_data=np.array(data['test'])
    train_data=np.array(data['train'])

    print("Test Data Shape "+str(np.shape(test_data)))
    print("Train Data Shape "+ str(np.shape(train_data)))

    X_test=test_data[:,1:]/255
    X_train=train_data[:,1:]/255

    Y_test_values = np.reshape(test_data[:, 0],(np.size(test_data,0),1))
    Y_test=np.zeros((np.size(Y_test_values,0),10))

    for i in range (np.size(Y_test,0)):
        Y_test[i,Y_test_values[i,0]]=1

    Y_train_values = np.reshape(train_data[:, 0],(np.size(train_data,0),1))
    Y_train=np.zeros((np.size(Y_train_values,0),10))
    for i in range (np.size(Y_train,0)):
        Y_train[i,Y_train_values[i,0]]=1

    print('X Train Shape '+ str(np.shape(X_train)))
    print('Y Train Shape '+str(np.shape(Y_train)))
    print('X Test Shape '+str(np.shape(X_test)))
    print('Y Test Shape '+ str(np.shape(Y_test)))

    return X_train,Y_train,X_test,Y_test

load_data()