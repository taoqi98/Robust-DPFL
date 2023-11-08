import numpy as np
import os 
import pickle as pkl
from emnist import extract_training_samples,extract_test_samples
import numpy as np
import struct
from keras.utils.np_utils import to_categorical
import json

def load_mnist_data(mode):
    if mode == 'train':
        file_path = '../Data/MNIST//train-images.idx3-ubyte'
        label_path = '../Data/MNIST//train-labels.idx1-ubyte'
    else:
        file_path = '../Data/MNIST//t10k-images.idx3-ubyte'
        label_path = '../Data/MNIST//t10k-labels.idx1-ubyte'
        
    binfile = open(file_path, 'rb') 
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    
    images = images.reshape((len(images),28,28,1))
    
    binfile = open(label_path, 'rb')
    buffers = binfile.read()
    magic,num = struct.unpack_from('>II', buffers, 0) 
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    
    images = images/255
    labels = to_categorical(labels)
    
    return images,labels

train_images, train_labels = load_mnist_data('train')
test_images, test_labels = load_mnist_data('test')


def load_femnist_data():
    train_images, train_labels = extract_training_samples('byclass')
    test_images, test_labels = extract_test_samples('byclass')

    train_images = train_images/255
    test_images = test_images/255

    train_images = train_images.reshape(list(train_images.shape)+[1])
    test_images = test_images.reshape(list(test_images.shape)+[1])

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return train_images, train_labels, test_images, test_labels


def load_cifar10_data(path='../Data/CIFAR10'):
    
    train_data = []
    train_label = []
    for i in range(5):
        with open(os.path.join(path,'data_batch_'+str(i+1)),'rb') as f:
            data = pkl.load(f, encoding='bytes')
            train_data.append(data[b'data'])
            train_label.append(data[b'labels'])
    train_data = np.concatenate(train_data,axis=0)
    train_label = np.concatenate(train_label,axis=0)
    train_data = train_data.reshape((-1,3,32,32))
    
    with open(os.path.join(path,'test_batch'),'rb') as f:
        data = pkl.load(f, encoding='bytes')
        test_data = data[b'data'].reshape((-1,3,32,32))
        test_label = data[b'labels']
        
    test_label = np.array(test_label)
        
    train_images = train_data/255
    test_images = test_data/255
    

    train_labels = to_categorical(train_label)
    test_labels = to_categorical(test_label)

    train_images = np.transpose(train_images,(0,2,3,1))
    test_images  = np.transpose(test_images, (0,2,3,1))

    return train_images,train_labels,test_images,test_labels

def load_data(dataset):
    
    if dataset == 'MNIST':
        train_images, train_labels = load_mnist_data('train')
        test_images, test_labels = load_mnist_data('test')

    elif dataset == 'FEMNIST':
        train_images, train_labels, test_images, test_labels = load_femnist_data()
    elif dataset == 'CIFAR10':
        train_images,train_labels, test_images,test_labels = load_cifar10_data()

    return train_images, train_labels, test_images, test_labels, train_images.shape[-1],train_labels.shape[-1]

def client_partation(train_labels,range_length = 1000):
    index = np.random.permutation(len(train_labels))

    train_users = {}

    for i in range(int(np.ceil(len(train_labels)//range_length))):
        s,ed = range_length*i, (i+1)*range_length
        ed = min(ed,len(train_labels))
        train_users[i] = index[s:ed]
        
    return train_users


def dump_result(Res,dataset,attack_mode,defense_mode,taxic_ratio,alpha,epsilon):
    Key = '-'.join([dataset,attack_mode,defense_mode,str(taxic_ratio),str(alpha),str(epsilon)])
    with open('../Result/'+Key+'.json','a') as f:
        s = json.dumps(Res) + '\n'
        f.write(s)