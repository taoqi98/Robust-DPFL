import numpy as np
from keras.utils.np_utils import to_categorical


class TriggerGenerator:
    def __init__(self,size,x,y,pix,label,NUM_CHANNEL,NUM_CLASS):
        self.trigger = np.zeros((size,size,NUM_CHANNEL)) + pix
        self.trigger = self.trigger/255
        
        self.size = size
        self.x = x
        self.y = y
        self.pix = pix
        self.label = to_categorical(label,num_classes=NUM_CLASS)
        
    def insert_trigger(self,data):
        data2 = np.copy(data)
        data2[self.x:self.x+self.size,self.y:self.y+self.size,:] = self.trigger
        return data2,self.label

def posison_data(taxic_ratio,trigger,train_users,train_images,train_labels):

    user_index = np.random.permutation(len(train_users))
    r = taxic_ratio
    N = int(r*len(user_index))
    N = max(1,N)
    all_trigger_images = []
    all_trigger_labels = []
    start = len(train_labels)

    for i in range(N):
        uix = user_index[i]
        data_index = train_users[uix]

        local_trigger_images = []
        local_trigger_labels = []

        for j in range(len(data_index)):
            inx = data_index[j]
            data = trigger.insert_trigger(train_images[inx])
            local_trigger_images.append(data[0])
            local_trigger_labels.append(data[1])

        local_trigger_images = np.array(local_trigger_images)
        local_trigger_labels = np.array(local_trigger_labels)

        all_trigger_images.append(local_trigger_images)
        all_trigger_labels.append(local_trigger_labels)


        ed = start + len(local_trigger_images)
        local_trigger_index = np.array([i for i in range(start,ed)])
        start = ed
        data_index = np.concatenate([data_index,local_trigger_index],axis=0)
        shuffled_data_index = np.random.permutation(len(data_index))
        data_index = data_index[shuffled_data_index]
        train_users[uix] = data_index

    all_trigger_images = np.concatenate(all_trigger_images,axis=0)
    all_trigger_labels = np.concatenate(all_trigger_labels,axis=0)

    train_images = np.concatenate([train_images,all_trigger_images],axis=0)
    train_labels = np.concatenate([train_labels,all_trigger_labels],axis=0)

    taxic_clients = set(user_index[:N].tolist())
    
    return train_images, train_labels, taxic_clients

def build_posison_eval_data(trigger,test_images,test_labels):
    trigger_test_images = []
    trigger_test_labels = []
    r = 1
    N = int(r*len(test_images))
    for i in range(N):
        data = trigger.insert_trigger(test_images[i])
        if test_labels[i].argmax() == trigger.label.argmax():
            continue
        trigger_test_images.append(data[0])
        trigger_test_labels.append(data[1])

    trigger_test_images = np.array(trigger_test_images)
    trigger_test_labels = np.array(trigger_test_labels)
    
    return trigger_test_images, trigger_test_labels