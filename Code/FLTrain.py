import numpy as np
from sklearn.cluster import KMeans

def computing_sigma(alpha,gamma,norm):
    delta = 10**(-8)
    epsilon = gamma - np.log(delta)/(alpha-1)
    sigma = np.sqrt(2*np.log(1.25/delta))*norm/epsilon
    return sigma
    
def LDP(weights,sigma,delta):
    norm = np.sqrt(np.square(weights).sum())
    #print(norm)
    if norm > delta and sigma>0:
        weights = weights*delta/norm
    if sigma>0:
        weights = weights + np.random.normal(0,sigma,size=weights.shape)
    return weights

def renorm(w1,w2):
    n1 = np.sqrt(np.square(w1).sum())
    n2 = np.sqrt(np.square(w2).sum())
    return w1*n2/n1

def func_flatten_weights(weights):
    a = []
    for i in range(len(weights)):
        w = weights[i].reshape((-1,))
        a.append(w)
    a = np.concatenate(a)
    return a 

def func_unflatten_weights(f_weights,old_weights):
    r_weights = []
    start = 0
    for i in range(len(old_weights)):
        ed = start + old_weights[i].reshape((-1,)).shape[0]
        lw = f_weights[start:ed]
        lw = lw.reshape(old_weights[i].shape)
        r_weights.append(lw)
        start = ed
    return r_weights

def RobustAggergation(mode,all_weights,old_weights):
    flaten_all_weights = all_weights
    
    if mode == 'FedAvg':
        agg_weights = np.mean(flaten_all_weights,axis=0)
    
    elif mode == 'Mid':
        agg_weights = np.median(flaten_all_weights,axis=0)
        
    elif mode == 'Krum':
        agg_weights = np.mean(flaten_all_weights,axis=0).reshape((1,flaten_all_weights.shape[1]))
        distance = np.abs(agg_weights-flaten_all_weights).sum(axis=-1)
        inx = distance.argmin()
        agg_weights = flaten_all_weights[inx]
        
    elif mode == 'MultiKrum':
        agg_weights = np.mean(flaten_all_weights,axis=0).reshape((1,flaten_all_weights.shape[1]))
        distance = np.abs(agg_weights-flaten_all_weights).sum(axis=-1)
        inx = distance.argsort()[:len(distance)//2]
        agg_weights = flaten_all_weights[inx].mean(axis=0)
        
        
    elif mode == 'Norm':
        norms = np.sqrt((flaten_all_weights**2).sum(axis=-1))
        
        mean_norms = norms.mean()
        norms/= mean_norms
        norms[norms<1] = 1
        
        norms = norms.reshape((len(norms),1))

        flaten_all_weights = flaten_all_weights/norms
        agg_weights = flaten_all_weights.mean(axis=0)
        
    elif mode == 'CONTRA':
        agg_weights = np.mean(flaten_all_weights,axis=0).reshape((1,flaten_all_weights.shape[1]))
        
        norms = np.sqrt((flaten_all_weights**2).sum(axis=-1))
        norms = norms.reshape((len(norms),1))
        cos_flaten_all_weights = flaten_all_weights/norms
        cos = (cos_flaten_all_weights*agg_weights).sum(axis=-1)
        index = np.where(cos>cos.mean())[0]
        agg_weights = flaten_all_weights[index].mean(axis=0)
        
    elif mode =='RobustDPFLM':
        z = np.abs(flaten_all_weights.mean(axis=-1))
        index = np.where(z<z.mean())[0].tolist()
        if len(index)==0:
            return old_weights
        agg_weights = flaten_all_weights[index].mean(axis=0)
        
    elif mode =='RobustDPFLC':
        z = np.abs(flaten_all_weights.mean(axis=-1))
        y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(z.reshape((-1,1)))
        index0 = np.where(y_pred==0)[0]
        index1 = np.where(y_pred==1)[0]
        if z[index0].mean()<z[index1].mean():
            index = index0
        else:
            index = index1
        agg_weights = flaten_all_weights[index].mean(axis=0)
        
    elif mode =='RobustDPFLT':
        z = np.abs(flaten_all_weights.mean(axis=-1))
        index = np.where(z<10**(-4))[0].tolist()
        if len(index)==0:
            return old_weights
        agg_weights = flaten_all_weights[index].mean(axis=0)
                
    agg_weights = func_unflatten_weights(agg_weights,old_weights)
    
    return agg_weights


def FL(attack_mode,mode,user_num,model,taxic_clients,train_users,train_images,train_labels,sigma=0.1,delta=10,intre_epoch=5):
    #user_indexs = np.random.randint(len(train_users),size=(user_num,))
    user_indexs = np.random.permutation(len(train_users))[:user_num]
    
    old_weights = model.get_weights()
    flatten_old_weights = func_flatten_weights(old_weights)
    all_weights = []
    f_possioned_gradients = []
    quotas = []
    for ui in range(len(user_indexs)):
        ui = user_indexs[ui]
        sample_indexs = train_users[ui]
        x = train_images[sample_indexs]
        y = train_labels[sample_indexs]
        bz = len(x)//5
        for i in range(intre_epoch):
            model.fit(x,y,verbose=0)
        
        weights = model.get_weights()
        weights = func_flatten_weights(weights)
        
        delta_weights = weights-flatten_old_weights
        
        if ui in taxic_clients:
            if attack_mode == 'V1':
                delta_weights = LDP(delta_weights,sigma,delta)
                weights = delta_weights+flatten_old_weights
                all_weights.append(weights)
            elif attack_mode == 'V2':
                weights = delta_weights+flatten_old_weights
                all_weights.append(weights)
            elif attack_mode == 'V3':
                weights2 = LDP(delta_weights,sigma,delta)
                weights2 = weights2+flatten_old_weights
                weights = delta_weights+flatten_old_weights
                weights = renorm(weights,weights2)
                all_weights.append(weights)
        else:
            delta_weights = LDP(delta_weights,sigma,delta)
            weights = delta_weights+flatten_old_weights
            all_weights.append(weights)
            
        model.set_weights(old_weights)

    all_weights = np.array(all_weights)
    weights = RobustAggergation(mode,all_weights,old_weights)    
    model.set_weights(weights)
