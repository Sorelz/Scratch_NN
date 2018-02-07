import numpy as np

def scratch_split(x,y,test_pct,seed):
    np.random.seed(seed)
    assert(len(x)==len(y))
    shuffled=np.random.permutation(len(x))
    test_size=int(test_pct*len(x))
    x_test=x[shuffled][:test_size]
    y_test=y[shuffled][:test_size]
    x_train=x[shuffled][test_size:]
    y_train=y[shuffled][test_size:]
    return x_train,x_test,y_train,y_test

def standardize_fit_transform(dataset):
    data=np.copy(dataset)
    fitted_mean=np.mean(data,axis=0)
    fitted_std=np.std(data,axis=0,ddof=1) #ddof for the unbiased std
    data=np.divide(dataset-fitted_mean,fitted_std)
    return data,fitted_mean,fitted_std

def standardize_transform(dataset,f_mean,f_std):
    return np.divide(dataset-f_mean,f_std)
            
            
            
