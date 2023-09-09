import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import tracemalloc
from sklearn.model_selection import train_test_split



l_data  = "./data/ml-latest/ratings.csv"
s_data = "./data/ml-latest-small/ratings.csv"

# reading small data_set
s_df = pd.read_csv(s_data, usecols = ['userId','movieId','rating'], dtype = {'userId':int, 'movieId':int, 'rating':float})
s_df = s_df.subtract([1,1,0], axis='columns')   # for zero based indexing
s_matrix = coo_array((s_df['rating'], (s_df['userId'], s_df['movieId'])))
m,n = s_matrix.shape

train, test = train_test_split(s_df, test_size=0.2)
train_matrix = coo_array((train['rating'], (train['userId'], train['movieId'])), shape = (m,n))
test_matrix = coo_array((test['rating'], (test['userId'], test['movieId'])), shape = (m,n))


# reading large data_set
l_df = pd.read_csv(l_data, usecols = ['userId','movieId','rating'], dtype = {'userId':int, 'movieId':int, 'rating':float})
l_df = l_df.subtract([1,1,0], axis='columns')   # for zero based indexing
l_matrix = coo_array((l_df['rating'], (l_df['userId'], l_df['movieId'])))
    


def SVD(k, matrix):
    er = []
    time = []
    space = []
    for i in tqdm(range(*k)):
        tracemalloc.start()
        start=datetime.now()
        U,S,Vt = svds(matrix, which = 'LM' ,k = i)
        t = S.argsort()[::-1]
        s = S[t]
        s = np.diag(s)
        u = U[:,t]
        vt = Vt[t,:]
        vt = s @ vt
        time.append((datetime.now() - start).total_seconds())
        space.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        err = np.square(u @ vt - matrix).sum() / (matrix.size)
        er.append(err)

    fig, ax1 = plt.subplots(1,3)

    ax1[0].plot(np.arange(*k), er)
    ax1[0].xlabel('l_factor(k)')
    ax1[0].ylabel('E')
    ax1[0].title('Error vs K')

    ax1[1].plot(np.arange(*k), time)
    ax1[1].xlabel('K')
    ax1[1].ylabel('Time')
    ax1[1].title('Time vs K')

    ax1[2].plot(np.arange(*k), space)
    ax1[2].xlabel('K')
    ax1[2].ylabel('Space')
    ax1[2].title('Space vs K')

    plt.show()


def cur_method(matrix, k):    
    temp = np.square(matrix)
    e_col = np.sum(temp, axis=0)
    e_row = np.sum(temp, axis=1)
    
    c_prob = e_col / sum(e_col)
    r_prob = e_row / sum(e_row)
    tempcol = np.random.choice(list(range(matrix.shape[1])), k, p = c_prob)
    temprow = np.random.choice(list(range(matrix.shape[0])), k, p = r_prob)

    n_col =  1 / ((c_prob * k)**0.5 + 1e-8)
    n_row =  1 / ((r_prob * k)**0.5 + 1e-8)

    t_matrix = matrix * n_col.reshape(1,matrix.shape[1])
    t_matrix = t_matrix * n_row.reshape(matrix.shape[0], 1)
    t_matrix = t_matrix.tocsr()[temprow,:].tocsc()[:,tempcol]
    return t_matrix.tocoo()


def CUR(k, matrix):
    er = []
    time = []
    space = []
    for i in tqdm(range(*k)):
        tracemalloc.start()
        start=datetime.now()
        mat = cur_method(matrix, i)
        u,s,vt = svds(mat, which = 'LM' ,k = mat.shape[0] - 1)
        t = np.argsort(s)[::-1]
        s = s[t]
        s = np.diag(s)
        u = u[:,t]
        vt = vt[t,:]
        vt = s @ vt
        time.append((datetime.now() - start).total_seconds())
        space.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        err = np.square(u @ vt - mat).sum() / (mat.size)
        er.append(err)
    
    fig, ax1 = plt.subplots(1,3)

    ax1[0].plot(np.arange(*k), er)
    ax1[0].xlabel('l_factor(k)')
    ax1[0].ylabel('E')
    ax1[0].title('Error vs K')

    ax1[1].plot(np.arange(*k), time)
    ax1[1].xlabel('K')
    ax1[1].ylabel('Time')
    ax1[1].title('Time vs K')

    ax1[2].plot(np.arange(*k), space)
    ax1[2].xlabel('K')
    ax1[2].ylabel('Space')
    ax1[2].title('Space vs K')

    plt.show()


# S-gradient Decent...
def s_gradD(train, test, k, lam = 0.1, lr = 1e-3, tolerate = 10):

    global P, Q, train_loss, test_loss

    train_non_zeros = train.size
    test_non_zeros = test.size
    mask_train = coo_array((np.ones(train_non_zeros), train.nonzero()), shape=train.shape).tocsr()
    mask_test = coo_array((np.ones(test_non_zeros), test.nonzero()), shape=train.shape).tocsr()

    train_csr = train.tocsr()
    test_csr = test.tocsr()

    loss = lambda error , nonzeros: np.square(error).sum() / nonzeros
    error_calc = lambda mask, matrx : ((P @ Q) * mask - matrx)

    train_const = 2 / train_non_zeros

    error = error_calc(mask_train, train_csr)
    train_loss.append(loss(error, train_non_zeros))
    test_loss.append(loss(error_calc(mask_test, test_csr), test_non_zeros))
    print(f"starting loss(train) - {train_loss[-1]:0.7f} loss(test) - {test_loss[-1]:0.7f}")

    e = 1
    lowest_error = test_loss[-1]
    bestP = P
    bestQ = Q
    while tolerate:
        error *= train_const
        
        P -= lr * (error @ Q.T + lam * P)
        error = error_calc(mask_train, train_csr) * train_const
        Q -= lr * (P.T @ error + lam * Q)
        error = error_calc(mask_train, train_csr)

        train_loss.append(loss(error, train_non_zeros))
        test_loss.append(loss(error_calc(mask_test, test_csr), test_non_zeros))

        if (e % 200 == 0):
            print(f"epoch-{e} \t loss(train) = {train_loss[-1]:0.7f} \t loss(test) = {test_loss[-1]:0.7f}")
        e += 1
        
        if train_loss[-2] < train_loss[-1]:
            P = bestP
            Q = bestQ
            lr /= 10
            print(f"new lr = {lr}")
        lowest_error = min(lowest_error, test_loss[-1])

        if lowest_error > test_loss[-1]:
            bestP = P
            bestQ = Q
            lowest_error = test_loss[-1]

        if (test_loss[-2] < test_loss[-1]) or ((test_loss[-2] - test_loss[-1]) < 1e-6):
            tolerate -= 1
        
    return bestP, bestQ    


def error_plot(train_loss, test_loss):

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['Training loss', 'Testing loss'])
    plt.xlabel('iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Error vs iter')
    plt.show()

if __name__ == "__main__":

    print('on small dataset...')
    print("SVD...")
    SVD((100,600,20), s_matrix)

    print("CUR...")
    CUR((100,600,20), s_matrix)

    print('on large dataset...')
    print("CUR...")
    CUR((100,5600,500), l_matrix)

    k = 250
    P = np.random.rand(s_matrix.shape[0], k)
    Q = np.random.rand(k, s_matrix.shape[1])
    train_loss = []
    test_loss = []

    print("GRADIENT_DISSENT...")
    P, Q = s_gradD(train_matrix, test_matrix, k, lam = 0.001,lr = 0.1, tolerate = 2)
    error_plot(train_loss, test_loss)
    print(f"train_MSE - {train_loss[-1]:0.7f} test_MSE - {test_loss[-1]:0.7f}")



