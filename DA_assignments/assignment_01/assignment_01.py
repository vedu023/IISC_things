
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# data preprocessing...
def data_pre(path):
    data = pd.read_csv(path)
    data = data.loc[data['Innings'] == 1]

    data['Total.Runs'] = data['Innings.Total.Runs']-data['Total.Runs']
    
    data['R.Overs'] = 50 - data['Over']

    data = data[['Total.Runs', 'R.Overs', 'Wickets.in.Hand']]
    
    return data

# model function...
def model(Z0, u, L):
  return Z0 * (1 - np.exp(-L * u / Z0))

# Z0 values for every wicket corresponding overs...
def z_optimize(u, z, L):
    res = []
    
    for i in range(len(u)):
        if u[i] == 'nan':
            z = -1
        res.append(model(z, u[i], L) )
    
    return np.array(res)

# Duckworth Lewis Method...for optimize parameters(curve fitting)...
def DL_Method(path):
    data = data_pre(path)

    Z = np.ones(11)
    L = np.ones(11)
    
    for i in range(11):
        w = data.loc[data['Wickets.in.Hand'] == i]
        u = np.array(w['R.Overs'])
        y = np.array(w['Total.Runs'])

        Z0, _ = curve_fit(z_optimize, u, y)
        Z[i] = Z0[0]
        L[i] = Z[1]
    
    return Z[1:], L[1:] 

# main function for calculating mse error and ploting data...
def main(path):

    df = data_pre(path)
    Z0, L = DL_Method(path)
 
    err = 0
    ec = 0

    print('\n\n\t\t----Duckworth Lewis Method----\n')
    for i in range(10):

        w = df.loc[df['Wickets.in.Hand'] == i+1]
        u = np.array(w['R.Overs'])
        y = np.array(w['Total.Runs']) 

        x = np.linspace(0, 50, 300)
        fx = model(Z0[i], x, L[i])    
        plt.plot(x, fx)
        
        yp = model(Z0[i], u, L[i])    
        err += sum((y - yp)**2)
        ec += len(y) 

        print(f'Z0({i+1}) = {Z0[i]:.4f} \t L = {L[i]:.4f} \t mse = {err/ec:.4f}')
        #print(f'{i+1} &{Z0[i]:.4f} &{L[i]:.4f} &{err/ec:.4f}')

path = './data/04_cricket_1999to2011.csv'

main(path)

# figure...
plt.title("Average Runs obtainable")
plt.xlabel('Overs')
plt.ylabel('Z0')
plt.xlim((0, 50))
plt.xticks([i*5 for i in range(11)])
plt.legend([f'{i}' for i in range(1,11)], loc = 'best')
plt.grid()
plt.show()
