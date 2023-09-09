 
import numpy as np
import pandas as pd 
import math as m
import matplotlib.pyplot as plt


# data loding...
data = pd.read_csv('./COVID19_data.csv') 
data['Confirmed'] = data['Confirmed'].diff()
data['Tested'] = data['Tested'].diff()
data['First Dose Administered'] = data['First Dose Administered'].diff()
 
Confirm = np.array(data[(data['Date']>='2021-03-09') & (data['Date']<='2021-09-20')]['Confirmed']) 
Tested = np.array(data[(data['Date']>='2021-03-09') & (data['Date']<='2021-09-20')]['Tested']) 
Vaccinated = np.array(data[(data['Date']>='2021-03-14') & (data['Date']<='2021-09-20')]['First Dose Administered']) 

# given parameters
alpha = 1/5.8
gamma = 1/5
epsilon = 0.66
N = 70000000

days = 42
S = np.zeros(days)
E = np.zeros(days)
I = np.zeros(days)
R = np.zeros(days)
CIR = np.zeros(days)
e = np.zeros(days)


# sol for Q-2...
def avg_param(params, V, T, days, N = N, a=alpha, g=gamma, esp=epsilon, waning = True):

    B, S[0], E[0], I[0], R[0], CIR[0] = params

    for day in range(days - 1):
        if day <= 30:
            delW = R[0] / 30
        elif day >= 180:
            if waning == True:
                delW = R[day - 180] + esp * V[day - 180][1]
            else:
                delW = 0
        else:
            delW = 0

        S[day + 1] = S[day] - B * S[day] * I[day] / N - esp * V[day] + delW
        E[day + 1] = E[day] + B * S[day] * I[day] / N - a * E[day]
        I[day + 1] = I[day] + a * E[day] - g * I[day]
        R[day + 1] = R[day] + g * I[day] + esp * V[day] - delW

    aS = np.zeros(days)
    aE = np.zeros(days)
    aI = np.zeros(days)
    aR = np.zeros(days)

    for day in range(days):
        c = 0
        for pd in range(day, day - 7, -1):
            if pd >= 0:
                aS[day] += S[pd]
                aE[day] += E[pd]
                aI[day] += I[pd]
                aR[day] += R[pd]
                c += 1
        aS[day] = aS[day] / c
        aE[day] = aE[day] / c
        aI[day] = aI[day] / c
        aR[day] = aR[day] / c

    for day in range(days):
        CIR[day] = CIR[0] * T[0] / T[day]
        e[day] = aE[day] / CIR[day] 

    return aS, aE, aI, aR, e

 # for Q-3 & 4...
ndays = 195
S1 = np.zeros(ndays)
E1 = np.zeros(ndays)
I1 = np.zeros(ndays)
R1 = np.zeros(ndays)
CIR1 = np.zeros(ndays)
e1 = np.zeros(ndays)

def f_avg_param(P, V, T, ndays, N=N, a=alpha, g=gamma, esp=epsilon, waning = True, closed_loop = False):

    B, S1[0], E1[0], I1[0], R1[0], CIR1[0] = P

    n_cases = []

    for d in range(ndays - 1):
        if closed_loop == True:

            if d % 7 == 1 and d >= 7:
                aCases = 0
                for i in range(7):
                    CIR1[d] = CIR1[0] * T[0] / T[d - i] 
                    aCases += a * (E1[d - i]) / CIR1[d] 
                aCases /= 7

                if aCases < 10000:
                    B = B 
                elif aCases < 25000:
                    B = B * 2 / 3
                elif aCases < 100000:
                    B = B / 2 
                else:
                    B = B / 3

        if d <= 30:
            delW = R1[0] / 30
        elif d >= 180:
            if waning == True:
                delW = R1[d - 180] + esp * V[d - 180]
            else:
                delW = 0
        else:
            delW = 0

        S1[d + 1] = S1[d] - B * S1[d] * I1[d] / N - esp * V[d] + delW
        E1[d + 1] = E1[d] + B * S1[d] * I1[d] / N - a * E1[d]
        I1[d + 1] = I1[d] + a * E1[d] - g * I1[d]
        R1[d + 1] = R1[d] + g * I1[d] + esp * V[d] - delW

        CIR1[d] = CIR1[0] * T[0] / T[d] 
        n_cases.append(a * E1[d])
 
    aS = np.zeros(ndays)
    aE = np.zeros(ndays)
    aI = np.zeros(ndays)
    aR = np.zeros(ndays)

    for d in range(ndays):
        c = 0
        for prev_d in range(d, d - 7, -1):
            if prev_d >= 0:
                aS[d] += S1[prev_d]
                aE[d] += E1[prev_d]
                aI[d] += I1[prev_d]
                aR[d] += R1[prev_d]
                c += 1
        aS[d] = aS[d] / c
        aE[d] = aE[d] / c
        aI[d] = aI[d] / c
        aR[d] = aR[d] / c

    for d in range(ndays):
        CIR1[d] = CIR1[0] * T[0] / T[d] 
        e1[d] = aE[d] / CIR1[d]

    return aS, aE, aI, aR, e1, n_cases


# loss function...
def lossFun(Param):
     
    e = avg_param(Param, Vaccinated, Tested, days)[4]
    e = alpha * e 
    e_avg = np.zeros(days)
    for day in range(days):
        count = 0
        for j in range(day, day - 7, -1):
            if j >= 0:
                count += 1
                e_avg[day] += e[j]
            else:
                break 
        e_avg[day] /= count 
    
    l = np.zeros(days)
    for i in range(days):
        try:
            l[i] = (m.log(Confirm[i]) - m.log(e_avg[i])) ** 2 
        except:
            l[i] = 0

    loss = np.sum(l)/days

    return loss

 # gradient function...
def deltas(x):
     
    y = x.copy()
    mn = lossFun(x)
    for j1 in [-0.01,0.01]:
        y[0]=x[0]+j1
        for i1 in [-1,1]:
            y[1]=x[1]+i1
            for i2 in [-1,1]:
                y[2]=x[2]+i2
                for i3 in [-1,1]:
                    y[3]=x[3]+i3
                    for i4 in [-1,1]:
                        if x[4] >= R[0] and x[4] <= R[1]:
                            y[4]=x[4]+i4
                        for j2 in [-0.1,0.1]:
                            if x[5] >= CIR[0] and x[4] <= CIR[1]:
                                y[5]=x[5]+j2 
                            l = lossFun(y)
                            if l > mn:
                                res = [j1,i1,i2,i3,i4,j2]
                                mn = l
    return res,mn    


# gradient decent algorithm...
def gradD_algo(params):
     
    [dB, dS, dE, dI, dR, dCIR], l = deltas(params)
    Bk, sk, ek, ik, rk, cirk = params

    k = 0
    th = 1e5
    while(k<th):

        n_Bk=Bk-(1/(k+1))*dB
        n_sk=sk-(1/(k+1))*dS
        n_ek=ek-(1/(k+1))*dE
        n_ik=ik-(1/(k+1))*dI
        n_rk=rk-(1/(k+1))*dR
        n_cirk=cirk-(1/(k+1))*dCIR

        n_P=(n_Bk,n_sk,n_ek,n_ik,n_rk,n_cirk)

        if(l < lossFun(n_P)):
            break

        l=lossFun(n_P)
        Bk,sk,ek,ik,rk,cirk = n_Bk,n_sk,n_ek,n_ik,n_rk,n_cirk

        if k % 200 == 0: 
            print(f'iter: {k},\t loss: {l}')
        k+=1
    
    print(f'final loss : {l:.4f}')

    return Bk,sk,ek,ik,rk,cirk


# Ploting...
def newCases(params, ndays = 188):

    t = params[0]
    beta = [t, (2/3)*t, t/2, t/3]
    plt.figure(figsize = (12, 8))

    n_cases_each_day = f_avg_param(params, Vaccinated, Tested, ndays=ndays, closed_loop=True)[-1]
    plt.plot(n_cases_each_day, label = 'Closed Loop')
    
    for beta in beta:

        params[0] = beta
        n_cases_each_day = f_avg_param(params, Vaccinated, Tested, ndays=ndays)[-1]
        plt.plot(n_cases_each_day, label = f'{beta:.3f} open loop')

    plt.legend(loc = 'best')
    plt.title('Predictions')
    plt.xlabel('Days')
    plt.ylabel('new cases')
    plt.show()


def Susceptible(params, ndays = 188):
    
    t = params[0]
    beta = [t, (2/3)*t, t/2, t/3]
    plt.figure(figsize = (12, 8))

    S = f_avg_param(params, Vaccinated, Tested, ndays=ndays, closed_loop=True)[0]
    plt.plot(S, label = 'Closed Loop')
    
    for beta in beta:

        params[0] = beta
        S = f_avg_param(params, Vaccinated, Tested, ndays=ndays)[0]
        plt.plot(S, label = f'{beta:.3f} open loop')

    plt.legend(loc = 'best')
    plt.title('Predictions')
    plt.xlabel('Days')
    plt.ylabel('Susceptible')
    plt.show()


# main function...
if __name__ == '__main__':

    # optimal initail parameter...
    P1 = [0.41, 6.9e7, 7.7e4, 7.7e4, N*0.20, 13]
    P2 = P1.copy()

    # Result
    print(gradD_algo(P1))
    Susceptible(P1)
    newCases(P2)
    
