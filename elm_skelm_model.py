from scipy.io import loadmat

bs_all = [
    'B0005',
    'B0006',
    'B0007',
    'B0018',
    'B0025',
    'B0026',
    'B0027',
    'B0028',
    'B0029',
    'B0030',
    'B0031',
    'B0032',
    'B0033',
    'B0034',
    'B0036',
    'B0038',
    'B0039',
    'B0040',
    'B0041',
    'B0042',
    'B0043',
    'B0044',
    'B0045',
    'B0046',
    'B0047',
    'B0048',
    'B0049',
    'B0050',
    'B0051',
    'B0052',
    'B0053',
    'B0054',
    'B0055',
    'B0056',
]

bs = [
    'B0005',
    'B0006',
    'B0007',
    'B0018',
    'B0025',
    'B0026',
    'B0027',
    'B0028',
    'B0029',
]

ds = []
for b in bs:
    ds.append(loadmat(f'DATA/{b}.mat'))

types = []
times = []
ambient_temperatures = []
datas = []

for i in range(len(ds)):
    x = ds[i][bs[i]]["cycle"][0][0][0]
    ambient_temperatures.append(x['ambient_temperature'])
    types.append(x['type'])
    times.append(x['time'])
    datas.append(x['data'])

for i in range(len(ds)):
    print(f'Battery: {bs[i]}')
    print(f'Cycles: {datas[i].size}')
    print()


import matplotlib.pyplot as plt
import numpy as np

params = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_charge', 'Voltage_charge']

from pprint import pprint

Cycles = {}
params = ['Temperature_measured', 'Voltage_measured', 'Voltage_load', 'Time']

for i in range(len(bs)):
    Cycles[bs[i]] = {}
    Cycles[bs[i]]['count'] = 168 # This is true for battery B0005, 06, 07
    for param in params:
        Cycles[bs[i]][param] = []
        for j in range(datas[i].size):
            if types[i][j] == 'discharge':
                Cycles[bs[i]][param].append(datas[i][j][param][0][0][0])
        
    cap = []
    for j in range(datas[i].size):
        if types[i][j] == 'discharge':
            cap.append(datas[i][j]['Capacity'][0][0][0][0])
    Cycles[bs[i]]['Capacity'] = np.array(cap)

## CRITICAL TIME POINTS FOR A CYCLE
## We will only these critical points for furthur training

## TEMPERATURE_MEASURED
## => Time at highest temperature

## VOLTAGE_MEASURED
## => Time at lowest Voltage

## VOLTAGE_LOAD
## => First time it drops below 1 volt after 1500 time


def getTemperatureMeasuredCritical(tm, time):
    high = 0
    critical = 0
    for i in range(len(tm)):
        if (tm[i] > high):
            high = tm[i]
            critical = time[i]
    return critical

def getVoltageMeasuredCritical(vm, time):
    low = 1e9
    critical = 0
    for i in range(len(vm)):
        if (vm[i] < low):
            low = vm[i]
            critical = time[i]
    return critical

def getVoltageLoadCritical(vl, time):
    for i in range(len(vl)):
        if (time[i] > 1500 and vl[i] < 1):
            return time[i]
    return -1
# First Cycle
f = getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][0], Cycles[bs[0]]['Time'][0])

# 100th Cycle
m = getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][100], Cycles[bs[0]]['Time'][100])

# Last Cycle
l = getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][167], Cycles[bs[0]]['Time'][167])

print(f'Temperature_Measured Critical points')
print(f'First Cycle:\t{f}')
print(f'100th Cycle:\t{m}')
print(f'Last Cycle:\t{l}')

## Conclusion
## !!BATTERY GET HOT QUICKER as they AGE!!

# First Cycle
f = getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][0], Cycles[bs[0]]['Time'][0])

# 100th Cycle
m = getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][100], Cycles[bs[0]]['Time'][100])

# Last Cycle
l = getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][167], Cycles[bs[0]]['Time'][167])

print(f'Voltage_measured Critical points')
print(f'First Cycle:\t{f}')
print(f'100th Cycle:\t{m}')
print(f'Last Cycle:\t{l}')

## Conclusion
## !!VOLTAGE HOLDS FOR LESS TIME as they AGE!!

# First Cycle
f = getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][0], Cycles[bs[0]]['Time'][0])

# 100th Cycle
m = getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][100], Cycles[bs[0]]['Time'][100])

# Last Cycle
l = getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][167], Cycles[bs[0]]['Time'][167])

print(f'Voltage_load Critical points')
print(f'First Cycle:\t{f}')
print(f'100th Cycle:\t{m}')
print(f'Last Cycle:\t{l}')

## Conclusion
## !!VOLTAGE HOLDS FOR LESS TIME as they AGE!!

temperature_measured = []
voltage_measured = []
voltage_load = []
capacity = Cycles[bs[0]]['Capacity']

for i in range(Cycles[bs[0]]['count']):
    temperature_measured.append(getTemperatureMeasuredCritical(Cycles[bs[0]]['Temperature_measured'][i], Cycles[bs[0]]['Time'][i]))
    voltage_measured.append(getVoltageMeasuredCritical(Cycles[bs[0]]['Voltage_measured'][i], Cycles[bs[0]]['Time'][i]))
    voltage_load.append(getVoltageLoadCritical(Cycles[bs[0]]['Voltage_load'][i], Cycles[bs[0]]['Time'][i]))


X = []
for i in range(Cycles[bs[0]]['count']):
    X.append(np.array([temperature_measured[i], voltage_measured[i], voltage_load[i]]))
X = np.array(X)
y = np.array(capacity)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from skelm import ELMRegressor
estimator = ELMRegressor(alpha = 1e6, n_neurons = 800, ufunc='relu', include_original_features = False)
estimator.fit(X_train, y_train)
prediction = estimator.predict(X_test)
y_pred_train = estimator.predict(X_train)

import elm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

stdsc = StandardScaler()

for i in range(len(y_test)):
    print(f'Real:\t\t{y_test[i]}')
    print(f'Predicted:\t{prediction[i]}')
    print(f'Difference:\t{(y_test[i] - prediction[i])}')
    print()

diff = 0
total = 0
for i in range(len(y_test)):
    diff += abs(y_test[i] - prediction[i])
    total += y_test[i]
    plt.scatter(i,y_test[i],c='red')
    plt.scatter(i,prediction[i],c='blue')

ax=plt.gca()
# adjust the y axis scale.
ax.locator_params('y', nbins=10)
plt.show()

diff /= len(y_test)
total /= len(y_test)
accuracy = ((total - diff) / total) * 100

print(f'Average Difference Between Predicted and Real Capacities: {diff}')
print(f'Accuracy: {accuracy} %')
print()

import math
MSE = np.square(np.subtract(y_test,prediction)).mean() 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error: ")
print(RMSE)
