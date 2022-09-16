
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
 for filename in filenames:
 print(os.path.join(dirname, filename))


/kaggle/input/tabular-playground-series-jul-2021/sample_submission.csv
/kaggle/input/tabular-playground-series-jul-2021/train.csv
/kaggle/input/tabular-playground-series-jul-2021/test.csv

data_train = pd.read_csv("../input/tabular-playground-series-jul2021/train.csv")

data_test =pd.read_csv("../input/tabular-playground-series-jul-2021/test.csv")

data_submit = pd.read_csv("../input/tabular-playground-series-jul2021/sample_submission.csv")

data_train.head(10)

 date_time deg_C relative_humidity absolute_humidity
sensor_1 \
0 2010-03-10 18:00:00 13.1 46.0 0.7578
1387.2
1 2010-03-10 19:00:00 13.2 45.3 0.7255
1279.1
2 2010-03-10 20:00:00 12.6 56.2 0.7502
1331.9
3 2010-03-10 21:00:00 11.0 62.4 0.7867
1321.0
4 2010-03-10 22:00:00 11.9 59.0 0.7888
1272.0
5 2010-03-10 23:00:00 11.2 56.8 0.7848
1220.9
6 2010-03-11 00:00:00 10.7 55.7 0.7603
1244.2
7 2010-03-11 01:00:00 10.3 57.0 0.7702
1181.4
8 2010-03-11 02:00:00 10.1 62.7 0.7648
1159.6
9 2010-03-11 03:00:00 10.5 59.6 0.7517
1030.2
 
sensor_2 sensor_3 sensor_4 sensor_5 target_carbon_monoxide \
0 1087.8 1056.0 1742.8 1293.4 2.5
1 888.2 1197.5 1449.9 1010.9 2.1
2 929.6 1060.2 1586.1 1117.0 2.2
3 929.0 1102.9 1536.5 1263.2 2.2
4 852.7 1180.9 1415.5 1132.2 1.5
5 697.5 1417.2 1462.6 949.0 1.2
6 669.3 1491.2 1413.0 769.6 1.2
7 631.7 1511.1 1359.7 715.4 1.0
8 602.9 1610.6 1212.2 657.2 0.9
9 521.7 1790.2 1148.6 491.0 0.6
 target_benzene target_nitrogen_oxides
0 12.0 167.7
1 9.9 98.9
2 9.2 127.1
3 9.7 177.2
4 6.4 121.8
5 4.4 88.1
6 3.7 59.5
7 3.4 63.9
8 2.2 46.4
9 1.6 43.0

data_test.head(10)
 date_time deg_C relative_humidity absolute_humidity
sensor_1 \
0 2011-01-01 00:00:00 8.0 41.3 0.4375
1108.8
1 2011-01-01 01:00:00 5.1 51.7 0.4564
1249.5
2 2011-01-01 02:00:00 5.8 51.5 0.4689
1102.6
3 2011-01-01 03:00:00 5.0 52.3 0.4693
1139.7
4 2011-01-01 04:00:00 4.5 57.5 0.4650
1022.4
5 2011-01-01 05:00:00 4.5 53.7 0.4759
1004.0
6 2011-01-01 06:00:00 3.3 54.8 0.4636
940.9
7 2011-01-01 07:00:00 3.2 60.7 0.4667
954.5
8 2011-01-01 08:00:00 2.5 65.7 0.4721
969.9
9 2011-01-01 09:00:00 3.9 57.8 0.4807
976.6
 
sensor_2 sensor_3 sensor_4 sensor_5
0 745.7 797.1 880.0 1273.1
1 864.9 687.9 972.8 1714.0
2 878.0 693.7 941.9 1300.8
3 916.2 725.6 1011.0 1283.0
4 838.5 871.5 967.0 1142.3
5 745.5 914.2 989.1 973.8
6 738.2 816.0 896.8 1049.4
7 713.9 834.7 935.6 956.3
8 679.1 943.8 959.3 892.0
9 655.5 996.0 906.0 817.5
data_train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7111 entries, 0 to 7110
Data columns (total 12 columns):
 # Column Non-Null Count Dtype
--- ------ -------------- -----
 0 date_time 7111 non-null object
 1 deg_C 7111 non-null float64
 2 relative_humidity 7111 non-null float64
 3 absolute_humidity 7111 non-null float64
 4 sensor_1 7111 non-null float64
 5 sensor_2 7111 non-null float64
 6 sensor_3 7111 non-null float64
 7 sensor_4 7111 non-null float64
 8 sensor_5 7111 non-null float64
 9 target_carbon_monoxide 7111 non-null float64
 10 target_benzene 7111 non-null float64
 11 target_nitrogen_oxides 7111 non-null float64
dtypes: float64(11), object(1)
memory usage: 666.8+ KB

x = data_train.iloc[:,1:9].values

y=data_train.iloc[:,9:12].values
x

array([[1.3100e+01, 4.6000e+01, 7.5780e-01, ..., 1.0560e+03,
1.7428e+03,
 1.2934e+03],
 [1.3200e+01, 4.5300e+01, 7.2550e-01, ..., 1.1975e+03,
1.4499e+03,
 1.0109e+03],
 [1.2600e+01, 5.6200e+01, 7.5020e-01, ..., 1.0602e+03,
1.5861e+03,
 1.1170e+03],
 ...,
 [9.6000e+00, 3.4600e+01, 4.3100e-01, ..., 8.6190e+02,
8.8920e+02,
 1.1591e+03],
 [8.0000e+00, 4.0700e+01, 4.0850e-01, ..., 9.0850e+02,
9.1700e+02,
 1.2063e+03],
 [8.0000e+00, 4.1300e+01, 4.3750e-01, ..., 7.9710e+02,
8.8000e+02,
 1.2731e+03]])


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

LinearRegression()

y_pred=lr.predict(x_test)
y_pred
array([[ 1.46692854, 8.5647448 , 76.99192769],
 [ 2.30979243, 14.77032555, 192.48110992],
 [ 0.92341324, 2.45323646, 208.66852616],
 ...,
 [ 1.93705476, -0.81555268, 220.65041195],
 [ 2.03063841, 10.92056782, 163.09029882],
 [ 1.51211348, 10.13751535, 65.7380882 ]])

from sklearn.metrics import r2_score

accuracy = r2_score(y_test,y_pred)
print(accuracy)
0.8179162192756918

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)
65.62766719588804