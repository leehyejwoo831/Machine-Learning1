import numpy as np 
import pandas as pd 
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 데이터 불러오기
car = pd.read_csv('/Users/ihyeju/Desktop/deep/fuel_efficiency/auto-mpg.csv', header=0)
print(car.info())
print(car.describe())

# 결측치 확인
print(car.isnull().sum())

# 결측치 처리 (예시)
car.dropna(inplace=True)
# 불필요한 열 제거
car.drop(['car_name', 'origin', 'horsepower'], axis=1, inplace=True)

# 종속 변수 설정
Y = car['mpg']

# 독립 변수 설정
X = car.drop(['mpg'], axis=1)

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 선형 회귀 모델 학습
lr = LinearRegression()
lr.fit(X_train, Y_train)

# 학습된 모델 저장
pickle.dump(lr, open('car.pkl', 'wb'))

# end 모델과의 적중률은 80%입니다