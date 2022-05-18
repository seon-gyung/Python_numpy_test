# 파이썬 라이브러리 설치
# python -m pip install numpy
# python -m pip install pandas
# python -m pip install matplotlib
# python -m pip install scikit-learn

from turtle import shapesize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

# csv 파일 불러오기
bream_length = pd.read_csv('C:/Users/82109/Downloads/csv파일/bream_length.csv')
bream_weight = pd.read_csv('C:/Users/82109/Downloads/csv파일/bream_weight.csv')
smelt_length = pd.read_csv('C:/Users/82109/Downloads/csv파일/smelt_length.csv')
smelt_weight = pd.read_csv('C:/Users/82109/Downloads/csv파일/smelt_weight.csv')
# print(smelt_length)

# 도미와 빙어 데이터 시각화
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 도미와 빙어 데이터 2차원 -> 1차원 리스트 변환
bream_length_np = np.array(bream_length).flatten().tolist()
smelt_length_np = np.array(smelt_length).flatten().tolist()
bream_weight_np = np.array(bream_weight).flatten().tolist()
smelt_weight_np = np.array(smelt_weight).flatten().tolist()

# 도미와 빙어 length, weight 합치기
length = bream_length_np + smelt_length_np
weight = bream_weight_np + smelt_weight_np

# print(length)
# print(weight)

# 도미와 빙어 합치기
fish_data = np.column_stack((length, weight))
fish_target = [1]*35 + [0]*14
# print(fish_data)
# print(fish_data[4])

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
# print(input_arr)
# print(target_arr)

# print(input_arr.shape) # (47,2) shape() 배열의 크기를 알려준다. (샘플 수, 특성 수)

# 배열 섞기
np.random.seed(42)
index = np.arange(47) # arrang() N-1까지 1씩 증가하는 배열을 만듬.
np.random.shuffle(index) # shuffle() 주어진 배열을 무작위로 섞음.

# print(index)

# index 배열의 33개를 input_arr와 target_arr에 전달하여 훈련 세트로 만들기
train_input = input_arr[index[:33]]
train_target = target_arr[index[:33]]

# 만들어진 index의 첫 번째 값은 27이다.
# 따라서 train_input의 첫 번째 원소는 input_arr의 28번째 원소가 들어 있다.
# print(input_arr[27], train_input[0])

# 나머지 12개를 테스트 세트로 만들기
test_input = input_arr[index[33:]]
test_target = target_arr[index[33:]]

# 셔플된 훈련 데이터, 테스트 데이터 시각화
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()
# 초록색이 훈련 세트, 빨강색이 테스트 세트

# fit() 모델 훈련하기
kn = kn.fit(train_input, train_target)

# 모델 테스트하기
# print(kn.score(test_input, test_target))

# predict() 예측 결과 [1 0 1 0 1 1 1 0 1 1 0 1 1 0]
# print(kn.predict(test_input))

# 실제 타겟 결과 [1 0 1 0 1 1 1 0 1 1 0 1 1 0]
# print(test_target)
