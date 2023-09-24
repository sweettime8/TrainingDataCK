import pandas as pd  # doc du lieu
import matplotlib.pyplot as plt  # ve bieu do
import numpy as np  # xu ly du lieu
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler  # chuan hoa du lieu
from keras.callbacks import ModelCheckpoint  # luu lai huan luyen tot nhat
from tensorflow.keras.models import load_model  # tai mo hinh

# Các lớp để xây dựng mô hình
from keras.models import Sequential  # Đầu vào
from keras.layers import LSTM  # Hoc phu thuoc
from keras.layers import Dropout  # Tránh học tủ
from keras.layers import Dense  # Đầu ra

# Kiểm tra độ chính xác mô hình
from sklearn.metrics import r2_score  # Đo mức độ phù hợp
from sklearn.metrics import mean_absolute_error  # Đo sai số tuyệt đối trung bình
from sklearn.metrics import mean_absolute_percentage_error  # Đo phần trăm sai số tuyệt đối trung bình

##### Đọc dữ liệu ############
# from google.colab import drive
# drive.mount('content/drive')
path = "./Dữ liệu Lịch sử HPG.csv"
df = pd.read_csv(path)
print(df)

##### Mô tả dữ liệu
# Định dạng cấu trúc thời gian
df["Ngày"] = pd.to_datetime(df.Ngày, format="%d/%m/%Y")

# Kích thước dữ liệu
df.shape

# Dữ liệu 5 dòng đầu
df.head()

# Xác định kiểu dữ liệu
df.info()

# Mô tả bộ dữ liệu
df.describe()

# convert string to float ( giá)
df['Lần cuối'] = df['Lần cuối'].str.replace(',', '', regex=True)

##### Tiền xử lý dữ liệu
# Lấy dữ liệu
df1 = pd.DataFrame(df, columns=['Ngày', 'Lần cuối'])
df1.index = df1.Ngày
df1.drop("Ngày", axis=1, inplace=True)
df1_copy = df1.copy()
# Lập biểu đồ giá đóng cửa
plt.figure(figsize=(10, 5))
plt.plot(df1['Lần cuối'], label='Giá thực tế', color='red')  # Lap bieu do
plt.title('Biểu đồ giá đóng cửa')
plt.xlabel('Thời gian')  # tên trục x
plt.ylabel('Giá đóng cửa (VND)')  # tên trục y
plt.legend()  # chú thích
plt.show()

# chia tệp dữ liệu
data = df1.values
train_data = data[:400]
test_data = data[400:]

# chuẩn hóa dữ liệu
sc = MinMaxScaler(feature_range=(0, 1))
sc_train = sc.fit_transform(data)

# tạo vòng lặp các giá trị
x_train, y_train = [], []
for i in range(50, len(train_data)):
    x_train.append(sc_train[i - 50:i, 0])  # xtrain bao gom cac mang chua 50 gia dong cua lien tuc
    y_train.append(sc_train[i, 0])  # ytrain gia dong cua cua ngay hom sau tuong ung x train

# xếp dữ liệu thành một mảng
x_train = np.array(x_train)
y_train = np.array(y_train)

# xếp dữ liệu thành mảng một chiều
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

# Bước 5 : xây dựng và huấn luyện mô hình
# xây dựng mô hình
model = Sequential()
model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(LSTM(units=64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')

# Huấn luyện mô hinh
save_model = "save_model.hdf5"
best_model = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, callbacks=[best_model])

# dữ liệu train
y_train = sc.inverse_transform(y_train)  # giá thực tế
final_model = load_model(save_model)
y_train_predict = final_model.predict(x_train)
y_train_predict = sc.inverse_transform(y_train_predict)  # giá dự đoán

### Bước 6 : sử dụng mô hình
# Xử lý dữ liệu test

test = df1[len(train_data) - 50:].values
test = test.reshape(-1, 1)
sc_test = sc.transform(test)

x_test = []
for i in range(50, test.shape[0]):
    x_test.append(sc_test[i - 50:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# dữ liệu test
y_test = data[400:]  # giá thực
y_test_predict = final_model.predict(x_test)
y_test_predict = sc.inverse_transform(y_test_predict)  # giá dự đoán

### Độ chính xác của mô hình
# lập biểu đồ so sánh

train_data1 = df1[50:400]
test_data1 = df1[400:]

plt.figure(figsize=(33, 11))

train_data1['dự đoán1'] = y_train_predict  # thêm dữ liệu
test_data1['dự đoán2'] = y_test_predict  # thêm dữ liệu
df1['dự đoán3']  = df1['Lần cuối']
print(train_data1.columns)
print(test_data1.columns)

merged_data = df1_copy.merge(train_data1[['Lần cuối', 'dự đoán1']], on='Lần cuối', how='left')
merged_data = merged_data.merge(test_data1[['Lần cuối', 'dự đoán2']], on='Lần cuối', how='left')
merged_data = merged_data.merge(df1[['Lần cuối', 'dự đoán3']], on='Lần cuối', how='left')

# plt.plot(train_data1['dự đoán'], label='giá dự đoán train', color='green')  # ĐƯờng giá dự báo train
# plt.plot(test_data1['dự đoán'], label='giá dự đoán test', color='blue')  # ĐƯờng giá dự báo test


plt.plot(merged_data['dự đoán1'], label='giá dự đoán train', color='green')  # ĐƯờng giá dự báo train
plt.plot(merged_data['dự đoán2'], label='giá dự đoán test', color='blue')  # ĐƯờng giá dự báo test
plt.plot(merged_data['dự đoán3'], label='Giá thực tế', color='red')  # đường giá thực

plt.title('So sánh giá đự đoán và giá thực tế')
plt.xlabel('thời gian')  # đặt tên hàm x
plt.ylabel('giá đóng cửa (VND)')
plt.legend()
plt.show()

merged_data.to_csv(f'merged_data.csv', encoding='utf-8', index=True)

print('Độ phù hợp tập train:', r2_score(y_train, y_train_predict))
print('Độ sai số tuyệt đối trung bình tập train:', mean_absolute_error(y_train, y_train_predict))
print('Phần trăm sai số tuyệt đối trung bình tập train:', mean_absolute_percentage_error(y_train, y_train_predict))

print('Độ phù hợp tập test:', r2_score(y_test, y_test_predict))
print('Độ sai số tuyệt đối trung bình tập test:', mean_absolute_error(y_test, y_test_predict))
print('Phần trăm sai số tuyệt đối trung bình tập test:', mean_absolute_percentage_error(y_test, y_test_predict))
