import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import LSTMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 添加这个函数用于预测未来的订单数量
def predict_future(model, x_input, days_to_predict):
    future_predictions = []
    for _ in range(days_to_predict):
        # 预测未来一天的订单数量
        predicted_value = model(x_input.unsqueeze(0))
        # 将预测值添加到future_predictions列表中
        future_predictions.append(predicted_value[0][0].detach().cpu().numpy())
        # 将x_input中的第一列（订单数量）替换为预测值
        x_input[:, 0] = torch.cat((x_input[1:, 0], predicted_value))
    return future_predictions

df = pd.read_csv("data/order_train1.csv", encoding="gbk")
df["order_date"]=df["order_date"].apply(pd.to_datetime,format='%Y-%m-%d')
# data.drop('order_date', axis=1, inplace=True)

df = df.set_index('order_date')
df_train = df[['sales_region_code', 'first_cate_code','second_cate_code','item_code','ord_qty']]

pre = pd.read_csv("pred-有.csv", encoding="gbk")
## 查看预测数据
data_pre = pd.DataFrame(pre,columns=['sales_region_code', 'item_code','first_cate_code', 'second_cate_code',])
# groupby(['sales_region_code'])['ord_qty'].sum().sort_values()

#训练集中有4个特征
df_train1 = df_train[['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code']]
df_train1 = df_train1.reset_index(drop=True)

predictions_df = pd.DataFrame(columns=['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', ])

# Loop over all sales_region_code and item_code combinations
for i in range(len(data_pre)):
    sales_region_code, item_code, first_cate_code, second_cate_code = data_pre.iloc[i, :]
    # print(sales_region_code, item_code, first_cate_code, second_cate_code)
    if len(df_train[(df_train['sales_region_code'] == sales_region_code) & (df_train['item_code'] == item_code) & (df_train['first_cate_code'] == first_cate_code) & (df_train['second_cate_code'] == second_cate_code)]) > 0:
        filtered_df = df_train[(df_train['sales_region_code'] == sales_region_code) & (df_train['item_code'] == item_code) & (df_train['first_cate_code'] == first_cate_code) & (df_train['second_cate_code'] == second_cate_code)]
        # print(filtered_df)
        filtered_df_byday = filtered_df.groupby([pd.Grouper(freq='D')])['ord_qty'].mean().reset_index()
        filtered_df_byday.set_index('order_date',inplace = True)
        filtered_df_byday = filtered_df_byday.fillna(method='ffill')
        # print(filtered_df)
        values = filtered_df_byday['ord_qty'].values.reshape(-1,1)
        # print(values.shape)

        if values.shape[0] > 0:

            # values.to_numpy(values)
            values = values.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            train_size = int(len(scaled) * 0.8)
            test_size = len(scaled) - train_size
            train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
            # print(test.shape)

            if test_size > 1:

                if test_size>30:
                    look_back = 30
                    print('>30', item_code, sales_region_code,first_cate_code, second_cate_code)
                    X_train, y_train = create_dataset(train, look_back)
                    X_test, y_test = create_dataset(test, look_back)

                if 30 >= test_size > 1:
                    print('1-30', item_code, sales_region_code,first_cate_code, second_cate_code)
                    look_back = test_size-1
                    X_train, y_train = create_dataset(train, look_back)
                    X_test, y_test = create_dataset(test, look_back)


                X_train, y_train = create_dataset(train, look_back)
                X_test, y_test = create_dataset(test, look_back)
                # print(X_train.shape)
                # print(X_test.shape)

                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                # Check if there is enough data to make a prediction
                if len(X_test) > 0:

                    input_dim = X_train.shape[2]
                    hidden_dim1 = 48
                    hidden_dim2 = 24
                    output_dim = 1

                    model = LSTMModel(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.15, patience=15, verbose=True)

                    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device))
                    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

                    val_split = 0.1
                    if int(len(train_dataset) * val_split) > 0:
                        val_size = int(len(train_dataset) * val_split)
                    else:
                        val_size = 1
                    train_size = len(train_dataset) - val_size
                    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

                    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

                    epochs = 500
                    early_stop_patience = 15
                    best_val_loss = float('inf')
                    counter = 0

                    for epoch in range(epochs):
                        model.train()
                        for inputs, targets in train_dataloader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for inputs, targets in val_dataloader:
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                                val_loss += loss.item()

                        val_loss /= len(val_dataloader)
                        scheduler.step(val_loss)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            counter = 0
                        else:
                            counter += 1

                        if counter >= early_stop_patience:
                            print("Early stopping")
                            break

                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
                    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).cuda()

                    with torch.no_grad():
                        y_pred = model(X_test_tensor)

                    y_pred_inv = scaler.inverse_transform(y_pred.cpu().numpy())
                    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

                    mse = mean_squared_error(y_test_inv, y_pred_inv)
                    mae = mean_absolute_error(y_test_inv, y_pred_inv)
                    r2 = r2_score(y_test_inv, y_pred_inv)
                    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv))
                    print('均方误差: %.3f' % mse)
                    print('平均绝对误差: %.3f' % mae)
                    print('均方根误差: %.3f' % rmse)
                    print('平均绝对百分比误差:', mape)
                    print('R2分数: %.3f' % r2)

                    """
                    plt.xticks(rotation=45, ha='right')
                    plt.plot(filtered_df_byday.index[-len(y_test):], y_test, label='Actual')
                    plt.plot(filtered_df_byday.index[-len(y_pred):], y_pred, label='Predicted')
                    plt.title('sales_region_code_' + str(int(sales_region_code)) + ' & ' + 'item_code_' + str(int(item_code))+ '\nfirst_cate_code_' + str(int(first_cate_code)) + ' & ' + 'second_cate_code_' + str(int(second_cate_code)))
                    plt.legend()
                    plt.show()
                    """

                    # 预测未来days_to_predict天的订单数量
                    days_to_predict_1_month = 30
                    days_to_predict_2_month = 60
                    days_to_predict_3_month = 90

                    # 提取最后一个时间窗口的输入数据
                    x_input = X_test_tensor[-1]
                    # print(X_test[-1])
                    # print(scaler.inverse_transform(np.array(X_test[-1]).reshape(-1, 1)).flatten())
                    # 使用模型进行未来30天的预测
                    future_predictions_1_month = predict_future(model, x_input, days_to_predict_1_month)
                    future_predictions_2_month = predict_future(model, x_input, days_to_predict_2_month)
                    future_predictions_3_month = predict_future(model, x_input, days_to_predict_3_month)


                    # 将预测结果逆归一化
                    future_sum_1_month = sum(scaler.inverse_transform(np.array(future_predictions_1_month).reshape(-1, 1)).flatten())
                    future_sum_2_month = sum(scaler.inverse_transform(np.array(future_predictions_2_month).reshape(-1, 1)).flatten()) - future_sum_1_month
                    future_sum_3_month = sum(scaler.inverse_transform(np.array(future_predictions_3_month).reshape(-1, 1)).flatten()) - future_sum_2_month - future_sum_1_month


                    # 将销售区域代码、物品代码、一级类别代码、二级类别代码以及未来30天的预测值总和追加到predictions_df
                    predictions_df = predictions_df._append({
                        'sales_region_code': sales_region_code,
                        'item_code': item_code,
                        'first_cate_code': first_cate_code,
                        'second_cate_code': second_cate_code,
                        'mse':mse,
                        'rmse':rmse,
                        'mape':mape,
                        'mae':mae,
                        'r2':r2,
                        'prediction_1_month': future_sum_1_month,
                        'prediction_2_month': future_sum_2_month,
                        'prediction_3_month': future_sum_3_month
                    }, ignore_index=True)
            if test_size == 1:
                # 将销售区域代码、物品代码、一级类别代码、二级类别代码以及未来30天的预测值总和追加到predictions_df
                predictions_df = predictions_df._append({
                    'sales_region_code': sales_region_code,
                    'item_code': item_code,
                    'first_cate_code': first_cate_code,
                    'second_cate_code': second_cate_code,
                    'mse': 0,
                    'rmse':0,
                    'mape':0,
                    'mae':0,
                    'r2':0,
                    'prediction_1_month': test,
                    'prediction_2_month': test,
                    'prediction_3_month': test
                }, ignore_index=True)
    torch.cuda.empty_cache()

# 循环结束后保存predictions_df到新的CSV文件中
predictions_df.to_csv("predictions-有.csv", index=False, encoding='utf-8')