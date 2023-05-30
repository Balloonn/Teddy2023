
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# 指定 EarlyStopping 回调函数
early_stop = EarlyStopping(monitor='val_loss', patience=20)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

predictions_df = pd.DataFrame(columns=['sales_region_code', 'item_code', 'first_cate_code', 'second_cate_code', ])

# Loop over all sales_region_code and item_code combinations
for i in range(len(data_pre)):
    sales_region_code, item_code, first_cate_code, second_cate_code = data_pre.iloc[i, :]
    # print(sales_region_code, item_code, first_cate_code, second_cate_code)
    # print(sales_region_code, item_code, first_cate_code, second_cate_code)
    filtered_df = df_train[(df_train['item_code'] == item_code) & (df_train['first_cate_code'] == first_cate_code) & (df_train['second_cate_code'] == second_cate_code)]

    filtered_df_byday = filtered_df.groupby([pd.Grouper(freq='D')])['ord_qty'].mean().reset_index()
    filtered_df_byday.set_index('order_date',inplace = True)
    filtered_df_byday = filtered_df_byday.fillna(method='ffill')
    # print(filtered_df)
    values = filtered_df_byday['ord_qty'].values.reshape(-1,1)
    # print(values.shape)
    # print(values.shape[0])
    # print(values.shape[1])

    if values.shape[0]>0:

        # values.to_numpy(values)
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        train_size = int(len(scaled) * 0.8)
        test_size = len(scaled) - train_size
        train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
        print(test[0][0])

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
                # 指定 EarlyStopping 回调函数
                early_stop = EarlyStopping(monitor='val_loss', patience=20)
                if len(X_test) > 0:
                    # Create the RNN model
                    model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(32),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1)
                ])
                     # Compile the model
                    model.compile(optimizer=Adam(0.0009), loss='mse')
                     # Train the model
                    history = model.fit(
                        X_train, y_train,
                        epochs=850,
                        batch_size=32,
                        validation_split=0.1,
                        shuffle=False,
                        callbacks=[early_stop]  # 添加 EarlyStopping 回调函数
                    )
                     # Evaluate the model
                    # loss = model.evaluate(X_test, y_test)
                    # print(loss)

                    # Predict the test values using the trained model
                    y_pred = model.predict(X_test)

                    # 将 y_pred 和 y_test 逆归一化
                    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
                    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

                    # 计算 MSE
                    mse = mean_squared_error(y_test_inv, y_pred_inv)
                    # print("MSE: ", mse)
                    tot_mse = tf.reduce_sum(mse)
                    avg_mse = tot_mse / mse.shape[0]
                    print('Test MSE: %.3f' % avg_mse)

                    # 设置x轴标签的格式
                    plt.xticks(rotation=45, ha='right')
                    plt.plot(filtered_df_byday.index[-len(y_test):], y_test, label='Actual')
                    plt.plot(filtered_df_byday.index[-len(y_pred):], y_pred, label='Predicted')
                    plt.title('sales_region_code_' + str(int(sales_region_code)) + ' & ' + 'item_code_' + str(int(item_code))+ '\nfirst_cate_code_' + str(int(first_cate_code)) + ' & ' + 'second_cate_code_' + str(int(second_cate_code)))
                    plt.legend()
                    plt.show()

                    # 预测未来days_to_predict天的订单数量
                    days_to_predict_1_month = 30
                    days_to_predict_2_month = 60
                    days_to_predict_3_month = 90

                    # 添加这个函数用于预测未来的订单数量
                    def predict_future(model, x_input, days_to_predict):
                        future_predictions = []
                        for _ in range(days_to_predict):
                            # 预测未来一天的订单数量
                            predicted_value = model.predict(x_input[np.newaxis, ...])
                            # 将预测值添加到future_predictions列表中
                            future_predictions.append(predicted_value[0][0])
                            # 将x_input中的第一列（订单数量）替换为预测值
                            x_input[:, 0] = np.append(x_input[1:, 0], predicted_value)
                        return future_predictions

                    # 提取最后一个时间窗口的输入数据
                    x_input = X_test[-1]
                    # print(X_test[-1])
                    # print(scaler.inverse_transform(np.array(X_test[-1]).reshape(-1, 1)).flatten())

                    # 使用模型进行未来30天的预测
                    future_predictions_1_month = predict_future(model, x_input, days_to_predict_1_month)
                    print(future_predictions_1_month)
                    future_predictions_2_month = predict_future(model, x_input, days_to_predict_2_month)
                    future_predictions_3_month = predict_future(model, x_input, days_to_predict_3_month)


                    # 将预测结果逆归一化
                    future_sum_1_month = sum(scaler.inverse_transform(np.array(future_predictions_1_month).reshape(-1, 1)).flatten())
                    future_sum_2_month = sum(scaler.inverse_transform(np.array(future_predictions_2_month).reshape(-1, 1)).flatten()) - future_sum_1_month
                    future_sum_3_month = sum(scaler.inverse_transform(np.array(future_predictions_3_month).reshape(-1, 1)).flatten()) - future_sum_2_month - future_sum_1_month


                    # 将销售区域代码、物品代码、一级类别代码、二级类别代码以及未来30天的预测值总和追加到predictions_df
                    predictions_df = predictions_df.append({
                        'sales_region_code': sales_region_code,
                        'item_code': item_code,
                        'first_cate_code': first_cate_code,
                        'second_cate_code': second_cate_code,
                        'mse': avg_mse,
                        'prediction_1_month': future_sum_1_month,
                        'prediction_2_month': future_sum_2_month,
                        'prediction_3_month': future_sum_3_month
                    }, ignore_index=True)
        if test_size == 1:
                # 将销售区域代码、物品代码、一级类别代码、二级类别代码以及未来30天的预测值总和追加到predictions_df
                predictions_df = predictions_df.append({
                    'sales_region_code': sales_region_code,
                    'item_code': item_code,
                    'first_cate_code': first_cate_code,
                    'second_cate_code': second_cate_code,
                    'mse': 0,
                    'prediction_1_month': test[0][0],
                    'prediction_2_month': test[0][0],
                    'prediction_3_month': test[0][0]
                }, ignore_index=True)

        else:
                predictions_df = predictions_df.append({
                    'sales_region_code': sales_region_code,
                    'item_code': item_code,
                    'first_cate_code': first_cate_code,
                    'second_cate_code': second_cate_code,
                    'mse': 0,
                    'prediction_1_month': 0,
                    'prediction_2_month': 0,
                    'prediction_3_month': 0
                }, ignore_index=True)


