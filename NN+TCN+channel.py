import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


# Define channel attention module
def channel_attention_block(inputs):
    # the input is(batch_size, time_steps, channels)
    avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)
    pool_outputs = Concatenate(axis=1)([avg_pool, max_pool])

    # Create a small network to learn the weights
    x = Dense(units=inputs.shape[-1] // 2, activation='relu')(pool_outputs)
    x = Dense(units=inputs.shape[-1], activation='sigmoid')(x)

    # Apply weights
    attention_weights = Multiply()([inputs, x])
    return attention_weights


# Define a model for handling one-dimensional vectors
def build_1d_vector_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)  # Prevent overfitting

    outputs = Dense(64, activation='relu')(x)  # Output 64-dimensional features
    return Model(inputs=inputs, outputs=outputs)


# Define a model for processing time series
def build_temporal_sequence_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    x = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(x)
    x = BatchNormalization()(x)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(256, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal')(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.25)(x)  # Prevent overfitting
    x = GlobalAveragePooling1D()(x)  # Convert timing information into fixed-size vectors
    outputs = Dense(64, activation='relu')(x)  # Output 64-dimensional features
    return Model(inputs=inputs, outputs=outputs)



def evalueation(y_test, y_pred):
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate the accuracy. For multi-classification problems, you need to select the average parameter.
    precision = precision_score(y_test, y_pred, average='macro')  # Calculate the accuracy for each category and then take the average
    print(f"Precision: {precision}")

    # To calculate the recall rate, you also need to select the average parameter.
    recall = recall_score(y_test, y_pred, average='macro')  # Calculate recall for each category and then average
    print(f"Recall: {recall}")

    # To calculate the F1 score, you also need to select the average parameter
    f1 = f1_score(y_test, y_pred, average='macro')  # Calculate the F1 score for each category and then take the average
    print(f"F1 Score: {f1}")

    return accuracy,precision, recall, f1


# Open and read json file
with open('/home/nslab/xyc/project/feature/features_dataset_new.json', 'r') as f:
    data = json.load(f)

def Static_attribute_characteristics():

    # Static attribute characteristics
    # Get sample feature data
    X_attaind = []

    # 逐行读取内容
    for prefix, line in data.items():
        X_attaind.append([line['is_same_organization'],line['is_same_country'],line['is_same_rir'],
                  line['Online_announce_rate'],line['hijack_RIR_gini_coefficient'],
                  line['MOAS_prefix_ratio_10'],line['MOAS_AS_num_10'],line['label']])

    # 转换为DataFrame
    df = pd.DataFrame(X_attaind, columns=['is_same_organization', 'is_same_country', 'is_same_rir',
                                  'Online_announce_rate','hijack_RIR_gini_coefficient',
                                  'MOAS_prefix_ratio_10','MOAS_AS_num_10','label'])

    # 删除包含至少一个nan值的行（样本）
    df_clean = df.dropna()

    # 选择列 'label' 并将其转换为列表
    y = df_clean['label'].tolist()
    y = [1 if x == 2 else 2 if x == 3 else 0 for x in y]

    # 去除'label'列
    df_dropped = df_clean.drop(columns=['label'])

    # 将DataFrame的每一行转换为二维向量（即列表）
    X = df_dropped.values.tolist()
    X = np.array(X)
    # y = to_categorical(y)

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)


    # 创建和编译模型
    # 定义输入
    vector_input = Input(shape=(7,))  # 一维向量特征只有7个元素

    # 创建子模型
    vector_model = build_1d_vector_model(vector_input.shape[1:])

    # 获取子模型的输出
    vector_output = vector_model(vector_input)

    # 应用通道注意力
    attention_output = channel_attention_block(vector_output)

    # 分类层
    # x = Dense(256, activation='relu')(attention_output)
    x = Dense(256, activation='relu')(vector_output)
    x = Dense(64, activation='relu')(x)
    features = Dense(32, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(features)  # 三分类输出

    # 创建并返回模型
    model = Model(inputs=vector_input, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy'])

    # 打印模型结构
    model.summary()

    # 早停机制
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # 训练模型
    model.fit(X_train, y_train_nn, epochs=50, batch_size=32, validation_split=0.1)

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test_nn)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


    # 使用训练好的模型提取特征
    feature_extractor = Model(inputs=vector_input, outputs=features)

    # 使用训练好的特征提取模型来提取特征
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # 使用随机森林分类器进行分类
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_features, y_train)

    # 预测测试集
    y_pred = rf.predict(X_test_features)

    accuracy, precision,recall,f1 = evalueation(y_test,y_pred)


    # # y_test是真实标签，y_pred是模型预测结果
    # y_test_int = np.argmax(y_test_nn, axis=1)  # 如果y_test是one-hot编码的
    # y_pred = model.predict(X_test)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    #
    # # 或者你也可以单独计算这些指标
    # precision = precision_score(y_test_int, y_pred_classes, average='weighted')
    # recall = recall_score(y_test_int, y_pred_classes, average='weighted')
    # f1 = f1_score(y_test_int, y_pred_classes, average='weighted')
    #
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1}')


def Dynamic_behavioral_characteristics():
    # 动态行为特征
    # 获取样本特征向量X和标签向量y
    X_attaind = []
    y = []

    # 逐行读取内容
    for prefix, line in data.items():
        X_temp = line['nb_new_A_hijacker'] + line['nb_A_hijacker'] + line['nb_dup_A_hijacker'] + \
                 line['nb_implicit_W_hijacker'] + line['nb_toshorter_hijacker'] + line['nb_A_prefix_hijacker'] + \
                 line['max_path_len_hijacker'] + line['max_editdist_hijacker'] + line['avg_path_len_hijacker'] + \
                 line['max_A_prefix_hijacker'] + line['avg_A_prefix_hijacker'] + line['nb_tolonger_hijacker'] + \
                 line['avg_editdist_hijacker']

        if len(X_temp) == 130:
            X_feature_timesteps = [
                line['nb_new_A_hijacker'] , line['nb_A_hijacker'] , line['nb_dup_A_hijacker'] ,
                line['nb_implicit_W_hijacker'] , line['nb_toshorter_hijacker'] , line['nb_A_prefix_hijacker'] ,
                line['max_path_len_hijacker'] , line['max_editdist_hijacker'] , line['avg_path_len_hijacker'] ,
                line['max_A_prefix_hijacker'] , line['avg_A_prefix_hijacker'] , line['nb_tolonger_hijacker'] ,
                line['avg_editdist_hijacker']
            ]
            # 转换为DataFrame
            df = pd.DataFrame(X_feature_timesteps)
            # 将DataFrame转置
            df_transposed = df.T
            # 按行遍历转置后的DataFrame，并将每一行转换为列表
            row_lists = [list(row) for index, row in df_transposed.iterrows()]
            # 将二维列表转换为NumPy数组
            data_array = np.array(row_lists)
            # 初始化StandardScaler
            scaler = StandardScaler()
            # 对数据进行标准化
            X_timestems_feature = scaler.fit_transform(data_array)

            X_attaind.append(X_timestems_feature)
            y.append(line['label'])

    X = np.array(X_attaind)
    y = [1 if x == 2 else 2 if x == 3 else 0 for x in y]
    # y = to_categorical(y)


    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)


    # 定义输入
    temporal_input = Input(shape=(10, 13))  # 时序数列，10个时间片，每个时间片13个特征

    # 创建子模型
    temporal_model = build_temporal_sequence_model(temporal_input.shape[1:])

    # 获取子模型的输出
    temporal_output = temporal_model(temporal_input)

    # 应用通道注意力
    attention_output = channel_attention_block(temporal_output)

    # 分类层
    x = Dense(64, activation='relu')(attention_output)
    features = Dense(32, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(features)  # 三分类输出

    # 创建并返回模型
    model = Model(inputs=temporal_input, outputs=outputs)

    # 编译模型
    model.compile(optimizer='adam',
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy'])

    # 打印模型结构
    model.summary()

    # 早停机制
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # 训练模型
    model.fit(X_train, y_train_nn, epochs=50, batch_size=32, validation_data=(X_test, y_test_nn))

    # # 评估模型
    # loss, accuracy = model.evaluate(X_test, y_test)
    # print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    #
    # # y_test是真实标签，y_pred是模型预测结果
    # y_test_int = np.argmax(y_test, axis=1)  # 如果y_test是one-hot编码的
    # y_pred = model.predict(X_test)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    #
    # # 或者你也可以单独计算这些指标
    # precision = precision_score(y_test_int, y_pred_classes, average='weighted')
    # recall = recall_score(y_test_int, y_pred_classes, average='weighted')
    # f1 = f1_score(y_test_int, y_pred_classes, average='weighted')
    #
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1}')


    # 使用训练好的模型提取特征
    feature_extractor = Model(inputs=temporal_input, outputs=features)

    # 使用训练好的特征提取模型来提取特征
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # 使用随机森林分类器进行分类
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_features, y_train)

    # 预测测试集
    y_pred = rf.predict(X_test_features)

    accuracy, precision, recall, f1 = evalueation(y_test, y_pred)



# 定义主模型
def build_main_model():
    # 混合特征
    # 获取样本特征向量X和标签向量y
    X_attaind = []
    X_1 = []
    X_2 = []
    y = []

    # 逐行读取内容
    for prefix, line in data.items():
        X_temp = line['nb_new_A_hijacker'] + line['nb_A_hijacker'] + line['nb_dup_A_hijacker'] + \
                 line['nb_implicit_W_hijacker'] + line['nb_toshorter_hijacker'] + line['nb_A_prefix_hijacker'] + \
                 line['max_path_len_hijacker'] + line['max_editdist_hijacker'] + line['avg_path_len_hijacker'] + \
                 line['max_A_prefix_hijacker'] + line['avg_A_prefix_hijacker'] + line['nb_tolonger_hijacker'] + \
                 line['avg_editdist_hijacker']

        if len(X_temp) == 130:
            # 静态属性特征的数据结构
            X_attaind.append([line['is_same_organization'], line['is_same_country'], line['is_same_rir'],
                              line['Online_announce_rate'], line['hijack_RIR_gini_coefficient'],
                              line['MOAS_prefix_ratio_10'], line['MOAS_AS_num_10']])

            # 转换为DataFrame
            df = pd.DataFrame(X_attaind, columns=['is_same_organization', 'is_same_country', 'is_same_rir',
                                                  'Online_announce_rate', 'hijack_RIR_gini_coefficient',
                                                  'MOAS_prefix_ratio_10', 'MOAS_AS_num_10'])

            if df.isnull().any(axis=0).any() == True:
                X_1_line = df.values.tolist()
                X_1.append(X_1_line[0])
                y_line = line['label']
                y.append(y_line)


                # 动态行为特征的数据结构
                X_feature_timesteps = [
                    line['nb_new_A_hijacker'], line['nb_A_hijacker'], line['nb_dup_A_hijacker'],
                    line['nb_implicit_W_hijacker'], line['nb_toshorter_hijacker'], line['nb_A_prefix_hijacker'],
                    line['max_path_len_hijacker'], line['max_editdist_hijacker'], line['avg_path_len_hijacker'],
                    line['max_A_prefix_hijacker'], line['avg_A_prefix_hijacker'], line['nb_tolonger_hijacker'],
                    line['avg_editdist_hijacker']
                ]
                # 转换为DataFrame
                df = pd.DataFrame(X_feature_timesteps)
                # 将DataFrame转置
                df_transposed = df.T
                # 按行遍历转置后的DataFrame，并将每一行转换为列表
                row_lists = [list(row) for index, row in df_transposed.iterrows()]
                # 将二维列表转换为NumPy数组
                data_array = np.array(row_lists)
                # 初始化StandardScaler
                scaler = StandardScaler()
                # 对数据进行标准化
                X_timestems_feature = scaler.fit_transform(data_array)
                X_2_line = X_timestems_feature
                X_2.append(X_2_line)

    X_1 = np.array(X_1)
    X_2 = np.array(X_2)
    y = [1 if x == 2 else 2 if x == 3 else 0 for x in y]
    # y = to_categorical(y)

    # 划分数据集为训练集和测试集
    X_1_train, X_1_test, X_2_train, X_2_test , y_train, y_test = train_test_split(X_1, X_2, y, test_size=0.2, random_state=42)
    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)


    # 输入数据是一个列表，其中包含对应于每个输入的数据
    X_train = [X_1_train, X_2_train]
    X_test = [X_1_test, X_2_test]


    # 定义输入
    vector_input = Input(shape=(7,))  # 假设一维向量特征只有一个元素
    temporal_input = Input(shape=(10, 13))  # 时序数列，10个时间片，每个时间片13个特征

    # 创建子模型
    vector_model = build_1d_vector_model(vector_input.shape[1:])
    temporal_model = build_temporal_sequence_model(temporal_input.shape[1:])

    # 获取子模型的输出
    vector_output = vector_model(vector_input)
    temporal_output = temporal_model(temporal_input)

    # 拼接输出
    concatenated = Concatenate()([vector_output, temporal_output])

    # 应用通道注意力
    attention_output = channel_attention_block(concatenated)

    # 分类层
    x = Dense(64, activation='relu')(attention_output)
    # x = Dense(64, activation='relu')(concatenated)
    features = Dense(32, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(features)  # 三分类输出

    # 创建并返回模型
    model = Model(inputs=[vector_input, temporal_input], outputs=outputs)

    # 编译模型
    model.compile(optimizer='adam',
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy'])

    # 打印模型结构
    model.summary()

    # 早停机制
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # 训练模型
    model.fit(X_train, y_train_nn, epochs=50, batch_size=32, validation_data=(X_test, y_test_nn))

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test_nn)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # # y_test是真实标签，y_pred是模型预测结果
    # y_test_int = np.argmax(y_test, axis=1)  # 如果y_test是one-hot编码的
    # y_pred = model.predict(X_test)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    #
    # # 或者你也可以单独计算这些指标
    # precision = precision_score(y_test_int, y_pred_classes, average='weighted')
    # recall = recall_score(y_test_int, y_pred_classes, average='weighted')
    # f1 = f1_score(y_test_int, y_pred_classes, average='weighted')
    #
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1}')

    # 使用训练好的模型提取特征
    feature_extractor = Model(inputs=[vector_input, temporal_input], outputs=features)

    # 使用训练好的特征提取模型来提取特征
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # 使用随机森林分类器进行分类
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_features, y_train)

    # 预测测试集
    y_pred = rf.predict(X_test_features)

    accuracy, precision, recall, f1 = evalueation(y_test, y_pred)

# 启动相关模型
# Static_attribute_characteristics()
# Dynamic_behavioral_characteristics()
build_main_model()




