import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses


class Regression_models:
    def __init__(self, x_train, y_train, x_test, y_test, standardlise=True):
        self.features_names_list = x_train.columns.tolist()
        self.targert_names_list = y_train.columns.tolist()
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        if standardlise == True:
            scaler = StandardScaler()
            self.x_train = scaler.fit_transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)
        if (len(self.targert_names_list) == 1):
            # make target data in np.array
            self.y_train = np.ravel(self.y_train)
            self.y_test = np.ravel(self.y_test)

    def bad_learning(self, true, pred):
        ae = np.abs(true - pred)
        threshold = np.percentile(ae, 99)
        # return index where mse>threshold
        outliers_array = np.where(ae > threshold)[0]
        return outliers_array

    def XGBR(self, search=True, best_depth_input=5, best_Nestimators=200, show_avsp=True,
             show_features_imp=False, search_L1_alpha=False, L1_alpha=1):
        if search == True:
            # 2 + i for i in range(tree_max_depth + 1)
            params = {'n_estimators': [100, 150, 200, 400], 'max_depth': [8, 10, 12]}
            # Use GridSearchCV to search over the parameter grid
            grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror'), params, cv=3,
                                       scoring='neg_mean_squared_error')
            grid_search.fit(self.x_train, self.y_train)
            best_parameters = grid_search.best_params_
            print('XGBoost Best Parameters:', best_parameters)
            print('XGBoost Best Score:', -grid_search.best_score_)
        else:
            best_parameters = {'n_estimators': best_Nestimators, 'max_depth': best_depth_input}
        if search_L1_alpha == True:
            alpha_param_grid = {
                'alpha': [0, 0.1, 0.5, 1, 2, 5, 10, 20]
            }
            grid_search2 = GridSearchCV(XGBRegressor(objective='reg:squarederror'), alpha_param_grid, cv=3,
                                        scoring='neg_mean_squared_error')
            grid_search2.fit(self.x_train, self.y_train)
            best_L1_parameters = grid_search2.best_params_
            print('XGBoost Best Parameters:', best_L1_parameters)
            print('XGBoost Best Score:', -grid_search2.best_score_)
            best_parameters['alpha'] = best_L1_parameters['alpha']
        else:
            best_parameters['alpha'] = L1_alpha

        # set L1/L2 regularization based on data characteristic
        # set eval_metric based on data type
        print('Best parameters: ', best_parameters)
        if L1_alpha == None:
            model = XGBRegressor(eval_metric='rmse', n_estimators=best_parameters['n_estimators'],
                                 max_depth=best_parameters['max_depth'])
        else:
            model = XGBRegressor(eval_metric='rmse', n_estimators=best_parameters['n_estimators'],
                                 max_depth=best_parameters['max_depth'], alpha=best_parameters['alpha'])
        model.fit(
            self.x_train,
            self.y_train)
        train_predictions = model.predict(self.x_train)
        test_predictions = model.predict(self.x_test)
        print('XGBR train R_sq: ', model.score(self.x_train, self.y_train))
        print('XGBR train MSE: ', mean_squared_error(self.y_train, train_predictions))
        print('XGBR test R_sq: ', model.score(self.x_test, self.y_test))
        print('XGBR test MSE: ', mean_squared_error(self.y_test, test_predictions))

        train_info_df = pd.DataFrame({'true': list(self.y_train), 'pred': list(train_predictions)})
        test_info_df = pd.DataFrame({'true': list(self.y_test), 'pred': list(test_predictions)})

        if show_avsp:
            plt.figure(figsize=(14, 6))
            plt.scatter(
                self.y_train,
                train_predictions,
                label='train',
                alpha=0.6,
                c='b')
            plt.scatter(
                self.y_test,
                test_predictions,
                label='test',
                alpha=0.6,
                c='r')
            plt.xlabel('actual')
            plt.ylabel('predictions')
            plt.title('XGBoost')
            plt.legend()
            plt.tight_layout()
            plt.show()

        if show_features_imp:
            plt.figure(figsize=(14, 10))
            sns.barplot(
                y=self.features_names_list,
                x=model.feature_importances_)
            plt.title('feature_importance')
            plt.tight_layout()
            plt.show()
        self.xgb_train_info_df = train_info_df
        self.xgb_test_info_df = test_info_df
        return model

    def NN_sequential(self, loss_func='rmse', epoch_num=30, bmc_noise_var=160, huber_delta=1):
        def r_squared(y_true, y_pred):
            residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
            total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
            r2 = 1 - tf.divide(residual, total)
            return r2

        def rmse_loss(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

        def bmc_loss(y_true, y_pred):
            """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
            Args:
              y_pred: A float tensor of size [batch, 1].
              y_true: A float tensor of size [batch, 1].
            Returns:
              loss: A float tensor. Balanced MSE Loss.
            """
            # Calculate the mean
            var_helper = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))
            # noise_var = tf.divide(var_helper, tf.cast(tf.size(y_true) - 1, dtype=tf.float32))
            noise_var = bmc_noise_var
            logits = -tf.divide(tf.pow(y_pred - tf.transpose(y_true), 2), (2 * noise_var))
            labels = tf.range(tf.shape(y_pred)[0])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = loss * (2 * noise_var)
            return loss

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float32)
            diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, 1e8))
            mape = 100. * tf.reduce_mean(diff) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
            return mape

        def huber_loss(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float32)
            huber_delta1 = tf.cast(huber_delta, dtype=tf.float32)
            residual = tf.abs(y_true - y_pred)
            condition = tf.less(residual, huber_delta1)
            small_res = 0.5 * tf.square(residual)
            large_res = huber_delta1 * residual - 0.5 * tf.square(huber_delta1)
            loss = tf.where(condition, small_res, large_res)
            return tf.reduce_mean(loss)

        if loss_func == 'rmse':
            loss_fun = rmse_loss
        elif loss_func == 'bmc':
            loss_fun = bmc_loss
        elif loss_func == 'mape':
            loss_fun = mean_absolute_percentage_error
        elif loss_func == 'huber':
            loss_fun = huber_loss
        else:
            loss_fun = 'mse'
        NN_model = tf.keras.Sequential()
        NN_model.add(layers.Dense(200, activation='relu'))
        NN_model.add(layers.Dense(400, activation='relu'))
        NN_model.add(layers.Dense(200, activation='relu'))
        NN_model.add(layers.Dense(100, activation='relu'))
        NN_model.add(layers.Dense(1, activation='relu'))
        NN_model.compile(optimizer='adam', loss=loss_fun, metrics=[r_squared])
        History = NN_model.fit(self.x_train, self.y_train, epochs=epoch_num,
                               verbose=True, batch_size=300)
        # print(History.history)
        y_predict_tr = NN_model.predict(self.x_train)
        r_sq_tr = r2_score(self.y_train, y_predict_tr)
        mse_tr = mean_squared_error(self.y_train, y_predict_tr)
        print('train R_square: ', r_sq_tr)
        print('train MSE: ', mse_tr)

        y_predict = NN_model.predict(self.x_test)
        r_sq = r2_score(self.y_test, y_predict)
        mse_te = mean_squared_error(self.y_test, y_predict)
        print('test R_square: ', r_sq)
        print('test MSE: ', mse_te)
        train_info_df = pd.DataFrame({'true': list(self.y_train), 'pred': list(y_predict_tr)})
        test_info_df = pd.DataFrame({'true': list(self.y_test), 'pred': list(y_predict)})
        self.nn_train_info_df = train_info_df
        self.nn_test_info_df = test_info_df
        return NN_model


class OutlierDetection_models:
    def __init__(self, X_train, X_test, encoding_dim):
        self.X_train = X_train
        self.X_test = X_test
        self.input_dim = X_train.shape[1]
        self.encoding_dim = encoding_dim

    def isOutlier(self, true, pred, threshold_pct):
        mse = np.mean(np.power(true - pred, 2), axis=1)
        threshold = np.percentile(mse, threshold_pct)
        # return index where mse>threshold
        outliers_array = np.where(mse > threshold)[0]
        return outliers_array

    def auto_encoder(self):
        input_layer = layers.Input(shape=(self.input_dim,))
        encoder = layers.Dense(self.encoding_dim, activation="tanh",
                               activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = layers.Dense(int(self.encoding_dim / 2), activation="relu")(encoder)
        decoder = layers.Dense(int(self.encoding_dim / 2), activation='tanh')(encoder)
        decoder = layers.Dense(self.input_dim, activation='relu')(decoder)
        autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(self.X_train, self.X_train, epochs=50, batch_size=32, shuffle=True, verbose=1)
        train_predictions = autoencoder.predict(self.X_train)
        test_predictions = autoencoder.predict(self.X_test)
        train_outliers = self.isOutlier(self.X_train, train_predictions, 99)
        test_outliers = self.isOutlier(self.X_test, test_predictions, 99)
        # print('Train Outliers', train_outliers)
        # print('Test Outliers', test_outliers)
        self.aec_train_outliers = train_outliers
        self.aec_test_outliers = test_outliers
        return autoencoder

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
train_df = train_df.drop('baseFare', axis=1)
test_df = test_df.drop('baseFare', axis=1)
y_train = train_df[['totalFare']]
x_train = train_df.drop('totalFare', axis=1)
y_test = test_df[['totalFare']]
x_test = test_df.drop('totalFare', axis=1)
Reg_models = Regression_models(x_train, y_train, x_test, y_test, standardlise=True)
# xgb_model = Reg_models.XGBR(search=False, best_depth_input=10, best_Nestimators=200, show_avsp=False,
#                             show_features_imp=False, search_L1_alpha=False, L1_alpha=None)

# seq_NN = Reg_models.NN_sequential(loss_func='huber', epoch_num=100,bmc_noise_var=220,huber_delta=1)

# OD_models = OutlierDetection_models(x_train, x_test, encoding_dim=64)
# autoencoder_model = OD_models.auto_encoder()
