# MAE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
# MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
# RMSE: https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python
# Huber Loss: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.huber.html
# MAPE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error as MAE, MSE,MAPE
from scipy.special import huber
MAE(y_train, y_test)
MSE(y_train, y_test)
mean_squared_error(y_train, y_test,squared=False)
MAPE(y_train, y_test)