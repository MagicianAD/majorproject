# import library
def get_min_max():
    import pandas as pd
    import numpy as np
    import os
    module_dir = os.path.dirname(__file__)
    file_loc = os.path.join(module_dir,'ff.csv')
    df = pd.read_csv(file_loc)
    df = df[['date', 'AS', 'AW', 'evening', 'morning', 'min', 'max']]
    df['AW'] = pd.to_numeric(df['AW'], errors='coerce')
    x_for_max = df.iloc[:, 1:6].values
    x_for_min = df.iloc[:, [1, 2, 3, 4, 6]].values
    # dependent varibale
    min_ = df.iloc[:, 5].values
    max_ = df.iloc[:, 6].values
    min_ = min_.reshape(-1, 1)
    max_ = max_.reshape(-1, 1)
    # handling missing values
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(x_for_max[:, :])
    x_for_max[:, :] = imputer.transform(x_for_max[:, :])
    imputer = imputer.fit(x_for_min[:, :])
    x_for_min[:, :] = imputer.transform(x_for_min[:, :])
    # handling missing values on the target value
    imputer_min = imputer.fit(min_)
    min_ = imputer_min.transform(min_).ravel()
    imputer_max = imputer.fit(max_)
    max_ = imputer_max.transform(max_).ravel()
    from sklearn.model_selection import train_test_split
    X_train_for_max, X_test_for_max, max_train, max_test = train_test_split(x_for_max, max_, test_size=0.001,
                                                                            random_state=0)
    X_train_for_min, X_test_for_min, min_train, min_test = train_test_split(x_for_min, min_, test_size=0.001,
                                                                            random_state=0)
    from sklearn.linear_model import LinearRegression
    linear_regression_for_max = LinearRegression()
    linear_regression_for_min = LinearRegression()
    linear_regression_for_max.fit(X_train_for_max, max_train)
    linear_regression_for_min.fit(X_train_for_min, min_train)
    import datetime
    date = datetime.date.today()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
    today_month = date.month
    today_day = date.day
    # filter data by day and month
    filter_1 = df[df['month'] == today_month]
    filter_final = filter_1[filter_1['day'] == today_day]
    mean = filter_final.mean(axis=0)
    todays_data_for_max = mean[['AS', 'AW', 'evening', 'morning', 'min']]
    todays_data_for_min = mean[['AS', 'AW', 'evening', 'morning', 'max']]

    f = open('abc.txt','a')

    #filter the data for next 1 day
    one_day = date+ datetime.timedelta(days=1)
    tomorrow_month = one_day.month
    tomorrow_day  = one_day.day
    filter_for_tomorrow = df[df['month'] == tomorrow_month]
    filter_final_tomorrow = filter_for_tomorrow[filter_for_tomorrow['day'] == tomorrow_day]
    mean = filter_final_tomorrow.mean(axis=0)
    tomorrow_data_for_max = mean[['AS', 'AW', 'evening', 'morning', 'min']]
    tomorrow_data_for_min = mean[['AS', 'AW', 'evening', 'morning', 'max']]


    # filter the data for the next 2 day
    two_day = date + datetime.timedelta(days=2)
    the_day_before_tomorrow_month = two_day.month
    the_day_before_tomorrow_day = two_day.day
    filter_for_after_2_day = df[df['month'] == the_day_before_tomorrow_month]
    filter_final_after_2_day = filter_for_after_2_day[filter_for_after_2_day['day'] == the_day_before_tomorrow_day]
    tomorrow_mean = filter_final_after_2_day.mean(axis=0)
    _2nd_day_data_for_max = tomorrow_mean[['AS', 'AW', 'evening', 'morning', 'min']]
    _2nd_day_data_for_min = tomorrow_mean[['AS', 'AW', 'evening', 'morning', 'max']]

    # filter the data for the next 2 day
    three_day = date + datetime.timedelta(days=3)
    _3_day_before_month = three_day.month
    _3_day_before_day= three_day.day
    filter_for_after_3_day = df[df['month'] == _3_day_before_month]
    filter_final_after_3_day = filter_for_after_3_day[filter_for_after_3_day['day'] == _3_day_before_day]
    _3_mean = filter_final_after_3_day.mean(axis=0)
    _3nd_day_data_for_max = _3_mean[['AS', 'AW', 'evening', 'morning', 'min']]
    _3nd_day_data_for_min = _3_mean[['AS', 'AW', 'evening', 'morning', 'max']]

    # filter the data for the next 2 day
    four_day = date + datetime.timedelta(days=4)
    _4_day_before_month = four_day.month
    _4_day_before_day= four_day.day
    filter_for_after_4_day = df[df['month'] == _4_day_before_month]
    filter_final_after_4_day = filter_for_after_4_day[filter_for_after_4_day['day'] == _4_day_before_day]
    _4_mean = filter_final_after_4_day.mean(axis=0)
    _4nd_day_data_for_max = _4_mean[['AS', 'AW', 'evening', 'morning', 'min']]
    _4nd_day_data_for_min = _4_mean[['AS', 'AW', 'evening', 'morning', 'max']]


    return (linear_regression_for_min.predict(np.array([todays_data_for_min])),
            linear_regression_for_max.predict(np.array([todays_data_for_max])),
            linear_regression_for_max.predict(np.array([tomorrow_data_for_max])),
            linear_regression_for_min.predict(np.array([tomorrow_data_for_min])),
            linear_regression_for_max.predict(np.array([_2nd_day_data_for_max])),
            linear_regression_for_min.predict(np.array([_2nd_day_data_for_min])),
            linear_regression_for_max.predict(np.array([_3nd_day_data_for_max])),
            linear_regression_for_min.predict(np.    array([_3nd_day_data_for_min])),
            linear_regression_for_max.predict(np.array([_4nd_day_data_for_max])),
            linear_regression_for_min.predict(np.array([_4nd_day_data_for_min]))
            )


def rain():
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    import os
    module_dir = os.path.dirname(__file__)
    file_loc = os.path.join(module_dir, 'ff.csv')
    df = pd.read_csv(file_loc)
    df = df[['date', 'AS', 'AW', 'min', 'max']]
    df['AW'] = pd.to_numeric(df['AW'], errors='coerce')

    print("==========================")
    xm = df['AS'].mean(axis=0)*3
    mean_AS = df[df.AS != 0.0].median(skipna=True)
    df.loc[df.AS == 0.0, "AS"] = mean_AS
    mean_AW = df[df.AW != 0.0].median(skipna=True)
    df.loc[df.AW == 0.0, "AW"] = mean_AW
    print(df[df['AW']==0.0])
    print("============================")
    x = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    y = y.reshape(-1, 1)
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(x[:, :])
    x[:, :] = imputer.transform(x[:, :])
    # handling missing values on the target value
    #imputer_y = imputer.fit(y)
    #y = imputer_y.transform(y).ravel()
    #split data set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.001, random_state=0)
    from sklearn.linear_model import LinearRegression
    lab_enc = preprocessing.LabelEncoder()
    y_encoded = lab_enc.fit_transform(y_train)
    logisticRegression = LinearRegression()
    logisticRegression.fit(X_train, y_encoded)
    import datetime
    date = datetime.date.today()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
    today_month = date.month
    today_day = date.day
    # filter data by day and month
    filter_1 = df[df['month'] == today_month]
    filter_final = filter_1[filter_1['day'] == today_day]
    mean = filter_final.mean(axis=0)
    todays_data_for_wind = mean[['AW','min', 'max']]

    # filter the data for next 1 day
    one_day = date + datetime.timedelta(days=1)
    tomorrow_month = one_day.month
    tomorrow_day = one_day.day
    filter_for_tomorrow = df[df['month'] == tomorrow_month]
    filter_final_tomorrow = filter_for_tomorrow[filter_for_tomorrow['day'] == tomorrow_day]
    mean = filter_final_tomorrow.mean(axis=0)
    tomorrow_data_for_wind = mean[['AW','min', 'max']]

    # filter the data for the next 2 day
    two_day = date + datetime.timedelta(days=2)
    the_day_before_tomorrow_month = two_day.month
    the_day_before_tomorrow_day = two_day.day
    filter_for_after_2_day = df[df['month'] == the_day_before_tomorrow_month]
    filter_final_after_2_day = filter_for_after_2_day[filter_for_after_2_day['day'] == the_day_before_tomorrow_day]
    tomorrow_mean = filter_final_after_2_day.mean(axis=0)
    _2nd_day_data_for_wind = tomorrow_mean[['AW','min', 'max']]

    # filter the data for the next 3 day
    three_day = date + datetime.timedelta(days=3)
    _3_day_before_month = three_day.month
    _3_day_before_day = three_day.day
    filter_for_after_3_day = df[df['month'] == _3_day_before_month]
    filter_final_after_3_day = filter_for_after_3_day[filter_for_after_3_day['day'] == _3_day_before_day]
    _3_mean = filter_final_after_3_day.mean(axis=0)
    _3nd_day_data_for_wind = _3_mean[['AW','min', 'max']]

    # filter the data for the next 2 day
    four_day = date + datetime.timedelta(days=4)
    _4_day_before_month = four_day.month
    _4_day_before_day = four_day.day
    filter_for_after_4_day = df[df['month'] == _4_day_before_month]
    filter_final_after_4_day = filter_for_after_4_day[filter_for_after_4_day['day'] == _4_day_before_day]
    _4_mean = filter_final_after_4_day.mean(axis=0)
    _4nd_day_data_for_wind = _4_mean[['AW','min', 'max']]

    return (logisticRegression.predict(np.array([todays_data_for_wind]))/xm,
            logisticRegression.predict(np.array([tomorrow_data_for_wind]))/xm,
            logisticRegression.predict(np.array([_2nd_day_data_for_wind]))/xm,
            logisticRegression.predict(np.array([_3nd_day_data_for_wind]))/xm,
            logisticRegression.predict(np.array([_4nd_day_data_for_wind]))/xm,
            )



def windspeed():
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    import os
    module_dir = os.path.dirname(__file__)
    file_loc = os.path.join(module_dir, 'ff.csv')
    df = pd.read_csv(file_loc)
    df = df[['date', 'AW', 'AS', 'evening', 'morning', 'min', 'max']]
    df['AW'] = pd.to_numeric(df['AW'], errors='coerce')
    print("==========================")
    mean_AS = df[df.AS != 0.0].mean()
    df.loc[df.AS == 0.0, "AS"] = mean_AS
    mean_AW = df[df.AW != 0.0].mean()
    df.loc[df.AW == 0.0, "AW"] = mean_AW
    print(df[df['AW'] == 0.0])
    print("============================")
    x = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    y = y.reshape(-1, 1)
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(x[:, :])
    x[:, :] = imputer.transform(x[:, :])
    # handling missing values on the target value
    imputer_y = imputer.fit(y)
    y = imputer_y.transform(y).ravel()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.001, random_state=0)
    from sklearn.linear_model import LinearRegression
    lab_enc = preprocessing.LabelEncoder()
    y_encoded = lab_enc.fit_transform(y_train)
    logisticRegression = LinearRegression()
    logisticRegression.fit(X_train, y_encoded)
    import datetime
    date = datetime.date.today()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
    today_month = date.month
    today_day = date.day
    # filter data by day and month
    filter_1 = df[df['month'] == today_month]
    filter_final = filter_1[filter_1['day'] == today_day]
    mean = filter_final.mean(axis=0)
    todays_data_for_wind = mean[['AW', 'evening', 'morning', 'min', 'max']]

    # filter the data for next 1 day
    one_day = date + datetime.timedelta(days=1)
    tomorrow_month = one_day.month
    tomorrow_day = one_day.day
    filter_for_tomorrow = df[df['month'] == tomorrow_month]
    filter_final_tomorrow = filter_for_tomorrow[filter_for_tomorrow['day'] == tomorrow_day]
    mean = filter_final_tomorrow.mean(axis=0)
    tomorrow_data_for_wind = mean[['AS', 'evening', 'morning', 'min', 'max']]

    # filter the data for the next 2 day
    two_day = date + datetime.timedelta(days=2)
    the_day_before_tomorrow_month = two_day.month
    the_day_before_tomorrow_day = two_day.day
    filter_for_after_2_day = df[df['month'] == the_day_before_tomorrow_month]
    filter_final_after_2_day = filter_for_after_2_day[filter_for_after_2_day['day'] == the_day_before_tomorrow_day]
    tomorrow_mean = filter_final_after_2_day.mean(axis=0)
    _2nd_day_data_for_wind = tomorrow_mean[['AS', 'evening', 'morning', 'min', 'max']]

    # filter the data for the next 3 day
    three_day = date + datetime.timedelta(days=3)
    _3_day_before_month = three_day.month
    _3_day_before_day = three_day.day
    filter_for_after_3_day = df[df['month'] == _3_day_before_month]
    filter_final_after_3_day = filter_for_after_3_day[filter_for_after_3_day['day'] == _3_day_before_day]
    _3_mean = filter_final_after_3_day.mean(axis=0)
    _3nd_day_data_for_wind = _3_mean[['AS', 'evening', 'morning', 'min', 'max']]

    # filter the data for the next 2 day
    four_day = date + datetime.timedelta(days=4)
    _4_day_before_month = four_day.month
    _4_day_before_day = four_day.day
    filter_for_after_4_day = df[df['month'] == _4_day_before_month]
    filter_final_after_4_day = filter_for_after_4_day[filter_for_after_4_day['day'] == _4_day_before_day]
    _4_mean = filter_final_after_4_day.mean(axis=0)
    _4nd_day_data_for_wind = _4_mean[['AS', 'evening', 'morning', 'min', 'max']]

    return (logisticRegression.predict(np.array([todays_data_for_wind])),
            logisticRegression.predict(np.array([tomorrow_data_for_wind])),
            logisticRegression.predict(np.array([_2nd_day_data_for_wind])),
            logisticRegression.predict(np.array([_3nd_day_data_for_wind])),
            logisticRegression.predict(np.array([_4nd_day_data_for_wind])),
            )
