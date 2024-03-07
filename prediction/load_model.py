from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
import numpy as np
from pvlib import location
from pvlib import irradiance

def data_process(list):
    df_weather = pd.DataFrame(list)
    df_weather.columns = ['T', 'U', 'Ff', 'RRR', 'DD', 'Po(p)', 'date_time']
    df_weather['date_time'] = pd.to_datetime(df_weather['date_time']).dt.floor('min')
    df_weather[['DD_WE', 'DD_NS']] = df_weather['DD'].apply(DD_to_N).apply(pd.Series)
    df_weather.drop('DD', axis=1, inplace=True)

    tz = 'Asia/Shanghai'
    lat, lon = 27.962847, 120.736522
    site = location.Location(lat, lon, tz=tz)

    # 假设倾斜25度，和朝南的阵列
    irradiance = get_irradiance(site, df_weather['date_time'].iloc[0], df_weather['date_time'].iloc[5], 25, 180)
    irradiance['date_time'] = irradiance['date_time'].dt.tz_localize(None)
    irradiance.reset_index(inplace=True)
    irradiance.drop('index', axis=1, inplace=True)
    df_weather = pd.merge(df_weather, irradiance, on='date_time', how='left')
    df_weather = get_diff(df_weather)
    df_weather = df_weather[1:]
    df_weather['hour'] = df_weather['date_time'].dt.hour
    return df_weather


def get_irradiance(site_location, start, end, tilt, surface_azimuth):
    print('获取光辐照数据中...')
    times = pd.date_range(start=start, end=end, freq='3h', tz=site_location.tz)
    clearsky = site_location.get_clearsky(times)
    solar_position = site_location.get_solarposition(times=times)
    POA_irradiance = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'])

    return pd.DataFrame({'date_time': times, 'POA': POA_irradiance['poa_global']})


def DD_to_N(dd):
    direction = {
        '北': '0/1/',
        '东': '1/0/',
        '南': '0/2/',
        '西': '2/0/'
    }
    if isinstance(dd, str):
        for char, num in direction.items():
            dd = dd.replace(char, num)
        numbers = re.findall('\d+', dd)[:4]
        numbers = [int(num) for num in numbers]
        numbers = [-1 if x == 2 else x for x in numbers]
        new_numbers = [numbers[i:i + 2] for i in range(0, len(numbers), 2)]
        if len(new_numbers) == 0:
            result = [0, 0]
        else:
            result = [0, 0]
            for i in range(len(new_numbers)):
                result = [a + b for a, b in zip(result, new_numbers[i])]
    else:
        result = [0, 0]
    return result

def get_diff(df):
    columns = ['T', 'U', 'Ff', 'RRR', 'DD_WE', 'DD_NS', 'Po(p)', 'POA']
    for col in columns:
        df[f'{col}_b'] = df[col].shift(1)
    return df


def do_predict(df, model):
    features = ['T', 'U', 'Ff', 'RRR', 'DD_WE', 'DD_NS', 'POA', 'Po(p)', 'hour',
                'T_b', 'U_b', 'Ff_b', 'RRR_b', 'DD_WE_b', 'DD_NS_b', 'POA_b', 'Po(p)_b']
    X = df[features]
    y = model.predict(X)
    y[y < 0] = 0
    return y

def post_process(id, results):
    dic = {
        6: 0.959014404167615,
        7: 0.9857749303856512,
        8: 0.9521389470967903,
        9: 1.1665365077382621,
        10: 0.9365352106116819
    }
    coe = dic[id]
    results = [x * coe for x in results]
    # for i in range(1, len(results)):
    #     results[i] += results[i - 1]
    dic_results = {}
    time_list = ["5-8点", "8-11点", "11-14点", "14-17点", "17-20点"]
    for i in range(5):
        dic_results[time_list[i]] = results[i]
    return dic_results

def predict(data,id):
    input_data = np.array(data)
    print(type(input_data))
    if input_data.ndim == 2:
        df = data_process(input_data)
        model = load('best_model.joblib')
        results = do_predict(df, model)
        results = post_process(id,results)
        return results
    else:
        return {'msg':"Invalid input: Data should be a 2D list.", "code":400}


if __name__ == "__main__":
    # 输入：data，id
    # data：[温度（摄氏度），湿度（百分比），风速（米/秒），降水量（毫米），风向，气压（百帕），日期（每天的5，8，11，14，17，20点，%Y-%m-%d %h:00:00）]
    data = [[4.7, 93.0, 3.0, 6.0, '北风', 1018.2, '2024-02-24 05:00:00'],
            [5.5, 89.0, 1.0, 5.0, '从西北偏西方向吹来的风', 1014.64, '2024-02-24 08:00:00'],
            [6.9, 87.0, 1.0, 5.0, '从北方吹来的风', 1013.71, '2024-02-24 11:00:00'],
            [8.6, 80.0, 1.0, 1.0, '从东北方吹来的风', 1011.86, '2024-02-24 14:00:00'],
            [9.0, 69.0, 2.0, 0.5, '从北方吹来的风', 1011.46, '2024-02-24 17:00:00'],
            [7.9, 66.0, 4.0, 0.5, '从北方吹来的风', 1011.99, '2024-02-24 20:00:00']]
    id = 6
    res = predict(data, id)
    print(res)

