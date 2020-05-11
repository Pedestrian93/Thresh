import pandas as pd
import baostock as bs


def get_all_stock_code(date):
    """
    fetch all the stock in the market in specific date
    :param date:
    :return:
    """
    stock_rs = bs.query_all_stock(date)
    stock_df = stock_rs.get_data()
    stock_df = stock_df[stock_df.tradeStatus == '1'].reset_index(drop=True)
    return stock_df


def get_stock_data(stock_codes, start_date, end_date, fields, save_path):
    # 获取指定日期的指数、股票数据
    data_df = pd.DataFrame()
    for i, code in enumerate(stock_codes):
        if i % 500 == 0:
            print("Downloading :" + code)
        k_rs = bs.query_history_k_data_plus(code, fields, start_date, end_date)
        data_df = data_df.append(k_rs.get_data())
    data_df.to_csv(save_path, encoding="gbk", index=False)
    return data_df


def preprocess_data(data):
    # type transformation
    int_type_cols = {col: 'int8' for col in ['adjustflag', 'tradestatus', 'isST']}
    float_type_cols = {col: 'float32' for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'peTTM','psTTM', 'pcfNcfTTM', 'pbMRQ']}
    type_dict = {**int_type_cols, **float_type_cols}
    data['turn'] = data['turn'].apply(lambda x: 0 if x=='' else float(x))
    data = data.astype(type_dict)
    return data


def create_statistic_for_data(data, stat_window_size_list = [1, 2, 3, 5, 7, 14, 20, 30, 60, 90, 300]):
    data['mid_price'] = (data['high'] + data['low']) / 2
    for price_col in ['close', 'mid_price']:
        for k in stat_window_size_list:
            # 1. increasing rate in last k days
            data[f'inc_rate_{k}'] = (data[f'{price_col}'] - data[f'{price_col}'].shift(-k)) / data[f'{price_col}'].shift(-k)
            if k != 1:
                # 2. rolling mean of price in window size k
                data[f'ma_{k}_{price_col}'] = data[price_col][::-1].rolling(k).mean().sort_index() # calculate the moving average of the past, not the future, so reverse the data first
    return data


def increasing_rate_strategy(data, date, s_name):
    """
    strategy of choosing stock
    :param data:
    :param date:
    :param s_name:
    :return:
    """
    used_data = data[data['date'] == date]
    # 0. not ST
    used_data = used_data[used_data.isST==0]
    # 1. increasing rate limit
    inc_rate_limit = (used_data['inc_rate_3_close'] <= -0.05) & \
                    (used_data['inc_rate_7_close'] <= 0) & \
                    (used_data['inc_rate_14_close'] > 0) & \
                    (used_data['inc_rate_20_close'] > 0.05) & \
                    (used_data['inc_rate_30_close'] > 0.08) & \
                    (used_data['inc_rate_60_close'] > 0.15)
    chosen_data = used_data[inc_rate_limit]
    # 2. volume limit
    chosen_data['strategy_name'] = s_name
    return chosen_data


def apply_strategy(strategies, data):
    """
    apply several strategies and then fetch the intersection of them
    :param strategies:
    :param data:
    :return: chosen_stock
    """
    df = pd.DataFrame()
    for strategy in strategies:
        tmp = strategy(data)
        print(f"executing {tmp['strategy_name'][0]}")
        df = df.append(tmp)
    chosen_stock = df.groupby('code')['strategy_name'].apply(lambda x: ','.join(set(x))).reset_index()
    return chosen_stock


def test_result(stocks, amount_map, buy_date, sale_date, profit_col='close'):
    """
    evaluate the performance of the strategy
    :param profit_col:
    :param stocks:
    :param amount_map:
    :param buy_date:
    :param sale_date:
    :return:
    """
    buy_data = pd.DataFrame()
    sale_data = pd.DataFrame()
    for code in stocks:
        buy_tmp = bs.query_history_k_data_plus(code, 'date,code,open,high,low,close', buy_date, buy_date).get_data()
        sale_tmp = bs.query_history_k_data_plus(code, 'date,code,open,high,low,close', sale_date, sale_date).get_data()
        buy_data = buy_data.append(buy_tmp)
        sale_data = sale_data.append(sale_tmp)
    buy_data = buy_data.reset_index(drop=True)
    sale_data = sale_data.reset_index(drop=True)
    buy_data.columns = ['buy_date','code'] + [f'buy_{col}' for col in buy_data.columns if col not in ['date','code']]
    sale_data.columns = ['sale_date','code'] + [f'sale_{col}' for col in sale_data.columns if col not in ['date','code']]
    print(buy_data.columns)
    df = sale_data.merge(buy_data, on='code', how='left')
    df['amount'] = df['code'].map(amount_map)
    # increasing rate
    for col in ['open', 'high', 'low', 'close']:
        for kind in ['buy', 'sale']:
            df[f'{kind}_{col}'] = df[f'{kind}_{col}'].astype('float32')
        df[f'{col}_increasing_rate'] = (df[f'sale_{col}'] - df[f'buy_{col}']) / df[f'buy_{col}'] * 100
    # profit calculated by close price
    df['profit'] = (df[f'sale_{profit_col}'] - df[f'buy_{profit_col}']) * df['amount']
    profit_sum = df['profit'].sum()
    profit_rate = profit_sum / (df[f'buy_{profit_col}'] * df[f'amount'])
    return df, profit_sum, profit_rate * 100

