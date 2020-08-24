# 동일 상품 별 시간 간격
def product_timelag(df):
    df["duration"] = df.groupby(["NEW상품명"])["방송일시"].diff()

    df["duration"] = df["duration"].fillna(0)

    # days, hours, minutes -> minutes 기준으로 합쳐 주기로 함 !
    time = df['duration'].dt.components[['days', 'hours', 'minutes']]
    time["동일상품시간차"] = time['days'] * 60 * 24 + time['hours'] * 60 + time['minutes']

    df = pd.concat([df, time], axis=1)
    df = df.drop(['duration', 'days', 'hours', 'minutes'], axis=1)

    return df