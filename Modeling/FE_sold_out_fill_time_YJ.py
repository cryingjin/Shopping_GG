# 0823
# 조기매진 함수 : 시간의 nan 값 채워주기 전에 생성할 것


def sold_out_fill_time(df):
    # 조기매진 feature 생성
    df['조기매진'] = df['노출(분)'].map(lambda x: 1 if ((x < 20) | (x > 20) & (x < 30)) else 0)     # 20분 이하, 20-30 분 사이
    # 방송일시의 nan 값 채워주는 부분
    df["노출(분)"] = df["노출(분)"].fillna(method='ffill')
    return df