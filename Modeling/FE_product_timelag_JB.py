import pandas as pd

# 동일 상품 / 브랜드 총 방송횟수 - 먼저 돌리기 !!
def broadcast_count(df) :
    item_count = df.dropna(subset = ["노출(분)"]).groupby('NEW상품명').count()['방송일시'].reset_index().rename(columns = {'방송일시' : '상품노출횟수'})
    brand_count = df.dropna(subset = ["노출(분)"]).groupby('브랜드').count()['방송일시'].reset_index().rename(columns = {'방송일시' : '브랜드노출횟수'})

    df = df.merge(item_count, on = 'NEW상품명', how = 'left')
    df = df.merge(brand_count, on = '브랜드', how = 'left')

    df["상품노출횟수"] = df["상품노출횟수"].fillna(method='ffill')
    df["브랜드노출횟수"] = df["브랜드노출횟수"].fillna(0)

    return df


# 동일 상품 / 브랜드 총 판매횟수
def sales_count(df):
    df["상품총판매횟수"] = df.groupby(["NEW상품명"])["방송일시"].transform('size')
    df["브랜드총판매횟수"] = df.groupby(["브랜드"])["방송일시"].transform('size')

    df["브랜드총판매횟수"] = df["브랜드총판매횟수"].fillna(0)

    return df



# 동일상품 시간차
# def product_timelag(df):
#     df["duration"] = df.groupby(["NEW상품명"])["방송일시"].diff()

#     df["duration"] = df["duration"].fillna(0)

#     # days, hours, minutes -> minutes 기준으로 합쳐 주기로 함 !
#     time = df['duration'].dt.components[['days', 'hours', 'minutes']]
#     time["동일상품시간차"] = time['days'] * 60 * 24 + time['hours'] * 60 + time['minutes']

#     df = pd.concat([df, time], axis=1)
#     df = df.drop(['duration', 'days', 'hours', 'minutes'], axis=1)

#     return df