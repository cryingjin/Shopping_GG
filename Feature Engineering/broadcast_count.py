# 동일 상품 / 브랜드 총 방송횟수 - 먼저 돌리기 !!
def broadcast_count(df) :
    # 동일 상품 / 브랜드 총 방송횟수 - 먼저 돌리기 !!
    item_count = df.dropna(subset = ["노출(분)"]).groupby('NEW상품명').count()['방송일시'].reset_index().rename(columns = {'방송일시' : '상품노출횟수'})
    brand_count = df.dropna(subset = ["노출(분)"]).groupby('브랜드').count()['방송일시'].reset_index().rename(columns = {'방송일시' : '브랜드노출횟수'})

    df = df.merge(item_count, on = 'NEW상품명', how = 'left')
    df = df.merge(brand_count, on = '브랜드', how = 'left')

    df["상품노출횟수"] = df["상품노출횟수"].fillna(method='ffill')
    df["브랜드노출횟수"] = df["브랜드노출횟수"].fillna(0)

    return df