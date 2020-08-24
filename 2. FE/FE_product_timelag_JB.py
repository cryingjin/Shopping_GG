# 동일 상품 / 브랜드 총 방송횟수 - 먼저 돌리기 !!
## 판매횟수만 따지기 위해서는 원래 데이터를 봐야 하고,
## 방송횟수를 따지기 위해서는 원래 데이터에서 '노출(분)' NaN값을 drop한 dataframe을 봐야 함 !
import pandas as pd

def broadcast_count(df) :
    
    # 동일 상품 / 브랜드 총 방송횟수 - 먼저 돌리기 !!
    item_count = df.dropna(subset = ["노출(분)"]).groupby('NEW상품명').count()['방송일시'].reset_index().rename(columns = {'방송일시' : '상품노출횟수'})
    brand_count = df.dropna(subset = ["노출(분)"]).groupby('브랜드').count()['방송일시'].reset_index().rename(columns = {'방송일시' : '브랜드노출횟수'})
    df = df.merge(item_count, on = 'NEW상품명', how = 'left')
    df = df.merge(brand_count, on = '브랜드', how = 'left')

    # 동일 상품 별 시간 간격
    df["동일상품시간차"] = df.groupby(["NEW상품명"])["방송일시"].diff()
    
    # 동일 상품 / 브랜드 총 판매횟수
    df["상품총판매횟수"] = df.groupby(["NEW상품명"])["방송일시"].transform('size')
    df["브랜드총판매횟수"] = df.groupby(["브랜드"])["방송일시"].transform('size')
    return df



def product_timelag(df) :
    df["동일상품시간차"] = df.groupby(["NEW상품명"])["방송일시"].diff()
    return df

# 동일 상품 / 브랜드 총 판매횟수
def sales_count(df) :
    df["상품총판매횟수"] = df.groupby(["NEW상품명"])["방송일시"].transform('size')
    df["브랜드총판매횟수"] = df.groupby(["브랜드"])["방송일시"].transform('size')
    return df