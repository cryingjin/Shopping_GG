# 동일 상품 / 브랜드 총 방송횟수 - 먼저 돌리기 !!
## 판매횟수만 따지기 위해서는 원래 데이터를 봐야 하고,
## 방송횟수를 따지기 위해서는 원래 데이터에서 '노출(분)' NaN값을 drop한 dataframe을 봐야 함 !

def broadcast_count(df) :
    onair = df.dropna(subset = ["노출(분)"]) # 21525 rows

    onair["상품총방송횟수"] = onair.groupby(["NEW상품명"])["방송일시"].transform('size')
    onair["브랜드총방송횟수"] = onair.groupby(["브랜드"])["방송일시"].transform('size')

    df = pd.merge(df, onair[["NEW상품명", '상품총방송횟수']].drop_duplicates() ,on = "NEW상품명", how="left")
    df = pd.merge(df, onair[["브랜드", '브랜드총방송횟수']].drop_duplicates(), on = "브랜드", how="left")

    return df 