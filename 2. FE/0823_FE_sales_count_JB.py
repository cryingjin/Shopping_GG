# 동일 상품 / 브랜드 총 판매횟수
def sales_count(df) :
    df["상품총판매횟수"] = df.groupby(["NEW상품명"])["방송일시"].transform('size')
    df["브랜드총판매횟수"] = df.groupby(["브랜드"])["방송일시"].transform('size')
    return df