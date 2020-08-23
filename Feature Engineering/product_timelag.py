# 동일 상품 별 시간 간격
def product_timelag(df) :
    df["동일상품시간차"] = df.groupby(["NEW상품명"])["방송일시"].diff()
    return df