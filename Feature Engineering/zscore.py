def engineering_zscore(df):
#     brand = df.columns[df.columns.str.contains("상품군&브랜드")]
#     for b in brand:
#         df[b] = df[b].fillna(0)

    def zscore(price, mean, std):
        if std == 0:
            return 0
        else:
            return (price - mean) / std

    df["상품군_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["상품군_평균판매단가"], x["상품군_표준편차"]), axis=1)
    df["상품군&브랜드_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["상품군&브랜드_평균판매단가"], x["상품군&브랜드_표준편차"]), axis=1)
    df["마더코드_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["마더코드_평균판매단가"], x["마더코드_표준편차"]), axis=1)
    df["NEW_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["NEW_평균판매단가"], x["NEW_표준편차"]), axis=1)

    # z-score 만들어 줬으니 평균, 분산, 표준편차 제거!
    stat_mean = df.columns[df.columns.str.contains("_평균판매단가")]
    stat_var = df.columns[df.columns.str.contains("분산")]
    stat_std = df.columns[df.columns.str.contains("표준편차")]

    stats = [stat_mean, stat_var, stat_std]
    for stat in stats:
        df.drop(stat, axis=1, inplace=True)

    return df