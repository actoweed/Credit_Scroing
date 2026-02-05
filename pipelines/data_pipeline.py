import pandas as pd
import os 

RAW_PATH = "data/raw/application_train.csv"
PROCESSED_PATH = "data/processed/application_clean.parquet"

drop_columns = ['FLAG_CONT_MOBILE',
 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_9',
 'FLAG_DOCUMENT_17',
 'FLAG_DOCUMENT_15',
 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_4',
 'FLAG_DOCUMENT_21',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_18',
 'FLAG_DOCUMENT_5',
 'FLAG_DOCUMENT_13',
 'FLAG_DOCUMENT_2',
 'FLAG_MOBIL',
 'FLAG_DOCUMENT_19',
 'FLAG_DOCUMENT_14',
 'REG_REGION_NOT_LIVE_REGION',
 'FLAG_DOCUMENT_12',
 'FLAG_DOCUMENT_10',
 'FLAG_DOCUMENT_11',
 'LIVE_REGION_NOT_WORK_REGION']

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Удаляем плохие колонки
    df = df.drop(columns=drop_columns, errors="ignore")
    return df

def process_features(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['SK_ID_CURR', 'TARGET']]
    for col in num_cols:
        df[col] = (df[col].fillna(df[col].median()))

    

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["HAS_CHILDREN"] = (df["CNT_CHILDREN"] > 0).astype(int)
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_MAX"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
    df["EXT_SOURCE_MIN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["EMPLOYMENT_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    cat_cols = df.select_dtypes(include = ['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna('nan').astype(str)

    return df

def save_data(df, path):
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = process_features(df)
    save_data(df, PROCESSED_PATH)
    print(f"Processed data saved to {PROCESSED_PATH}")