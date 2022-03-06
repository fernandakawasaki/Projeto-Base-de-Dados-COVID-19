import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

categorical_cols = ['intubed', 'pneumonia', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr',
                     'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 
                    'tobacco', 'contact_other_covid', 'covid_res', 'icu', 'patient_type']

diseases_col = ['diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension','other_disease',
                 'cardiovascular', 'obesity', 'renal_chronic']

def pre_process(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # convertendo dados para datetime 
    df['entry_date'] = pd.to_datetime(df['entry_date'], format= '%d-%m-%Y')
    df['date_symptoms'] = pd.to_datetime(df['date_symptoms'], format= '%d-%m-%Y')

    # tratando datas invalidas
    df['date_died'] = df['date_died'].replace({'9999-99-99':np.nan})
    df['date_died'] = pd.to_datetime(df['date_died'], format= '%d-%m-%Y')

    return df

def categorical_process(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Mudando 2's para 0's
    df[categorical_cols[:-1]] = df[categorical_cols[:-1]].replace({2:0})

    # Dropando valores 3 em covid_res
    df = df.drop(df[df['covid_res'] == 3].index)
    
    # Replace colunas com 97 (nao se aplica)
    df[['icu','intubed']] = df[['icu','intubed']].replace({97:0})
    df['pregnancy'] = df['pregnancy'].replace({97:0})

    return df

def imputing_nan(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Imputando valores nan
    df[categorical_cols] = df[categorical_cols].replace({99: np.nan})

    # Usando moda em colunas com pouco NaN's
    imputer = SimpleImputer(strategy='most_frequent')
    df[['intubed', 'pneumonia', 'icu']] = imputer.fit_transform(df[['intubed', 'pneumonia', 'icu']])

    # Usando uma constante para categoria com muito NaN's
    imputer = SimpleImputer(strategy='constant', fill_value=2)
    df[['contact_other_covid']] = imputer.fit_transform(df[['contact_other_covid']])

    return df

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Feature engineering
    df['died'] = np.where(df['date_died'].isna(), 0, 1)
    df['entry_symptoms_time'] = (df['entry_date'] - df['date_symptoms']).dt.days
    df['has_disease'] = np.where(df[df[diseases_col] == 1].any(axis=1), 0, 1)

    return df

def full_processing(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # convertendo dados para datetime 
    df['entry_date'] = pd.to_datetime(df['entry_date'], format= '%d-%m-%Y')
    df['date_symptoms'] = pd.to_datetime(df['date_symptoms'], format= '%d-%m-%Y')

    # tratando datas invalidas
    df['date_died'] = df['date_died'].replace({'9999-99-99':np.nan})
    df['date_died'] = pd.to_datetime(df['date_died'], format= '%d-%m-%Y')
    
    # tratamento dados categoricos
    df[categorical_cols[:-1]] = df[categorical_cols[:-1]].replace({2:0})

    df = df.drop(df[df['covid_res'] == 3].index)
    
    df[['icu','intubed']] = df[['icu','intubed']].replace({97:0})
    df['pregnancy'] = df['pregnancy'].replace({97:0})
    
    # Imputando valores nan
    df[categorical_cols] = df[categorical_cols].replace({99: np.nan})

    imputer = SimpleImputer(strategy='most_frequent')
    df[['intubed', 'pneumonia', 'icu']] = imputer.fit_transform(df[['intubed', 'pneumonia', 'icu']])

    imputer = SimpleImputer(strategy='constant', fill_value=2)
    df[['contact_other_covid']] = imputer.fit_transform(df[['contact_other_covid']])

    # Feature engineering
    df['died'] = np.where(df['date_died'].isna(), 0, 1)
    df['entry_symptoms_time'] = (df['entry_date'] - df['date_symptoms']).dt.days
    
    df['has_disease'] = np.where(df[df[diseases_col] == 1].any(axis=1), 0, 1)

    return df