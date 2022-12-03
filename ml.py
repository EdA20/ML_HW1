from fastapi import FastAPI, File, UploadFile, Response
from io import BytesIO
import csv
import codecs
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import re

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

df_train = df_train.loc[~df_train.drop('selling_price', axis=1).duplicated(keep='first')]
df_train.reset_index(drop=True, inplace=True)

df_train['mileage'] = df_train['mileage'].apply(lambda x: x.split()[0] if pd.isna(x) == False else x).astype(float)
df_train['engine'] = df_train['engine'].apply(lambda x: x.split()[0] if pd.isna(x) == False else x).astype(float)
s = df_train['max_power'].apply(lambda x: re.findall('\d+.\d+|\d+', x)[0] if pd.isna(x) == False and len(x.split()) > 1 else x)
df_train['max_power'] = pd.to_numeric(s, errors = 'coerce')
df_train = df_train.drop('torque', axis=1)

fill_values = df_train.describe().loc['50%', 'mileage':'seats']
df_train = df_train.fillna(value=fill_values)

df_train['engine'] = df_train['engine'].astype(int)
df_train['seats'] = df_train['seats'].astype(int)

ss = StandardScaler()
y_train = df_train['selling_price'].values
X_train_cat = df_train.drop(['selling_price', 'name'], axis=1)
X_train_cat['seats'] = X_train_cat['seats'].astype('category')

ohe = OneHotEncoder(sparse=False, drop='first')
data = ohe.fit_transform(X_train_cat.select_dtypes(exclude='number'))
feature_names = ohe.get_feature_names_out(X_train_cat.select_dtypes(exclude='number').columns)
df_cats = pd.DataFrame(data, columns=feature_names)

df_num = pd.DataFrame(ss.fit_transform(X_train_cat.select_dtypes(include='number')),
                      columns=X_train_cat.select_dtypes(include='number').columns)

X_train_cat = pd.concat([df_num, df_cats], axis=1)

ridge = Ridge(10)
ridge.fit(X_train_cat, y_train)

app = FastAPI()


class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_d = item.dict()
    df = pd.DataFrame(item_d, index=[0])

    df['mileage'] = df['mileage'].apply(lambda x: x.split()[0] if pd.isna(x) == False else x).astype(float)
    df['engine'] = df['engine'].apply(lambda x: x.split()[0] if pd.isna(x) == False else x).astype(float)
    s = df['max_power'].apply(
        lambda x: re.findall('\d+.\d+|\d+', x)[0] if pd.isna(x) == False and len(x.split()) > 1 else x)
    df['max_power'] = pd.to_numeric(s)

    df['seats'] = df['seats'].astype('category')
    df['engine'] = df['engine'].astype(int)

    df_numm = pd.DataFrame(ss.transform(df.select_dtypes(include='number')),
                           columns=df.select_dtypes(include='number').columns)

    df_catss = pd.DataFrame(ohe.transform(df.select_dtypes(exclude='number')),
                            columns=feature_names)

    df = pd.concat([df_numm, df_catss], axis=1)
    pred = ridge.predict(df)
    return pred[0]

# подглядел здесь
# https://stackoverflow.com/questions/74573656/how-to-upload-a-csv-file-using-jinja2-templates-and-fastapi-and-return-it-afte

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> Response():
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer, index_col=0)
    df1 = df.copy()
    buffer.close()
    file.file.close()

    df['mileage'] = df['mileage'].apply(lambda x: x.split()[0] if pd.isna(x) == False else x).astype(float)
    df['engine'] = df['engine'].apply(lambda x: x.split()[0] if pd.isna(x) == False else x).astype(float)
    s = df['max_power'].apply(
        lambda x: re.findall('\d+.\d+|\d+', x)[0] if pd.isna(x) == False and len(x.split()) > 1 else x)
    df['max_power'] = pd.to_numeric(s)

    df['seats'] = df['seats'].astype('category')
    df['engine'] = df['engine'].astype(int)

    df_numm = pd.DataFrame(ss.transform(df.select_dtypes(include='number')),
                           columns=df.select_dtypes(include='number').columns)

    df_catss = pd.DataFrame(ohe.transform(df.select_dtypes(exclude='number')),
                            columns=feature_names)

    df = pd.concat([df_numm, df_catss], axis=1)
    preds = ridge.predict(df)
    df1['predicts'] = preds

    return Response(df1.to_csv(), media_type='text/csv')