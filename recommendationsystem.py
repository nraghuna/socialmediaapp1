import pandas as pd
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import svd

from sklearn.preprocessing import LabelEncoder

def normalize(pred):
    return (pred-pred.min())/(pred.max()-pred.min())

def generate_prediction_df(mat,pt_df,n_factors):
    if not 1 <= n_factors < min(mat.shape):
        raise ValueError("Must be 1 <= n_factors < min(mat.shape)")
    mat = mat.astype(float)
    u,s,v=svds(mat, k = n_factors)
    print(v.shape)
    s=np.diag(s)
    pred_ratings=np.dot(np.dot(u,s),v)
    pred_ratings= np.array(normalize(pred_ratings))
    print(pred_ratings.shape)
    pred_ratings_reshaped = pred_ratings
    print(pred_ratings_reshaped)
    pred_df = pd.DataFrame(
        pred_ratings_reshaped,
        columns=pt_df.columns,
        index=list(pt_df.index)
    ).transpose()
    return pred_df

def recommend_items(pred_df,usr_id,n_recs,mat,df):
    recommend=[]
    recs=''
    if(pred_df[usr_id][1]>0.05 or pred_df[usr_id][1]>0.05):
        recs = df[df['Movies'] == usr_id]['Books']
        recommend.append(recs)
    h=np.argmax(pred_df.loc[:, usr_id].values)
    usr_predss = pred_df.loc[:, usr_id]
    usr_preds = df.iloc[h]
    sorted_items = usr_preds.sort_values(ascending=False)
    recommended_items = sorted_items.head(n_recs)
    usr_pred= pred_df[usr_id].sort_values(ascending=False).reset_index().rename(columns = {usr_id : 'sim'})
    rec_df= usr_pred.sort_values(by='sim',ascending=False).head(n_recs)
    return rec_df,recs

if __name__ == '__main__':
    df = pd.read_csv('customer_data.csv')
    pt_df= df.pivot_table(columns='Gender',index='Movies',values='Books', aggfunc='sum')
    mat=np.array(pt_df.values)
    df2 = pd.DataFrame(mat)
    encoded_df = pd.get_dummies(df2)
    csr = csr_matrix(encoded_df.values, dtype=np.int8)
    l= LabelEncoder()
    a=mat.reshape(-1,1)
    cs= l.fit_transform(a)
    a= cs.reshape(7,2)
    csr2 = csr_matrix(a, dtype=np.int8)
    pred=generate_prediction_df(csr2,pt_df,1)
    h,recs=recommend_items(pred,'Action',5,mat,df)
    print(recs)