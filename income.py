import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

st.title("Income Prediction with Machine Learning")
df = pd.read_csv("adult.csv")
st.write(df.head())
df.drop_duplicates(inplace=True)
df["workclass"]=df["workclass"].replace("?",np.nan)
df["occupation"]=df["occupation"].replace("?",np.nan)
df["native.country"]=df["native.country"].replace("?",np.nan)
df.dropna(inplace=True)
df["income"]=df["income"].apply(lambda x:0 if x=="<=50K" else 1)
scaler=MinMaxScaler((0,100))
df[["capital_gain_norm"]]=scaler.fit_transform(df[["capital.gain"]])
df[["capital_loss_norm"]]=scaler.fit_transform(df[["capital.loss"]])

selected_columns_heatmap=['age','fnlwgt','education.num','hours.per.week','capital.gain', 'capital.loss','income']
sns.heatmap(df[selected_columns_heatmap].corr(),annot=True,cmap='coolwarm')
st.pyplot(plt)

label_encoder_workclass=LabelEncoder()
label_encoder_maritalstatus=LabelEncoder()
label_encoder_occupation=LabelEncoder()
label_encoder_relationship=LabelEncoder()
label_encoder_race=LabelEncoder()
label_encoder_sex=LabelEncoder()
label_encoder_nativecountry=LabelEncoder()


df["workclass_enc"]=label_encoder_workclass.fit_transform(df["workclass"])
df['marital.status_enc']=label_encoder_maritalstatus.fit_transform(df['marital.status'])
df['occupation_enc']=label_encoder_occupation.fit_transform(df['occupation'])
df['relationship_enc']=label_encoder_relationship.fit_transform(df['relationship'])
df['race_enc']=label_encoder_race.fit_transform(df['race'])
df['sex_enc']=label_encoder_sex.fit_transform(df['sex'])
df['native.country_enc']=label_encoder_nativecountry.fit_transform(df['native.country'])

X=df[['age','education.num','hours.per.week','capital_gain_norm', 'capital_loss_norm', 'workclass_enc',
       'marital.status_enc', 'occupation_enc', 'relationship_enc', 'race_enc',
       'sex_enc', 'native.country_enc']]
y=df["income"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model_xgb = XGBClassifier(random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb=model_xgb.predict(X_test)

report = classification_report(y_test, y_pred_xgb)
st.text("Classification Report")
st.text(report)

cm=confusion_matrix(y_test,y_pred_xgb)



fig= plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

st.pyplot(fig)



st.sidebar.header("Make Prediction by selecting input features")

workclass=st.sidebar.selectbox("WorkClass",label_encoder_workclass.classes_,index=2)

marital_status = st.sidebar.selectbox('Marital Status', label_encoder_maritalstatus.classes_,index=2)
occupation = st.sidebar.selectbox('Occupation', label_encoder_occupation.classes_,index=3)
relationship = st.sidebar.selectbox('Relationship', label_encoder_relationship.classes_)
race = st.sidebar.selectbox('Race', label_encoder_race.classes_,index=4)
sex = st.sidebar.selectbox('Sex', label_encoder_sex.classes_,index=1)
native_country = st.sidebar.selectbox('Native Country', label_encoder_nativecountry.classes_,index=38)

workclass_enc_st = label_encoder_workclass.transform([workclass])[0]#As array is given to transform method containing one element, to extract integer value [0] is required
marital_status_enc_st = label_encoder_maritalstatus.transform([marital_status])[0]
occupation_enc_st = label_encoder_occupation.transform([occupation])[0]
relationship_enc_st = label_encoder_relationship.transform([relationship])[0]
race_enc_st = label_encoder_race.transform([race])[0]
sex_enc_st = label_encoder_sex.transform([sex])[0]
native_country_enc_st = label_encoder_nativecountry.transform([native_country])[0]

age_st = st.sidebar.number_input('Age', min_value=18, max_value=90, value=40)
education_num_st = st.sidebar.number_input('Education Number', min_value=0, max_value=16, value=14)
hours_per_week_st = st.sidebar.number_input('Hours Per Week', min_value=0, max_value=100, value=40)
capital_gain_st= st.sidebar.number_input('Capital Gain', min_value=0, max_value=1000000, value=0)
capital_loss_st = st.sidebar.number_input('Capital Loss', min_value=0, max_value=1000000, value=0)
capital_gain_norm_st = scaler.transform([[capital_gain_st]])[0][0]
capital_loss_norm_st = scaler.transform([[capital_loss_st]])[0][0]
threshold_settings=st.sidebar.number_input('Threshold for predicting Income >50K',min_value=0.2,max_value=0.9,value=0.40)
X_user_input = np.array([[age_st, education_num_st, hours_per_week_st, capital_gain_norm_st, capital_loss_norm_st, workclass_enc_st, marital_status_enc_st, occupation_enc_st, relationship_enc_st, race_enc_st, sex_enc_st, native_country_enc_st]]) #To convert it into 2D array which is the required format to give as input to predict method
prediction=0
if st.sidebar.button("Predict Income"):
    #prediction=model_xgb.predict(X_user_input)[0]
    y_pred_prob=model_xgb.predict_proba(X_user_input)[:,1]
    y_pred_adjusted=(y_pred_prob>=threshold_settings).astype(int)[0]
    if y_pred_adjusted==0:
        st.write("Predicted Income is less than 50K")
    else:
        st.write("Predicted Income is greater than 50K")


