import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


st.header("Yüz Ölçülerinden Cinsiyet Tahmini Test Ekranı")
st.subheader("Yüz Ölçülerini Aşağıya Giriniz:",divider="blue")


gender=pd.read_csv(r"C:\Users\burak\Desktop\4.Sınıf Projeler\Serhat\Vize\Cinsiyet Tahmini\gender_classification_v7.csv")

gender["gender"]=gender["gender"].apply(lambda x:1 if x=="Male" else 0)

X = gender.drop("gender", axis=1)
y = gender["gender"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Modellerin Kurulması
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

model_svc = SVC(random_state=42)
model_svc.fit(X_train, y_train)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)


uzunSac=st.number_input("Saç Uzunluğunu Girin (Uzun=1 Kısa=0):", min_value=0, max_value=1, step=1)
alinGenislik=st.number_input("Alın Genişliğini Girin:", min_value=11, max_value=16, step=1)
alinUzunluk=st.number_input("Alın Uzunluğunu Girin:", min_value=5, max_value=8, step=1)
burunGenislik=st.number_input("Burun Genişliğini Seçin (Geniş=1 Geniş Olmayan=0):", min_value=0, max_value=1, step=1)
burunUzunluk=st.number_input("Burun Uzunluğunu Seçin (Uzun Burun=1 Kısa Burun=0):", min_value=0, max_value=1, step=1)
dudakIncelik=st.number_input("Dudak İnceliğini Seçin(İnce Dudak=1 Kalın Dudak=0)", min_value=0, max_value=1, step=1)
dudakBurunUzaklık=st.number_input("Dudak Burun Mesafesi Seçin (Mesafe Uzun=1 Mesafe Kısa=0)", min_value=0, max_value=1, step=1)




if st.button("Random Forest ile Sonucu Hesapla"):
    pred_rf = model_rf.predict([[uzunSac, alinGenislik, alinUzunluk, burunGenislik, burunUzunluk,dudakIncelik,dudakBurunUzaklık]])
    if pred_rf == 0:
        st.success("Teste Göre Cinsiyetiniz Kadın")
    elif pred_rf == 1:
        st.info("Teste Göre Cinsiyetiniz Erkek")


if st.button("Support Vector Machine ile Sonucu Hesapla"):
    pred_svc = model_svc.predict([[uzunSac, alinGenislik, alinUzunluk, burunGenislik, burunUzunluk,dudakIncelik,dudakBurunUzaklık]])
    if pred_svc == 0:
        st.success("Teste Göre Cinsiyetiniz Kadın")
    elif pred_svc == 1:
        st.info("Teste Göre Cinsiyetiniz Erkek")


if st.button("K-Nearest Neighbors ile Sonucu Hesapla"):
    pred_knn = model_knn.predict([[uzunSac, alinGenislik, alinUzunluk, burunGenislik, burunUzunluk,dudakIncelik,dudakBurunUzaklık]])
    if pred_knn == 0:
        st.success("Teste Göre Cinsiyetiniz Kadın")
    elif pred_knn == 1:
        st.info("Teste Göre Cinsiyetiniz Erkek")


if st.button("Decision Tree ile Sonucu Hesapla"):
    pred_dt = model_dt.predict([[uzunSac, alinGenislik, alinUzunluk, burunGenislik, burunUzunluk,dudakIncelik,dudakBurunUzaklık]])
    if pred_dt == 0:
        st.success("Teste Göre Cinsiyetiniz Kadın")
    elif pred_dt == 1:
        st.info("Teste Göre Cinsiyetiniz Erkek")

