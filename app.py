import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# === Page Config ===
st.set_page_config(page_title="Mpox Dashboard", layout="wide")

# === Load Excel Data ===
@st.cache_data
def load_data():
    df = pd.read_excel("mpox_africa_dataset.xlsx", engine="openpyxl", parse_dates=["Report_Date"])
    return df

df= load_data()

# === Sidebar Filters ===
st.sidebar.title("📊 Filter Data")
countries = st.sidebar.multiselect("Select Country", sorted(df["Country"].dropna().unique()), default=sorted(df["Country"].dropna().unique()))
cade = st.sidebar.multiselect("Select Clade", sorted(df["Clade"].dropna().unique()), default=sorted(df["Clade"].dropna().unique()))
ade = st.sidebar.multiselect("Select Month", sorted(df["Report_Date"].dropna().unique()), default=sorted(df["Report_Date"].dropna().unique()))

# === Filtered Data ===
filtered_df = df[df["Country"].isin(countries) & df["Clade"].isin(cade) & df["Report_Date"].isin(ade)]
l= filtered_df.reset_index(drop=True)
# === App Title ===
st.title("🦠 Mpox Outbreak Analysis Dashboard")

st.subheader("📌 Summary Statistics")
st.dataframe(filtered_df.select_dtypes(include=["number"]).describe(), use_container_width=True)

def list_of_list (data_set,reference_column,list_column):
    list_of_seller_id = []
    list_of_seller_ids = []
    list_of_seller_id_index = []

    list_of_seller_list_of_list_column = []
    seller_id_index = 0

    for i in range (0,data_set.shape[0]):
        if pd.isnull(data_set.loc[i,reference_column]) == False :
            list_of_seller_ids.append(data_set.loc[i,reference_column])
        else: continue
        if data_set.loc[i,reference_column] not in list_of_seller_id :
            if pd.isnull(data_set.loc[i,list_column]) == False :
                list_of_seller_list_of_list_column.append([data_set.loc[i,list_column]])
            else:  list_of_seller_list_of_list_column.append([])
        
            list_of_seller_id_index.append(seller_id_index)
            list_of_seller_id.append (data_set.loc[i,reference_column])
            seller_id_index +=1
        else:
            if pd.isnull(data_set.loc[i,list_column]) == False :
                list_of_seller_list_of_list_column[list_of_seller_id.index(data_set.loc[i,reference_column])].append(data_set.loc[i,list_column])
    return list_of_seller_list_of_list_column,list_of_seller_id


cnfd_css , cntr = list_of_list (l,'Country','Confirmed_Cases')
dt = [np.mean(i) for i in list_of_list (l,'Country','Deaths')[0]]
cfdr = [np.mean(i) for i in list_of_list (l,'Country','Case_Fatality_Rate')[0]]
wnc = [np.max(i) for i in list_of_list (l,'Country','Weekly_New_Cases')[0]]
tt = [np.mean(i) for i in list_of_list (l,'Country', 'Testing_Laboratories')[0]]
dc = [np.mean(i) for i in list_of_list (l,'Country','Deployed_CHWs')[0]]
va = [np.mean(i) for i in list_of_list (l,'Country','Vaccinations_Administered')[0]]
vc = [np.mean(i) for i in list_of_list (l,'Country','Vaccine_Coverage')[0]]
ass = [np.mean(i) for i in list_of_list (l,'Country','Active_Surveillance_Sites')[0]]
sc = [np.sum(i) for i in list_of_list (l,'Country','Suspected_Cases')[0]]
vdp = [np.sum(i) for i in list_of_list (l,'Country','Vaccine_Dose_Deployed')[0]]
vac = [np.sum(i) for i in list_of_list (l,'Country','Vaccine_Dose_Allocated')[0]]
vdc = [vdp[i]/vac[i] for i in range (0, len(vdp))]


lp = l.copy()
lp = lp.drop('Country',axis=1)
lp = lp.drop('Report_Date',axis=1)
lp = lp.drop('Clade',axis=1)
lp = lp.drop('Surveillance_Note',axis=1)


st.subheader("📊 Correlation Heatmap")
numeric_cols = filtered_df.select_dtypes(include=["number"])
corr = numeric_cols.corr()
fig4, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig4)

for i in lp.columns: 
    list_,clade = list_of_list (l,'Clade',i)
    list_ = [np.mean(x) for x in list_]
    if lp.columns.to_list().index(i) == 0:
        _table_clade = {'Clade' : clade , i : list_}
        _table_clade = pd.DataFrame (_table_clade)
    else: _table_clade.insert(lp.columns.to_list().index(i) + 1, i , list_)
_table_clade = _table_clade.sort_values(by= 'Confirmed_Cases', ascending= False)


wnc1,rpd = list_of_list (l,'Report_Date','Weekly_New_Cases')
wnc1 = [np.mean(i) for i in wnc1]
vc1 = [np.mean(i) for i in list_of_list (l,'Report_Date','Vaccine_Coverage')[0]]

col = st.sidebar.selectbox("Weekly New Cases vs Vaccine Coverage color" , options = ['Country','Clade','Report_Date'])


_table_month2 = {'Report_Date' : rpd[:-1],  'Weekly_New_Cases': wnc1[:-1], 'Vaccine_Coverage' :vc1[1:]}
st.subheader("Weekly New Cases vs Vaccine Coverage")
fig2 = px.scatter(l, x='Weekly_New_Cases', y='Vaccine_Coverage', color=col, hover_data=['Report_Date','Country'], title=f"Testing Laboratories vs Confirmed by {col.title()}")
st.plotly_chart(fig2, use_container_width=True)


for i in lp.columns: 
    list_,month = list_of_list (l,'Report_Date',i)
    list_ = [np.mean(x) for x in list_]
    if lp.columns.to_list().index(i) == 0:
        _table_month1 = {'Report_Date' : month , i : list_}
        _table_month1 = pd.DataFrame (_table_month1)
    else: _table_month1.insert(lp.columns.to_list().index(i) + 1, i , list_)
_table_month1 = _table_month1.sort_values(by= 'Confirmed_Cases', ascending= False)

cnfd_css = [np.mean(i) for i in cnfd_css]
_table_month = {'Country' : cntr , 'Confirmed_Cases' : cnfd_css , 'Deaths': dt , 'Case_Fatality_Rate' : cfdr ,  'Weekly_New_Cases': wnc, 'Vaccine_Deployed_Ratio' : vdc, 'Testing_Laboratories':tt,'Deployed_CHWs' : dc ,'Active_Surveillance_Sites' : ass,'Suspected_Cases' : sc, 'Vaccinations_Administered':va , 'Vaccine_Coverage' :vc}
_table_month = pd.DataFrame (_table_month)
st.subheader("📈 Choropleth map of Africa")
value_col = st.sidebar.selectbox("Choropleth map info" , options = ['Confirmed_Cases','Deaths','Weekly_New_Cases','Vaccinations_Administered','Vaccine_Coverage'])

fig1 = px.choropleth(_table_month, locations="Country", locationmode="country names", color=value_col, scope='africa', title=f"{value_col.title()} by Country")
fig1.update_layout(width = 2000, height = 600)
st.plotly_chart(fig1, use_container_width=True)

# === Confirmed Cases Over Time ===

st.subheader("📅 Case Fatality Rate vs Country")
fig5 = px.bar(_table_month, x="Country", y="Case_Fatality_Rate",  title="Case Fatality Rate")
st.plotly_chart(fig5, use_container_width=True)

st.subheader("📅 Vaccine Deployed Ratio vs Country")
fig5 = px.bar(_table_month, x="Country", y="Vaccine_Deployed_Ratio",  title="Vaccine Deployed Ratio")
st.plotly_chart(fig5, use_container_width=True)
# === Clade Distribut
st.subheader("📅 Weekly_New_Cases vs Report_Date")
fig5 = px.bar(_table_month1, x="Report_Date", y="Weekly_New_Cases",  title="Vaccine Deployed Ratio")
st.plotly_chart(fig5, use_container_width=True)
# === Clade Distribution ===

st.subheader("📊 Correlation Heatmap")
numeric_cols = filtered_df.loc[:,['Testing_Laboratories','Deployed_CHWs','Active_Surveillance_Sites','Confirmed_Cases','Suspected_Cases']]
corr = numeric_cols.corr()
fig4, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig4)


col1 = st.sidebar.selectbox("Confirmed cases scattered plot" , options = ['Deployed_CHWs','Active_Surveillance_Sites','Testing_Laboratories','Vaccinations_Administered','Vaccine_Coverage','Trained_CHWs','Vaccine_Deployed','Vaccine_Dose_Allocated'])
st.subheader(f"{col1.title()} vs Confirmed_Cases")
fig2 = px.scatter(_table_month, x="Confirmed_Cases", y="Deployed_CHWs", color="Country", hover_data=["Country"], title=f"{col1.title()} vs Confirmed Cases")
st.plotly_chart(fig2, use_container_width=True)

col2 = st.sidebar.selectbox("Comparing Clades" , options = ['Confirmed_Cases','Deaths','Weekly_New_Cases','Case_Fatality_Rate'])
if "Clade" in filtered_df.columns:
    st.subheader(f"🧬 {col2.title()} by Clade")
    fig6 = px.pie(_table_clade,values = col2, names="Clade", title=f"{col2.title()} by Clade")
    st.plotly_chart(fig6)

# === Show Raw Data ===
with st.expander("🔎 View Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)
