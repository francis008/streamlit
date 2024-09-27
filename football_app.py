import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import ssl

st.title('NFL Football Stats (Rushing) Explorer')

st.markdown("""
This app performs simple webscraping of NFL Football player stats data (rushing stats).
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2025))))

# Web scraping of NFL player stats
@st.cache_data
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
    html = pd.read_html(url, header = 1)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk', 'Succ%'], axis=1)
    return playerstats

playerstats = load_data(selected_year)


# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
select_all_unique_team = ["All"] + sorted_unique_team
selected_team = st.sidebar.multiselect('Team', select_all_unique_team, default=None, placeholder='Choose teams')
if "All" in selected_team:
    selected_team = sorted_unique_team
    

# Sidebar - Position selection
unique_pos = ['RB','QB','WR','FB','TE']
select_all_unique_pos = ["All"] + unique_pos
selected_pos = st.sidebar.multiselect('Position', select_all_unique_pos, default=None, placeholder='Choose positions')
if "All" in selected_pos:
    selected_pos = unique_pos

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header("Display Player Stats of Selected Team(s)")
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    df = df.select_dtypes(include=[np.number])

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)


