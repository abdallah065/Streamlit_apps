import pandas as pd # data processing
import joblib
import streamlit as st #web app

#ML model
from sklearn.metrics.pairwise import cosine_similarity


df1 = pd.read_csv("anime_similarity1.csv")
df2 = pd.read_csv("anime_similarity2.csv")
df3 = pd.read_csv("anime_similarity3.csv")
df4 = pd.read_csv("anime_similarity4.csv")
df5 = pd.read_csv("anime_similarity5.csv")


#concat all dataframes
ani_sim_df = pd.concat([df1,df2,df3,df4,df5], axis = 0)

anime_similarity = joblib.load("anime/anime_rec_model.joblib")


def anime_recommendation(ani_name):
    """
    This function will return the top 5 shows with the highest cosine similarity value and show match percent
    
    example:
    >>>Input: 
    
    anime_recommendation('Death Note')
    
    >>>Output: 
    
    Recommended because you watched Death Note:

                    #1: Code Geass: Hangyaku no Lelouch, 57.35% match
                    #2: Code Geass: Hangyaku no Lelouch R2, 54.81% match
                    #3: Fullmetal Alchemist, 51.07% match
                    #4: Shingeki no Kyojin, 48.68% match
                    #5: Fullmetal Alchemist: Brotherhood, 45.99% match 

               
    """
    
    number = 1
    rec_list = []
    for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6]:
        rec_dict = {}
        rec_dict["name"] = ani_sim_df.iloc[anime][0]
        rec_dict["match"] = round(ani_sim_df.iloc[anime][ani_name]*100,2)
        
        rec_list.append(rec_dict)
    return rec_list

#anime name list
anime_list = ani_sim_df['name'].values.tolist()

#streamlit app
st.title("Anime Recommendation System")
st.write("This app will recommend you 5 anime shows based on the anime you enter")
#selectbox from anime_list only one selection
ani_name = st.selectbox("Select an anime", anime_list)

if st.button("Recommend"):
    reco_list = anime_recommendation(ani_name)
    
    #show every anime in the list in box
    st.write(f"## Recommended because you watched {ani_name}:")
    for number in range(len(reco_list)):
        st.write(f"### {number}: {reco_list[number]['name']}:  {reco_list[number]['match']}% match")
        number += 1
