import streamlit as st 
import pickle
st.title("Penguin Classifier")
st.write("This app uses 6 inputs to predict the species of penguin using"
         " A model built on the palmer penguin's dataset. Use the form below"
         " to get started")

rf_pickle = open("random_forest_penguin.pickle", 'rb')
map_pickle = open("output_penguin.pickle", 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()
island = st.selectbox("Penguin Island", options=['Biscoe', 'Dream', 'Torgerson'])
sex = st.selectbox("Sex", options=['Female', 'Male'])

bill_length_mm = st.number_input("Bill Length (mm) ", min_value=0)
bill_depth_mm = st.number_input("Bill Depth (mm) ", min_value=0)
flipper_length_mm = st.number_input("Flipper Length (mm) ", min_value=0)
body_mass_g = st.number_input("Body Mass (g) ", min_value=0)
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == "Male":
    sex_male = 1

new_prediction = rfc.predict([[bill_length_mm, 
                               bill_depth_mm, 
                               flipper_length_mm, 
                               body_mass_g,
                               island_biscoe, 
                               island_dream, 
                               island_torgerson,
                                sex_male, 
                                sex_female]
                                ]
                                )
prediction_species = unique_penguin_mapping[new_prediction][0]
st.subheader("Predicting your Penguin's species:")
st.write(f"We predict your penguin is of the {prediction_species} species")
st.write(
    """
We used a Machine learning (Random Forest) algorithm to predict 
the species, the features used in this prediction are ranked by 
a relative importance below"""
)
st.image("feature_importance.png")