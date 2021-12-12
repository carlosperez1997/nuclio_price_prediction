import streamlit as st
import pandas as pd
import numpy as np
import pickle

status_opts = ('Ready to move', 'Under Construction')

location_opts = ('Selaiyur', 'Pammal', 'Sholinganallur', 'Perumbakkam',
       'Medavakkam', 'Chromepet', 'Kelambakkam', 'Madhavaram',
       'Perungudi', 'Veppampattu', 'Other')

builder_opts = ('DAC Promoters', 'Casagrand Builder Private Limited',
       'Appaswamy Real Estate', 'Radiance Realty Developers', 'seller',
       'viswaraj', 'Propsource Realty Private Limited', 'Karthick',
       'MC Foundation', 'ARB HOMES', 'Other')

variables = ['bathroom', 'age', 'area', 'bhk',
 'status_Ready to move',
 'status_Under Construction',
 'location_reduced_Chromepet',
 'location_reduced_Kelambakkam',
 'location_reduced_Madhavaram',
 'location_reduced_Medavakkam',
 'location_reduced_Other',
 'location_reduced_Pammal',
 'location_reduced_Perumbakkam',
 'location_reduced_Perungudi',
 'location_reduced_Selaiyur',
 'location_reduced_Sholinganallur',
 'location_reduced_Veppampattu',
 'builder_reduced_ARB HOMES',
 'builder_reduced_Appaswamy Real Estate',
 'builder_reduced_Casagrand Builder Private Limited',
 'builder_reduced_DAC Promoters',
 'builder_reduced_Karthick',
 'builder_reduced_MC Foundation',
 'builder_reduced_Other',
 'builder_reduced_Propsource Realty Private Limited',
 'builder_reduced_Radiance Realty Developers',
 'builder_reduced_seller',
 'builder_reduced_viswaraj']



st.title('Chennai House Price Prediction')

area = st.sidebar.number_input(
    'Introduce area:', 
    min_value=300, 
    max_value=6500, 
    value= 1000,
    step=20
)

age = st.sidebar.number_input(
    'Introduce age:',
    min_value=0, 
    max_value=30, 
    value= 1,
    step=1
)

rooms = st.sidebar.number_input(
    'Introduce Number of rooms:', 
    min_value=1, 
    max_value=8, 
    value= 1,
    step=1
)

bathrooms = st.sidebar.number_input(
    'Introduce Number of bathrooms:', 
    min_value=1, 
    max_value=5, 
    value= 1,
    step=1
)

status = st.sidebar.selectbox(
    'Select status:',
    status_opts
)

location = st.sidebar.selectbox(
    'Select the Location:',
    location_opts
)

builder = st.sidebar.selectbox(
    'Select the Builder:',
    builder_opts
)

intro = st.write("""App developed with Python and Streamlit. \n
Dataset: https://www.kaggle.com/amaanafif/chennai-house-price \n
**Predict the price of a house in Chennai (Vietnam).** \n 
Select the location, builder, area, number of rooms and bathrooms and click Predict Price.""")

if st.button('Predict Price'):
    filename = './house_price_prediction_model.sav'
    model = pickle.load(open(filename, 'rb'))

    # Construct the dataframe to predict
    predict_dict = {}
    for x in variables:
        predict_dict[x] = 0

    predict_dict['bathroom'] = bathrooms
    predict_dict['bhk'] = bathrooms
    predict_dict['age'] = age
    predict_dict['area'] = area

    predict_dict['status_'+status] = 1
    predict_dict['builder_reduced_'+builder] = 1
    predict_dict['location_reduced_'+location] = 1

    predict_df = pd.DataFrame.from_dict( predict_dict, orient='index' ).transpose()
    
    X = np.array(predict_df[variables])
    result = model.predict(X)

    st.write('**Predicted Price:** %s' % np.round(result[0],2))