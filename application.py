import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'flight.pkl')

# Load the model
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.title('Flight Price Prediction')

def convert_to_minutes(h, m):
    t_min = int(h * 60) + int(m)
    return t_min

def main():
    st.markdown('### Enter Flight Details')
    
    arr_time = st.text_input('Arrival Time (YYYY-MM-DDTHH:MM)')
    dep_time = st.text_input('Departure Time (YYYY-MM-DDTHH:MM)')
    stops = st.number_input('Number of Stops', min_value=0, max_value=10)
    flight_class = st.selectbox('Class', ['Economy', 'First', 'PremiumEconomy'])
    airline = st.selectbox('Airline', ['AirAsia', 'Vistara', 'AllianceAir', 'GO FIRST', 'Indigo', 'SpiceJet', 'AkasaAir', 'StarAir'])
    source = st.selectbox('Source', ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Banglore', 'Ahmedabad'])
    destination = st.selectbox('Destination', ['Ahmedabad', 'Delhi', 'Banglore', 'Hyderabad', 'Kolkata', 'Mumbai'])
    
    if st.button('Predict'):
        day_left = (pd.to_datetime(dep_time) - pd.to_datetime(arr_time)).days
        dep_hour, dep_min = pd.to_datetime(dep_time).hour, pd.to_datetime(dep_time).minute
        arr_hour, arr_min = pd.to_datetime(arr_time).hour, pd.to_datetime(arr_time).minute
        dept_time = convert_to_minutes(dep_hour, dep_min)
        arr_time = convert_to_minutes(arr_hour, arr_min)
        duration_min = abs(arr_time - dept_time)
        
        if flight_class == 'Economy':
            class_ECONOMY, class_FIRST, class_PREMIUMECONOMY = 1, 0, 0
        elif flight_class == 'First':
            class_ECONOMY, class_FIRST, class_PREMIUMECONOMY = 0, 1, 0
        else:
            class_ECONOMY, class_FIRST, class_PREMIUMECONOMY = 0, 0, 1
        
        airlines = ['AirAsia', 'Vistara', 'AllianceAir', 'GO FIRST', 'Indigo', 'SpiceJet', 'AkasaAir', 'StarAir']
        airline_AirAsia, airline_AkasaAir, airline_AllianceAir, airline_GOFIRST, airline_Indigo, airline_SpiceJet, airline_StarAir, airline_Vistara = [1 if a == airline else 0 for a in airlines]
        
        sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Banglore', 'Ahmedabad']
        source_Delhi, source_Kolkata, source_Mumbai, source_Chennai, source_Banglore, source_Ahmedabad = [1 if s == source else 0 for s in sources]
        
        destinations = ['Ahmedabad', 'Delhi', 'Banglore', 'Hyderabad', 'Kolkata', 'Mumbai']
        destination_Mumbai, destination_Ahmedabad, destination_Delhi, destination_Banglore, destination_Hyderabad, destination_Kolkata = [1 if d == destination else 0 for d in destinations]
        
        feature_vector = [
            duration_min, dept_time, arr_time, stops, day_left, 
            pd.to_datetime(dep_time).month, pd.to_datetime(dep_time).day,
            airline_AirAsia, airline_AkasaAir, airline_AllianceAir, airline_GOFIRST, 
            airline_Indigo, airline_SpiceJet, airline_StarAir, airline_Vistara, 
            class_ECONOMY, class_FIRST, class_PREMIUMECONOMY, 
            source_Delhi, source_Kolkata, source_Mumbai, source_Chennai, source_Banglore, source_Ahmedabad,
            destination_Mumbai, destination_Ahmedabad, destination_Delhi, destination_Banglore, destination_Hyderabad, destination_Kolkata
        ]
        
        prediction = model.predict(np.array(feature_vector).reshape(1, -1))
        output = round(prediction[0], 2)
        
        st.markdown(f'Your Flight price is Rs. {output}')
    
if __name__ == "__main__":
    main()
