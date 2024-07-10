import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import psycopg2
from psycopg2 import sql
import pandas as pd


# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Database connection parameters
db_params = {
    'dbname': 'health_predictions',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

# Function to create table if not exists
def create_table_if_not_exists(table_name, columns):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.SQL, columns))
        )
        cursor.execute(create_table_query)
        conn.commit()
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error while creating table: {error}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Function to insert data into database
def insert_data(table_name, data):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        placeholders = sql.SQL(', ').join(sql.Placeholder() * len(data))
        columns = sql.SQL(', ').join(map(sql.Identifier, data.keys()))
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            columns,
            placeholders
        )
        cursor.execute(query, list(data.values()))
        conn.commit()
        st.success(f"Data saved to {table_name} table")
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error while inserting data: {error}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'View Data'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    create_table_if_not_exists('diabetes_predictions', [
        "id SERIAL PRIMARY KEY",
        "pregnancies FLOAT",
        "glucose FLOAT",
        "blood_pressure FLOAT",
        "skin_thickness FLOAT",
        "insulin FLOAT",
        "bmi FLOAT",
        "diabetes_pedigree_function FLOAT",
        "age FLOAT",
        "prediction INTEGER",
        "diagnosis TEXT",
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ])

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        
        data = {
            'pregnancies': user_input[0],
            'glucose': user_input[1],
            'blood_pressure': user_input[2],
            'skin_thickness': user_input[3],
            'insulin': user_input[4],
            'bmi': user_input[5],
            'diabetes_pedigree_function': user_input[6],
            'age': user_input[7],
            'prediction': int(diab_prediction[0]),
            'diagnosis': diab_diagnosis
        }
        
        insert_data('diabetes_predictions', data)
        st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    create_table_if_not_exists('heart_disease_predictions', [
        "id SERIAL PRIMARY KEY",
        "age FLOAT",
        "sex FLOAT",
        "cp FLOAT",
        "trestbps FLOAT",
        "chol FLOAT",
        "fbs FLOAT",
        "restecg FLOAT",
        "thalach FLOAT",
        "exang FLOAT",
        "oldpeak FLOAT",
        "slope FLOAT",
        "ca FLOAT",
        "thal FLOAT",
        "prediction INTEGER",
        "diagnosis TEXT",
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ])

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])
        heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
        
        data = {
            'age': user_input[0],
            'sex': user_input[1],
            'cp': user_input[2],
            'trestbps': user_input[3],
            'chol': user_input[4],
            'fbs': user_input[5],
            'restecg': user_input[6],
            'thalach': user_input[7],
            'exang': user_input[8],
            'oldpeak': user_input[9],
            'slope': user_input[10],
            'ca': user_input[11],
            'thal': user_input[12],
            'prediction': int(heart_prediction[0]),
            'diagnosis': heart_diagnosis
        }
        
        insert_data('heart_disease_predictions', data)
        st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    create_table_if_not_exists('parkinsons_predictions', [
        "id SERIAL PRIMARY KEY",
        "fo FLOAT",
        "fhi FLOAT",
        "flo FLOAT",
        "jitter_percent FLOAT",
        "jitter_abs FLOAT",
        "rap FLOAT",
        "ppq FLOAT",
        "ddp FLOAT",
        "shimmer FLOAT",
        "shimmer_db FLOAT",
        "apq3 FLOAT",
        "apq5 FLOAT",
        "apq FLOAT",
        "dda FLOAT",
        "nhr FLOAT",
        "hnr FLOAT",
        "rpde FLOAT",
        "dfa FLOAT",
        "spread1 FLOAT",
        "spread2 FLOAT",
        "d2 FLOAT",
        "ppe FLOAT",
        "prediction INTEGER",
        "diagnosis TEXT",
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input = [float(x) for x in user_input]
        parkinsons_prediction = parkinsons_model.predict([user_input])
        parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        
        data = {
            'fo': user_input[0],
            'fhi': user_input[1],
            'flo': user_input[2],
            'jitter_percent': user_input[3],
            'jitter_abs': user_input[4],
            'rap': user_input[5],
            'ppq': user_input[6],
            'ddp': user_input[7],
            'shimmer': user_input[8],
            'shimmer_db': user_input[9],
            'apq3': user_input[10],
            'apq5': user_input[11],
            'apq': user_input[12],
            'dda': user_input[13],
            'nhr': user_input[14],
            'hnr': user_input[15],
            'rpde': user_input[16],
            'dfa': user_input[17],
            'spread1': user_input[18],
            'spread2': user_input[19],
            'd2': user_input[20],
            'ppe': user_input[21],
            'prediction': int(parkinsons_prediction[0]),
            'diagnosis': parkinsons_diagnosis
        }
        
        insert_data('parkinsons_predictions', data)
        st.success(parkinsons_diagnosis)

if selected == 'View Data':
    st.title('View Disease Prediction Data')

    # Function to fetch data from a table
    def fetch_data(table_name):
        try:
            conn = psycopg2.connect(**db_params)
            query = f"SELECT * FROM {table_name} ORDER BY created_at DESC LIMIT 100"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error fetching data from {table_name}: {e}")
            return pd.DataFrame()

    # Diabetes Data
    st.header('Diabetes Predictions')
    diabetes_df = fetch_data('diabetes_predictions')
    if not diabetes_df.empty:
        st.dataframe(diabetes_df)
    else:
        st.write("No diabetes prediction data available.")

    # Heart Disease Data
    st.header('Heart Disease Predictions')
    heart_df = fetch_data('heart_disease_predictions')
    if not heart_df.empty:
        st.dataframe(heart_df)
    else:
        st.write("No heart disease prediction data available.")

    # Parkinson's Disease Data
    st.header("Parkinson's Disease Predictions")
    parkinsons_df = fetch_data('parkinsons_predictions')
    if not parkinsons_df.empty:
        st.dataframe(parkinsons_df)
    else:
        st.write("No Parkinson's disease prediction data available.")

    # Add a refresh button
    if st.button('Refresh Data'):
        st.experimental_rerun()