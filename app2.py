import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl



palette = ['#FF5733', '#FFC300', '#DAF7A6', '#C70039', '#900C3F']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)

# Load your pre-trained model
rfc = joblib.load('model2.joblib')

# Set page config
st.set_page_config(page_title="Machine Predictive Maintenance", layout="wide", page_icon="🔧")

img = Image.open("factory_image.jpg")
img_cropped = img.crop((100, 50, img.width, 170))  

# Display factory image
st.image(img_cropped)

# Adding a title to the app
st.title("👷 Predictive Maintenance Dashboard")

st.markdown("""
Welcome to the **Machine Predictive Maintenance Dashboard**! This application leverages advanced machine learning techniques to predict the maintenance needs of your machines, helping you prevent unexpected failures and optimize your maintenance schedule. 
""")
    
# Tabs
tab1, tab2, tab3 = st.tabs(["🔮Prediction", "🔍 Data Exploration", "👨‍💻 About the Author / Project"])

# Mapping for machine types
type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = pd.DataFrame(columns=['Machine Type', 'Air Temperature [K]', 'Process Temperature [K]', 'Rotational Speed [RPM]', 'Torque [N-m]', 'Tool Wear [min]', 'Prediction'])

with tab1:
    st.markdown("## 🔮Predictive Maintenance Prediction")
    st.markdown("""
    This tool predicts potential failures in factory machinery to help with maintenance planning.
    Input the operational parameters of the machine and get a prediction on whether a failure might occur.
    """)

    # Sidebar for user inputs only visible in the Prediction tab
    with st.sidebar:
        st.header("Input Features")
        selected_type = st.selectbox('Select Machine Type', list(type_mapping.keys()), key='type_select')
        air_temperature = st.slider("Air Temperature [K]", min_value=290.0, max_value=310.0, value=299.5, step=0.1, key='air_temp')
        process_temperature = st.slider("Process Temperature [K]", min_value=300.0, max_value=320.0, value=308.7, step=0.1, key='proc_temp')
        rotational_speed = st.slider("Rotational Speed [RPM]", min_value=1000, max_value=3000, value=2134, key='rot_speed')
        torque = st.slider("Torque [N-m]", min_value=3.5, max_value=77.0, value=45.33, key='torque')
        tool_wear = st.slider("Tool Wear [min]", min_value=0, max_value=250, value=127, key='tool_wear')

        # if st.button('Predict Failure', key='predict_button'):
        #     prediction = rfc.predict([[type_mapping[selected_type], float(air_temperature), 
        #                                 float(process_temperature), int(rotational_speed),
        #                                 float(torque), int(tool_wear)]])
        #     failure_pred = 'Failure' if prediction[0] == 1 else 'No Failure'

        if st.button('Predict Failure', key='predict_button'):
            prediction = rfc.predict([[type_mapping[selected_type], float(air_temperature), 
                                    float(process_temperature), int(rotational_speed),
                                    float(torque), int(tool_wear)]])
            if prediction[0] == 1:
                st.error(f'Prediction: Failure')
                failure_pred = 'Failure'
            else:
                st.success(f'Prediction: No Failure')
                failure_pred = 'No Failure'
                        # Add the prediction to the history
            new_entry = pd.DataFrame({
                'Machine Type': [selected_type],
                'Air Temperature [K]': [air_temperature],
                'Process Temperature [K]': [process_temperature],
                'Rotational Speed [RPM]': [rotational_speed],
                'Torque [N-m]': [torque],
                'Tool Wear [min]': [tool_wear],
                'Prediction': [failure_pred]
            })
            st.session_state['prediction_history'] = pd.concat([st.session_state['prediction_history'], new_entry], ignore_index=True)

    # Display the operational parameters and their impacts
    col1, col2 = st.columns([0.25, 0.75])

    with col1:
        st.subheader("Operational Parameters")
        st.json({
            "Machine Type": selected_type,
            "Air Temperature [K]": air_temperature,
            "Process Temperature [K]": process_temperature,
            "Rotational Speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool Wear [min]": tool_wear
        })

    with col2:
        st.subheader("Predictions History")
        st.write("""
        Here you can display the history of predictions made by the model.
        """)
        if not st.session_state['prediction_history'].empty:
            st.write(st.session_state['prediction_history'])

with tab2:
    st.markdown("## 🔍 Data Exploration")
    st.markdown("Here we can get more details on the effect of variables to machine failures.")
    
    # Display the dataset
    # Load the dataset
    dataset = pd.read_csv('predictive_maintenance.csv')

    # Put the filters in columns to make the layout cleaner and add separation
    col1, col2 = st.columns(2, gap='large')
    # add separation between the columns

    with col1:
        machine_type = st.selectbox('Select Machine Type', list(type_mapping.keys()))
        temperature_air = st.slider("Air Temperature [K]", min_value=290.0, max_value=310.0, value=(290.0, 310.0), step=0.1)
        temperature_process = st.slider("Process Temperature [K]", min_value=300.0, max_value=320.0, value=(300.0, 320.0), step=0.1)

    with col2:
        rotational_speed = st.slider("Rotational Speed [RPM]", min_value=1000, max_value=3000, value=(1000, 3000))    
        torque = st.slider("Torque [N-m]", min_value=3.5, max_value=77.0, value=(3.5, 77.0))
        tool_wear = st.slider("Tool Wear [min]", min_value=0, max_value=250, value=(0, 250))



    reverse_type_mapping = {"Low": "L", "Medium": "M", "High": "H"}    # Filter the dataset based on the user inputs
    filtered_data = dataset[(dataset['Type'] == reverse_type_mapping[machine_type]) &
                            (dataset['Air temperature [K]'] >= temperature_air[0]) & (dataset['Air temperature [K]'] <= temperature_air[1]) &
                            (dataset['Process temperature [K]'] >= temperature_process[0]) & (dataset['Process temperature [K]'] <= temperature_process[1]) &
                            (dataset['Torque [Nm]'] >= torque[0]) & (dataset['Torque [Nm]'] <= torque[1]) &
                            (dataset['Tool wear [min]'] >= tool_wear[0]) & (dataset['Tool wear [min]'] <= tool_wear[1]) &
                            (dataset['Rotational speed [rpm]'] >= rotational_speed[0]) & (dataset['Rotational speed [rpm]'] <= rotational_speed[1])]
    st.write(filtered_data.head())

    col1, col2 = st.columns([0.35,0.65], gap='large')

    with col1:        
        st.write("#### Distribution of Machine Failures")
        st.bar_chart(filtered_data[(filtered_data['Target'] == 1) & (filtered_data['Failure Type'] != 'No Failure')]['Failure Type'].value_counts())
    
    with col2:
        st.write("#### Correlation across variables")
        numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = filtered_data[numeric_cols].corr()
        # Display the correlation matrix as a heatmap
        heatmap = correlation_matrix.style.background_gradient(cmap='Greens', axis=None)
        st.write(heatmap)
    
    col1, col2, col3 = st.columns(3)
    failure_types = filtered_data['Failure Type'].unique()

    with col1:
        st.write("#### Air Temperature vs. Failure")
        fig, ax = plt.subplots()
        data = [filtered_data.loc[filtered_data['Failure Type'] == ft, 'Air temperature [K]'].values for ft in failure_types]
        ax.boxplot(data, meanline=True, showmeans=True, meanprops={'color': 'red'})
        ax.set_xticklabels(failure_types)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("#### Process Temperature vs. Failure")
        fig, ax = plt.subplots()
        data = [filtered_data.loc[filtered_data['Failure Type'] == ft, 'Process temperature [K]'].values for ft in failure_types]
        ax.boxplot(data, meanline=True, showmeans=True, meanprops={'color': 'red'})
        ax.set_xticklabels(failure_types)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col3:
        st.write("#### Tool Wear vs. Failure")
        fig, ax = plt.subplots()
        data = [filtered_data.loc[filtered_data['Failure Type'] == ft, 'Tool wear [min]'].values for ft in failure_types]
        ax.boxplot(data, meanline=True, showmeans=True, meanprops={'color': 'red'})
        ax.set_xticklabels(failure_types)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    col1, col2, col3= st.columns(3)
    with col1:
        st.write("#### Rotational Speed vs. Failure")
        fig, ax = plt.subplots(figsize=(5, 4))  
        data = [filtered_data.loc[filtered_data['Failure Type'] == ft, 'Rotational speed [rpm]'].values for ft in failure_types]
        ax.boxplot(data, meanline=True, showmeans=True, meanprops={'color': 'red'})
        ax.set_xticklabels(failure_types)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("#### Torque vs. Failure")
        fig, ax = plt.subplots(figsize=(5, 4))  

        # Get unique failure types
        failure_types = filtered_data['Failure Type'].unique()

        # Create a list of arrays, each containing 'Torque [Nm]' values for a specific failure type
        data = [filtered_data.loc[filtered_data['Failure Type'] == ft, 'Torque [Nm]'].values for ft in failure_types]

        # Create the boxplot with a red mean line
        ax.boxplot(data, meanline=True, showmeans=True, meanprops={'color': 'red'})

        # Set the x-tick labels to the failure types
        ax.set_xticklabels(failure_types)
        plt.xticks(rotation=45)

        st.pyplot(fig)

    with col3:
        # plot the mean features of all variables when the 'target' ==1
        st.write("#### Mean Features of Variables when there is an error")
        
        # Select only numeric columns and drop the 'target' column and 'UDI' column
        numeric_columns = filtered_data.select_dtypes(include=[np.number]).drop(columns=['Target', 'UDI'])  

        # Calculate mean of numeric columns
        mean_features = numeric_columns.groupby(filtered_data['Failure Type']).mean().reset_index()

        # Drop the 'No Failure' row
        mean_features = mean_features[mean_features['Failure Type'] != 'No Failure']
        
        st.write(mean_features)
    with tab3:
        st.markdown("## 👨‍💻 About the Author / Project")
        st.markdown("""
        This project was created by [Joan Solé Roig](https://www.linkedin.com/in/joan-sole-roig/). 
        The project is part of a project of the Prototyping With AI elective of the MSc in Business Analytics at the ESADE Business School.
        The aim of the project is to simulate a real predictive maintenance software that could be used in a factory to predict machine failures.
        The project Repository can be found on GitHub [here](https://github.com/joansoleroig/Predictive-Maintenance-Streamlit-App/).           
        """)
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            #load a jpeg image
            img = Image.open("joan.jpeg")
            # make the image smaller in size
            st.image(img)
        with col2:
            st.markdown("""
            MSc in Business Analytics student at ESADE with strong analytical skills to design and manage business systems, and a focus on financial applications.
            Demonstrated ability to work with stakeholders, communicate insights, and adapt to new technologies.  
            Skilled in data analysis, utilizing SQL and Python for complex datasets, and visualizing results for effective communication.
            Excel in cross-functional teams and thrive in collaborative, detail-oriented environments.  
            Founder of LendaHand, a successful online teaching startup, showcasing proficiency in project management and strategic marketing, as well as a passion for finance and entrepreneurship.  
            Enthusiastic member of ESADE DataHub, Finance Society, 180° Consulting, E3, and Ennova associations.  
            With a proven track record, a blend of technical expertise and business acumen is employed to drive innovative solutions.  
            Looking to leverage experience, skills, and insights to contribute meaningfully in a dynamic business landscape.
            """)
st.markdown("""
    <hr/>
    <footer style="text-align: center;">
        <p>Connect with me:</p>
        <a href="https://github.com/joansoleroig" target="_blank" style="margin-right: 15px;">
            <img src="https://raw.githubusercontent.com/joansoleroig/Predictive-Maintenance-Streamlit-App/stream/github-mark.png" alt="Github Logo" width="30" height="30">
        </a>
        <a href="https://www.linkedin.com/in/joan-sole-roig/" target="_blank">
            <img src="https://raw.githubusercontent.com/joansoleroig/Predictive-Maintenance-Streamlit-App/stream/lkd.png" alt="LinkedIn Logo" width="30" height="30">
        </a>
    </footer>
    """, unsafe_allow_html=True)