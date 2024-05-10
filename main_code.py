import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from scipy import stats
import os

file_name = "futuristic_city_traffic.csv"
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, file_name)


@st.cache
def load_data():
    data = pd.read_csv(file_path, nrows=10000)
    
    # Define columns for which outliers need to be detected
    numerical_columns = ['Speed', 'Energy Consumption']

    # Drop outliers using Z-score method
    z_scores = np.abs(stats.zscore(data[numerical_columns]))
    threshold = 2
    data_no_outliers = data[(z_scores < threshold-1).all(axis=1)]

    return data_no_outliers


def load_models():
    data = load_data()
    
    # Define features and target variables
    X = data[['City', 'Weather', 'Economic Condition', 'Day Of Week', 'Hour Of Day']]
    y_vehicle_type = data['Vehicle Type']
    y_speed = data['Speed']
    y_random_event = data['Random Event Occurred']
    y_energy_consumption = data['Energy Consumption']
    y_is_peak_hour = data['Is Peak Hour']

    # One hot encode categorical variables
    column_transformer = ColumnTransformer(
        [('onehot', OneHotEncoder(), ['City', 'Weather', 'Economic Condition', 'Day Of Week'])],
        remainder='passthrough')

    X_encoded = column_transformer.fit_transform(X)


    # Train the Random Forest models
    rf_model_vehicle_type = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_vehicle_type.fit(X_encoded, y_vehicle_type)

    rf_model_speed = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_speed.fit(X_encoded, y_speed)

    rf_model_random_event = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_random_event.fit(X_encoded, y_random_event)

    rf_model_energy_consumption = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_energy_consumption.fit(X_encoded, y_energy_consumption)

    rf_model_is_peak_hour = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_is_peak_hour.fit(X_encoded, y_is_peak_hour)

    return (rf_model_vehicle_type, rf_model_speed, rf_model_random_event, rf_model_energy_consumption, rf_model_is_peak_hour), column_transformer




def main():
    # Set background image using set_page_config
    st.set_page_config(
        page_title="futuristic city",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    
    )
    st.sidebar.title("Navigation Bar")
    page = st.sidebar.radio("Go to", ["Statistics", "Model's Performance", "Model Testing"])

    if page == "Statistics":
        statistics_page()
    elif page == "Model's Performance":
        # Load data
        data = load_data()
        
        # Prepare X and y variables
        X = data[['City', 'Weather', 'Economic Condition', 'Day Of Week', 'Hour Of Day']]
        y_vehicle_type = data['Vehicle Type']
        y_speed = data['Speed']
        y_random_event = data['Random Event Occurred']
        y_energy_consumption = data['Energy Consumption']
        y_is_peak_hour = data['Is Peak Hour']

        # Load models
        models, column_transformer = load_models()

        # Call the performance function
        models_performance_page(X, y_vehicle_type, y_speed, y_random_event, y_energy_consumption, y_is_peak_hour, models, column_transformer)
    elif page == "Model Testing":
        model_testing_page()

def statistics_page():
    st.title("Statistics")
    data = load_data()
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    for col in numerical_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data[col], ax=ax)
        ax.set_title(f'Boxplot of {col} for Outlier Detection')
        st.pyplot(fig)
        plt.close(fig)

    for col in categorical_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=data[col], ax=ax)
        ax.set_title(f'Bar Chart of {col}')
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Hour Of Day', y='Speed', data=data, hue='Day Of Week', ax=ax)
    ax.set_title('Speed vs. Hour Of Day Colored by Day Of Week')
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Hour Of Day', y='Energy Consumption', data=data, hue='Day Of Week', ax=ax)
    ax.set_title('Energy Consumption vs. Hour Of Day Colored by Day Of Week')
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Hour Of Day', y='Traffic Density', data=data, hue='Day Of Week', ax=ax)
    ax.set_title('Traffic Density vs. Hour Of Day Colored by Day Of Week')
    st.pyplot(fig)
    plt.close(fig)

    numeric_data = data.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    plt.close(fig)

    fig = sns.pairplot(data[numerical_columns])
    fig.fig.suptitle('Pairwise Relationships Among Numerical Variables')
    st.pyplot(fig)
    plt.close(fig)

def models_performance_page(X, y_vehicle_type, y_speed, y_random_event, y_energy_consumption, y_is_peak_hour, models, column_transformer):
    st.title("Model's Performance")

    # Make predictions
    X_encoded = column_transformer.transform(X)
    predicted_vehicle_type = models[0].predict(X_encoded)
    predicted_speed = models[1].predict(X_encoded)
    predicted_random_event = models[2].predict(X_encoded)
    predicted_energy_consumption = models[3].predict(X_encoded)
    predicted_is_peak_hour = models[4].predict(X_encoded)

    # Calculate performance metrics
    vehicle_type_accuracy = accuracy_score(y_vehicle_type, predicted_vehicle_type)
    speed_mse = mean_squared_error(y_speed, predicted_speed)
    random_event_mse = mean_squared_error(y_random_event, predicted_random_event)
    energy_consumption_mse = mean_squared_error(y_energy_consumption, predicted_energy_consumption)
    is_peak_hour_accuracy = accuracy_score(y_is_peak_hour, predicted_is_peak_hour)

    # Display performance metrics
    st.write("Vehicle Type Accuracy:", vehicle_type_accuracy)
    st.write("Speed Mean Squared Error:", speed_mse)
    st.write("Random Event Mean Squared Error:", random_event_mse)
    st.write("Energy Consumption Mean Squared Error:", energy_consumption_mse)
    st.write("Is Peak Hour Accuracy:", is_peak_hour_accuracy)

def model_testing_page():
    st.title("Model Testing")

    cities = ["SolarisVille", "AquaCity", "Neuroburg", "Ecoopolis", "TechHaven","MetropolisX"]
    weather_conditions = ["Solar Flare", "Snowy", "Clear","Rainy","Electromagnetic Storm"]
    economic_conditions = ["Booming", "Recession", "Stable"]
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    



    city = st.selectbox("City", cities)
    weather = st.selectbox("Weather", weather_conditions)
    economic_condition = st.selectbox("Economic Condition", economic_conditions)
    day_of_week = st.selectbox("Day Of Week", days_of_week)
    hour_of_day = st.number_input("Hour Of Day", min_value=0, max_value=23, step=1)
    distance_in_km = st.number_input("Distance in Km",min_value=0, max_value=10000, step=10)

    
    

    if st.button("Travel"):
        models, column_transformer = load_models()
    

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'City': [city],
            'Weather': [weather],
            'Economic Condition': [economic_condition],
            'Day Of Week': [day_of_week],
            'Hour Of Day': [hour_of_day],
            'Energy Consumption': [0]
        
            
            
        })
        # Check if 'Hour Of Day' is present before dropping it
        if 'Hour Of Day' in input_data.columns:
            # Predict 'Is Peak Hour'
            X_is_peak_hour = input_data.drop(columns=['Energy Consumption'])
            predicted_is_peak_hour = models[4].predict(column_transformer.transform(X_is_peak_hour))[0]
        else:
            st.error("Column 'Hour Of Day' is missing in the input data.")

        # Transform input data using column transformer
        input_data_encoded = column_transformer.transform(input_data)
        # Make predictions
        predicted_vehicle_type = models[0].predict(input_data_encoded)[0]
        predicted_speed = models[1].predict(input_data_encoded)[0]
        predicted_energy_consumption = models[3].predict(input_data_encoded)[0]
        predicted_random_event = models[2].predict(input_data_encoded)[0]

        # Predict 'Is Peak Hour'
        X_is_peak_hour = input_data_encoded
        predicted_is_peak_hour = models[4].predict(X_is_peak_hour)[0]



        # Calculate total hours and minutes
        total_hours = int((distance_in_km / predicted_speed) + hour_of_day)
        total_minutes = int(((distance_in_km / predicted_speed) + hour_of_day - total_hours) * 60)
    
    # Calculate arrival day
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        start_index = days_of_week.index(day_of_week)
        resulting_index = (start_index + total_hours // 24) % 7
        resulting_day = days_of_week[resulting_index]
    
    # Calculate remaining hours
        formatted_time = '{:02d}:{:02d}'.format(total_hours % 24, total_minutes)

        # Display predictions
        st.write("Predicted Vehicle Type:", predicted_vehicle_type)
        st.write(f"Estimated arrival time: {resulting_day} at: {formatted_time}")
        st.write("Predicted Energy Consumption in KWH:", predicted_energy_consumption*(distance_in_km/predicted_speed))
        st.write(f"Probability of a random event occured is {predicted_random_event*100}%")

        if predicted_is_peak_hour :
            st.write(f"<span style='color: yellow;'>Traveling on {day_of_week} at {hour_of_day} is a peak hour. It would be advisable to consider alternative routes or adjust the departure time if possible.</span>", unsafe_allow_html=True)

        else:
            st.write(f"{hour_of_day}h is not a Peak Hour. have a nice ride ;)")
        



if __name__ == "__main__":
    main()


