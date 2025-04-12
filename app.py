import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

# -------------------
# CONFIG
# -------------------
API_KEY = "54227622caf10f3698a6c14399c42dc6"
BASE_URL = "https://api.openweathermap.org/data/2.5/"
DATA_PATH = "../data/weather.csv"

# -------------------
# FETCH CURRENT WEATHER
# -------------------
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'],
        'cloud_coverage': data['clouds']['all'],
        'precipitation': data.get('rain', {}).get('1h', 0.0)
    }

# -------------------
# DATA PROCESSING
# -------------------
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp','Precipitation']]
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(7):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

# -------------------
# MAIN FUNCTION FOR STREAMLIT
# -------------------
def main():
    st.title("🌦️ Weather Forecast Dashboard")
    city = st.text_input("Enter city name", value="Mumbai")

    if st.button("Fetch Weather & Predict"):
        try:
            current_weather = get_current_weather(city)
            data = read_historical_data(DATA_PATH)
            X, y, le = prepare_data(data)
            rain_model = train_rain_model(X, y)

            # Direction conversion
            deg = current_weather['wind_gust_dir'] % 360
            compass_points = [
                ("N",0,11.25),("NNE",11.25,33.75),("NE",33.75,56.25),("ENE",56.25,78.75),("E",78.75,101.25),
                ("ESE",101.25,123.75),("SE",123.75,146.25),("SSE",146.25,168.75),("S",168.25,191.25),
                ("SSW",191.25,213.75),("SW",213.75,236.25),("WSW",236.25,258.75),("W",258.75,281.25),
                ("WNW",281.25,303.75),("NW",303.75,326.25),("NNW",326.25,348.75)
            ]
            direction = next(p for p, s, e in compass_points if s <= deg < e)
            direction_encoded = le.transform([direction])[0] if direction in le.classes_ else -1

            current_data = pd.DataFrame([{
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'WindGustDir': direction_encoded,
                'WindGustSpeed': current_weather['Wind_Gust_Speed'],
                'Humidity': current_weather['humidity'],
                'Pressure': current_weather['pressure'],
                'Temp': current_weather['current_temp'],
                'Precipitation': current_weather['precipitation']
            }])

            rain_prediction = rain_model.predict(current_data)[0]

            # Regression models
            temp_model = train_regression_model(*prepare_regression_data(data, 'Temp'))
            hum_model = train_regression_model(*prepare_regression_data(data, 'Humidity'))
            wind_model = train_regression_model(*prepare_regression_data(data, 'WindGustSpeed'))
            precip_model = train_regression_model(*prepare_regression_data(data, 'Precipitation'))

            future_temp = predict_future(temp_model, current_weather['temp_min'])
            future_humidity = predict_future(hum_model, current_weather['humidity'])
            future_windspeed = predict_future(wind_model, current_weather['Wind_Gust_Speed'])
            future_precip = predict_future(precip_model, current_weather['precipitation'])

            timezone = pytz.timezone('Asia/Kolkata')
            base_time = datetime.now(timezone).replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
            future_days = [(base_time + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

            forecast_df = pd.DataFrame({
                'Date': future_days,
                'Predicted Temp (°C)': future_temp,
                'Predicted Humidity (%)': future_humidity,
                'Predicted Wind Speed (km/h)': future_windspeed,
                'Predicted Precipitation (mm)': future_precip,
            })

            


            st.subheader(f"🌍 Current Weather in {city.title()}, {current_weather['country']}")
            st.markdown(f"""
                - **Temperature:** {current_weather['current_temp']}°C (Feels like {current_weather['feels_like']}°C)
                - **Humidity:** {current_weather['humidity']}%
                - **Wind:** {current_weather['Wind_Gust_Speed']} km/h from {direction}
                - **Clouds:** {current_weather['cloud_coverage']}%
                - **Rain?** {'Yes' if rain_prediction else 'No'}
                - **Description:** {current_weather['description'].title()}
            """)

            st.subheader("📈 7-Day Weather Forecast")
            st.dataframe(forecast_df,use_container_width=True,hide_index=True)

            st.download_button("Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecast.csv")
            st.subheader("📊 Forecast Trends")

            st.subheader("Temperature")
            st.line_chart(forecast_df.set_index('Date')[['Predicted Temp (°C)']])
            st.subheader("Humidity")
            st.line_chart(forecast_df.set_index('Date')[['Predicted Humidity (%)']])
            st.subheader("Wind Speed")
            st.line_chart(forecast_df.set_index('Date')[['Predicted Wind Speed (km/h)']])
            st.subheader("Precipitation")
            st.line_chart(forecast_df.set_index('Date')[['Predicted Precipitation (mm)']])
            st.dataframe(forecast_df)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# -------------------
# RUN APP
# -------------------
if __name__ == "__main__":
    main()
