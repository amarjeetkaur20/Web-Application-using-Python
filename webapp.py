#TASK 1 IMPORTING NECESSARY LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

#TASK 2 IMPORTING DATA
st.title("Prediction Model")  
st.set_option("deprecation.showfileUploaderEncoding", False)

pic = "https://www.nicepng.com/png/full/138-1385735_data-analytics-and-visualization-analysis-clipart.png"
st.image(pic,use_column_width=True)

st.header("Import the file here:")  
df = st.file_uploader("Import the time series csv file here. Columns must be labeled ds and y. The input to Prophet is always a dataframe with two columns: ds and y. The ds(datestamp) column should have time stored in it and y column should have the values of item to be predicted")
if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce')
    st.write(data)  
    max_data = data['ds'].max()
    st.write(max_data)

#TASK 3 SELECTING FORECAST HORIZON 
st.header("Period to be Predicted:")     
period_input = st.slider('How many periods would you like to forecast into the future?',1,365)
if df is not None:
    m = Prophet()
    m.fit(data)

#TASK 4 VISUALIZE FORECAST DATA
st.header("Predicted Values")
if df is not None:
    future = m.make_future_dataframe(periods=period_input)
    forecast = m.predict(future)
    fcst = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    fcst_filtered = fcst[fcst['ds'] > max_data]
    st.write(fcst_filtered)
    st.header("Predicted Trend")
    fig1 = m.plot(forecast)
    st.write(fig1)
    st.header("Yearly and Monthly Trend")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

#TASK 5 Download the Forecast Data
#The below link allows you to download the newly created forecast to your computer for further analysis and use.
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    
    b64 = base64.b64encode(csv_exp.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
