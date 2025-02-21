"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})
df

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

map_data = {
    "City": [
        "Kathmandu",
        "Thamel",
        "Patan",
        "Bhaktapur",
        "Pokhara",
        "Lalitpur",
        "Bharatpur",
        "Biratnagar",
        "Birgunj",
        "Dharan",
        "Butwal",
        "Hetauda",
        "Janakpur",
    ],
    "lat": [
        27.7172,
        27.7149,
        27.6667,
        27.6710,
        28.2096,
        27.6667,
        27.6833,
        26.4831,
        27.0167,
        26.8126,
        27.7000,
        27.4167,
        26.7337,
    ],
    "lon": [
        85.3240,
        85.3123,
        85.3167,
        85.4278,
        83.9856,
        85.3167,
        84.4333,
        87.2833,
        84.8667,
        87.2833,
        83.4500,
        85.0333,
        85.9167,
    ],
}
if st.checkbox("Show Map"):
    st.map(map_data)
    # st.table(map_data)

