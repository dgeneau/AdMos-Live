'''
Attempting Live Streaming from the AdMos Live 
- WebSocket stream 10Hz GPS data

'''

import plotly.graph_objs as go
from dash import Dash, dcc, html, State
from dash.dependencies import Input, Output
from datetime import datetime
import threading
import websocket
import json
import pandas as pd
import requests
import numpy as np
from scipy.signal import find_peaks

#basic functions in code
def calculate_cumulative_distance(df, speed_column, delta_t):
    """
    Calculate cumulative distance from instantaneous speed data in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the speed data.
    - speed_column: Name of the column containing speed values (in m/s).
    - delta_t: Constant time interval between measurements (in seconds).

    Returns:
    - DataFrame with a new column for cumulative distance.
    """
    if speed_column not in df.columns:
        raise ValueError(f"'{speed_column}' column not found in the DataFrame.")
    
    # Calculate incremental distances
    df['Incremental_Distance'] = df[speed_column] * delta_t
    
    # Calculate cumulative distance
    df['Cumulative_Distance'] = df['Incremental_Distance'].cumsum()

    return df



client_id = 'M9d3J3axpMU9z9xMfaqNtOB4BdxmsyPMK2v63yBC'
client_secret = 'POh9pZ1djjNOtS8rX9FzTYHAv3ARYIvaht9pfXlKPc3axcTaCPoxYehS3OVOGhRUn9ahaujugmftpajWC7zAW0LoxVEBMhhEIg86D4Yp5g05KfT9SLGSrin6oyd0SNnd'
#redirect_uri = 'http://localhost:55665'  # for local development only

#Your device manager account info here
username = "dgeneau@csipacific.ca"
password = "Joegeneau!1959"


data_login = {
  'grant_type': 'password',
  'username': username,
  'password': password
}

response = requests.post('https://api.asi.swiss/oauth2/token/',
                         data=data_login,
                         verify=False,
                         allow_redirects=False,
                         auth=(client_id, client_secret)).json()

access_token = response['access_token']


# now that we have an access token, we are free to access the
# user's data and do anything with it

devices_list = requests.get(f'https://api.asi.swiss/api/v1/devices/',
                            params={'current': True},
                            headers={'Authorization': f"Bearer {access_token}"},
                            verify=False).json()

# Get device ID in order to pull live information
print(devices_list)

#If number of devices is more than one, we can change it up here
device = devices_list.get('results')[0].get('id')

'''
Websocket pulling for identified device above
'''


# Initialize an empty DataFrame to store the accumulated data
data = pd.DataFrame(columns=['timestamp', 'speed'])



def on_message(ws, message):
    global data
    parsed_message = json.loads(message)
    

    # Handle different types of messages
    if "message" in parsed_message and parsed_message["message"] == "Authorization needed":
        print("Authorization needed. Sending authorization...")
        # Send authorization details (modify as per your server requirements)
        auth_message = json.dumps({"action": "authorize", "token": access_token})
        ws.send(auth_message)
    elif "message" in parsed_message and parsed_message["message"] == "Authorized":
        print("Authorized. Sending")
    elif "gnss" in parsed_message:
        # Extract data points under 'gnss' key
        gnss_data = parsed_message["gnss"]
        gnss_df = pd.DataFrame(gnss_data)
        #gnss_df = calculate_cumulative_distance(gnss_df, 'speed', 0.1)

        # Append new data to the global DataFrame
        data = pd.concat([data, gnss_df], ignore_index=True)

        data.to_csv('GNSS_data.csv')

        
    else:
        print("Unexpected message format.")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("### WebSocket closed ###")

def on_open(ws):
    print("### WebSocket connected ###")

def start_websocket():
    websocket.enableTrace(True)

    websocket_url = f"wss://api.asi.swiss/ws/v1/preprocessed-data/{device}/"

    # Add token to headers if required by your WebSocket server
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    ws = websocket.WebSocketApp(websocket_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                header=headers)

    ws.on_open = on_open
    ws.run_forever()

# Start the websocket in a separate thread
websocket_thread = threading.Thread(target=start_websocket)
websocket_thread.start()


# Dash app setup
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Boat Speed Dashboard"),
    
    # Button to start collecting data
    html.Button("Collect Data", id="collect-data-button", n_clicks=0),
    
    # Interval component (disabled by default)
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0, disabled=True),
    
    # Graph to display data
    dcc.Graph(id='live-graph')
])

@app.callback(
    Output('interval-component', 'disabled'),
    Input('collect-data-button', 'n_clicks')
)
def enable_interval(n_clicks):
    """
    Enables the interval (i.e., data collection/graph updates) only 
    after the 'Collect Data' button is clicked at least once.
    """
    if n_clicks and n_clicks > 0:
        return False  # Enable interval
    return True       # Keep it disabled if button hasn't been clicked

@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    global data
    
    if data.empty:
        return go.Figure()

    # Filter the DataFrame for the last 1000 samples
    recent_data = data.iloc[-1001:-1,].reset_index(drop=True)

    # Convert timestamps from ms to sec
    recent_data['timestamp'] = recent_data['timestamp'] / 1000

    def convert_timestamp(timestamp_s):
        dt = datetime.utcfromtimestamp(timestamp_s)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] 

    recent_data['timestamp'] = recent_data['timestamp'].apply(convert_timestamp)
    recent_data['accel'] = np.diff(recent_data['speed'])/(1/10)

    stroke_peaks, _ = find_peaks(np.array(recent_data['accel']*-1), height=3, distance=10)
    if stroke_peaks is None or len(stroke_peaks) < 2:
        stroke_rate = []
    else:
        # stroke_peaks[i+1] - stroke_peaks[i] ~ # of samples between strokes
        stroke_rate = 60/(np.diff(stroke_peaks)/10)

    # Create figure
    fig = go.Figure()

    # Example: Just show the current speed and stroke rate indicators
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=float(round(recent_data['speed'].iloc[-1], 2)),
        delta={"reference": 5.67, "valueformat": ".0f"},
        title={"text": "Instantaneous Speed (m/s)"},
        domain={'y': [0, 1], 'x': [1, 0.75]},
                    number={
                "font": {
                    "color": "white",   # <-- Change this to your desired color
                    "size": 40       # You can also control font size here if you want
                }},
    ))
    
    if len(stroke_rate) < 1:
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=0.0,
            delta={"reference": 32, "valueformat": ".0f"},
            title={"text": "Stroke Rate (SPM)"},
            domain={'y': [0, .2], 'x': [0, 1]},
                        number={
                "font": {
                    "color": "white",   # <-- Change this to your desired color
                    "size": 40       # You can also control font size here if you want
                }},
        ))
    else:
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=float(round(stroke_rate[-1], 2)),
            delta={"reference": 32, "valueformat": ".0f"},
            title={"text": "Stroke Rate (SPM)"},
            domain={'y': [0, .2], 'x': [0.75, 1]},
                        number={
                "font": {
                    "color": "white",   # <-- Change this to your desired color
                    "size": 40       # You can also control font size here if you want
                }},
        ))
        mean_vel = np.mean(recent_data['speed'][stroke_peaks[-2]:stroke_peaks[-1]])
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=float(round(mean_vel, 2)),
            delta={"reference": 5.67, "valueformat": ".0f"},
            title={"text": "Stroke Velocity (m/s)"},
            number={
                "font": {
                    "color": "white",   # <-- Change this to your desired color
                    "size": 40       # You can also control font size here if you want
                }},
            domain={'y': [0, .2], 'x': [0.25, 0.75]}
        ))
    fig.update_layout(paper_bgcolor = "#2d2d2d")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

