
from bokeh.io import curdoc
from bokeh.models import Slider, ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column, row
import pandas as pd
import numpy as np
import nret
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# Create a dictionary with the fixed values
data = {
    'n': 1.5,
    'cab': 45,
    'car': 7,
    'cbrown': 0,
    'cw': 0.015,
    'cm': 0.005,
    'lai': [],
    'lidfa': 60,
    'hspot': 0.5,
    'ant': 1,
    'prot': 0.001,
    'cbc': 0.005,
    'rsoil': 0.5,
    'psoil': 0.5
}

# Fill in the 'lai' column with values ranging from 0 to 7 with a step of 0.01
data['lai'] = [round(1 + x * 0.01, 2) for x in range(701)]

# Create a DataFrame using the dictionary
input_samples_var_lai = pd.DataFrame(data)

# Display the DataFrame
print(input_samples_var_lai.head())


tts = 40
tto = 0
psi = 0

# Assuming your DataFrame is named 'input_samples_var_lai'
# Create an empty list to store the reflectance arrays
reflectance_data = []
fapar_data = []
rho = []

# Load up the spectral response functions for S2

srf = np.loadtxt(r"C:\Users\Mahmo\OneDrive - Delft University of Technology\Store\IHE\WSD\13_MOD_9\Thesis\NRET\S2A-SRF.csv", skiprows=1,
                 # Uncomment id you're uploading the spectral response function  instead of providing a pat to the Drive
                 delimiter=",")[100:, :]
srf[:, 1:] = srf[:, 1:]/np.sum(srf[:, 1:], axis=0)
srf_land = srf[:, [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]].T

# Loop over the DataFrame rows
for index, row in input_samples_var_lai.iterrows():
    n = row['n']
    cab = row['cab']
    car = row['car']
    cbrown = row['cbrown']
    cw = row['cw']
    cm = row['cm']
    lai = row['lai']
    lidfa = row['lidfa']
    hspot = row['hspot']
    ant = row['ant']
    prot = row['prot']
    cbc = row['cbc']
    rsoil_value = row['rsoil']
    psoil_value = row['psoil']

    # Run PROSAIL for each set of parameters
    rho_canopy = nret.run_prosail(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts, tto,
                                  psi, ant, prot, cbc, rsoil=np.full(2101, rsoil_value), psoil=np.full(2101, psoil_value))
    rho.append(rho_canopy)

    # calculate fAPAR for each set of parameters
    fapar_data.append(np.round(nret.calculate_fapar(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts,
                      tto, psi, ant, prot, cbc, rsoil=np.full(2101, rsoil_value), psoil=np.full(2101, psoil_value)), 4))

    # Calculate the reflectance by applying the spectral response functions
    reflectance_data.append(
        np.round(np.sum(rho_canopy * srf_land, axis=-1), 4))


# Add the lists as new columns in the DataFrame
input_samples_var_lai['canopy_reflectance'] = rho
input_samples_var_lai['reflectance'] = reflectance_data
input_samples_var_lai['fAPAR'] = fapar_data

# =========================================================================cab==================================================================
# Create a dictionary with the fixed values
data = {
    'n': 1.5,
    'cab': [],
    'car': 7,
    'cbrown': 0,
    'cw': 0.015,
    'cm': 0.005,
    'lai': 4,
    'lidfa': 60,
    'hspot': 0.5,
    'ant': 1,
    'prot': 0.001,
    'cbc': 0.005,
    'rsoil': 0.5,
    'psoil': 0.5
}

# Fill in the 'cab' column with values ranging from 0.001 to 0.0025 with a step of 0.00001
data['cab'] = [round(20 + x * 0.5, 2) for x in range(141)]

# Create a DataFrame using the dictionary
input_samples_var_cab = pd.DataFrame(data)

# Display the DataFrame
print(input_samples_var_cab.head())

tts = 40
tto = 0
psi = 0

# Assuming your DataFrame is named 'input_samples_var_cab'
# Create an empty list to store the reflectance arrays
reflectance_data = []
fapar_data = []
rho = []

# Load up the spectral response functions for S2 (Source: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/document-library/-/asset_publisher/Wk0TKajiISaR/content/sentinel-2a-spectral-responses)

# Loop over the DataFrame rows
for index, row in input_samples_var_cab.iterrows():
    n = row['n']
    cab = row['cab']
    car = row['car']
    cbrown = row['cbrown']
    cw = row['cw']
    cm = row['cm']
    lai = row['lai']
    lidfa = row['lidfa']
    hspot = row['hspot']
    ant = row['ant']
    prot = row['prot']
    cbc = row['cbc']
    rsoil_value = row['rsoil']
    psoil_value = row['psoil']

    # Run PROSAIL for each set of parameters
    rho_canopy = nret.run_prosail(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts, tto,
                                  psi, ant, prot, cbc, rsoil=np.full(2101, rsoil_value), psoil=np.full(2101, psoil_value))
    rho.append(rho_canopy)

    # calculate fAPAR for each set of parameters
    fapar_data.append(np.round(nret.calculate_fapar(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts,
                      tto, psi, ant, prot, cbc, rsoil=np.full(2101, rsoil_value), psoil=np.full(2101, psoil_value)), 4))

    # Calculate the reflectance by applying the spectral response functions
    reflectance_data.append(
        np.round(np.sum(rho_canopy * srf_land, axis=-1), 4))


# Add the lists as new columns in the DataFrame
input_samples_var_cab['canopy_reflectance'] = rho
input_samples_var_cab['reflectance'] = reflectance_data
input_samples_var_cab['fAPAR'] = fapar_data

# ==========================================================================figures==============================================================================

# Initialize the Dash app
app = Dash(__name__)
server = app.server

# Sample wavelength data
wavelengths = np.arange(400, 2501)

# Dummy data for canopy reflectance
# Replace with your actual `input_samples_var_cab` DataFrame

# Define the layout of the app
app.layout = html.Div([
    html.H1("Interactive Canopy Reflectance Plot"),

    dcc.Graph(id='reflectance-graph'),

    dcc.Slider(
        id='chl-slider',
        min=20.00,
        max=90.00,
        step=0.50,
        value=20.00,
        marks={str(i): str(i) for i in range(20, 91, 10)},
        tooltip={"placement": "bottom", "always_visible": True},
    ),

    html.Div(id='slider-output-container')
])

# Define the callback to update the graph


@app.callback(
    Output('reflectance-graph', 'figure'),
    Input('chl-slider', 'value')
)
def update_graph(cab_value):
    # Select the appropriate reflectance data
    selected_row = input_samples_var_cab[input_samples_var_cab['cab'] == cab_value]
    reflectance = selected_row['canopy_reflectance'].values[0]

    # Create the Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavelengths, y=reflectance,
                  mode='lines', name=f"chl = {cab_value}"))

    fig.update_layout(
        title=f"Canopy Reflectance for chl = {cab_value}",
        xaxis_title="Wavelength",
        yaxis_title="Reflectance",
        template="plotly_white",
        height=600,
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
