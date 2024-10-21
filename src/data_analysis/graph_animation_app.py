import dash
from dash import html, dcc, Output, Input
import dash_cytoscape as cyto
import json
import time
from threading import Thread

# Load extra layouts
cyto.load_extra_layouts()

# Load graph data from JSON file
with open('/Users/ajithsenthil/Desktop/SibilanceAIWebsite/CompPsychoBackend/src/data_analysis/graph_data.json', 'r') as f:
    graph_data = json.load(f)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # For deploying if needed

# Create the app layout
app.layout = html.Div([
    html.H1("Evolving Graph Network Animation"),
    cyto.Cytoscape(
        id='cytoscape',
        elements=graph_data[0]['elements'],  # Initialize with the first time step
        style={'width': '100%', 'height': '600px'},
        layout={'name': 'preset'},  # Use preset positions
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'background-color': '#0074D9',
                    'width': 20,
                    'height': 20
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 'mapData(weight, 0, 1, 1, 10)',
                    'line-color': '#FF4136'
                }
            }
        ]
    ),
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    ),
    html.Button('Play/Pause', id='play-button', n_clicks=0),
    html.Button('Next', id='next-button', n_clicks=0),
    html.Div(id='current-time', style={'margin-top': '20px'})
])

# Variables to control playback
is_playing = True
current_time = 0

@app.callback(
    Output('cytoscape', 'elements'),
    Output('current-time', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('play-button', 'n_clicks'),
    Input('next-button', 'n_clicks')
)
def update_graph(n_intervals, play_clicks, next_clicks):
    global is_playing, current_time
    ctx = dash.callback_context

    if not ctx.triggered:
        # Initial call
        return graph_data[current_time]['elements'], f"Time: {current_time}"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'play-button':
            is_playing = not is_playing
        elif button_id == 'next-button':
            current_time = (current_time + 1) % len(graph_data)
            return graph_data[current_time]['elements'], f"Time: {current_time}"
        elif button_id == 'interval-component' and is_playing:
            current_time = (current_time + 1) % len(graph_data)
        # Update the elements
        return graph_data[current_time]['elements'], f"Time: {current_time}"

if __name__ == '__main__':
    app.run_server(debug=True)