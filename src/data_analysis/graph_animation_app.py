import dash
from dash import html, dcc, Output, Input, State
import dash_cytoscape as cyto
import json

# Load extra layouts
cyto.load_extra_layouts()

# Load graph data from JSON file
with open('/Users/ajithsenthil/Desktop/SibilanceAIWebsite/CompPsychoBackend/src/data_analysis/graph_data.json', 'r') as f:
    graph_data = json.load(f)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # For deploying if needed

# Variables to control playback
is_playing = True
current_time = 0
direction = 1  # 1 for forward, -1 for backward

# Create the app layout
app.layout = html.Div([
    html.H1("Evolving Graph Network Animation"),
    cyto.Cytoscape(
        id='cytoscape',
        elements=graph_data[0]['elements'],  # Initialize with the first time step
        style={'width': '100%', 'height': '600px'},
        layout={'name': 'preset'},  # Use preset positions
        stylesheet=[
            # Node styles
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'background-color': '#636efa',
                    'width': 80,
                    'height': 80,
                    'font-size': 10,
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'color': '#000000',
                    'text-wrap': 'wrap',
                    'text-max-width': 70,
                    'text-overflow': 'ellipsis',
                    'overflow-wrap': 'break-word',
                    'overlay-padding': '6px',
                    'z-index': 10
                }
            },
            # Character-specific node styles
            {
                'selector': 'node[agent = "watson"]',
                'style': {
                    'background-color': '#1f77b4',  # Blue
                }
            },
            {
                'selector': 'node[agent = "sherlock"]',
                'style': {
                    'background-color': '#ff7f0e',  # Orange
                }
            },
            # Edge styles
            {
                'selector': 'edge',
                'style': {
                    # Adjust 'weight_scaled' for visualization
                    'label': 'data(weight)',
                    'font-size': 8,
                    'text-rotation': 'autorotate',
                    'text-margin-y': -10,
                    'width': 'mapData(weight_scaled, 0, 0.01, 2, 10)',
                    'line-color': '#FF0000',
                    'curve-style': 'bezier',
                    'opacity': 1,
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#FF0000',
                    'arrow-scale': 1.5
                }
            },
            # Hover effects
            {
                'selector': 'node:hover',
                'style': {
                    'border-color': '#000',
                    'border-width': 2,
                    'shadow-blur': 10,
                    'shadow-color': '#aaa',
                    'shadow-opacity': 0.7
                }
            }
        ]
    ),
    html.Div([
        html.Button('Previous', id='previous-button', n_clicks=0),
        html.Button('Play/Pause', id='play-button', n_clicks=0),
        html.Button('Next', id='next-button', n_clicks=0),
        html.Button('Reverse', id='reverse-button', n_clicks=0),
    ], style={'margin': '10px'}),
    html.Div(id='current-time', style={'margin-top': '10px', 'font-size': '20px'}),
    dcc.Interval(
        id='interval-component',
        interval=100,  # Update every 0.1 seconds
        n_intervals=0
    )
])

@app.callback(
    Output('cytoscape', 'elements'),
    Output('current-time', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('play-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    Input('previous-button', 'n_clicks'),
    Input('reverse-button', 'n_clicks'),
    State('cytoscape', 'elements')
)
def update_graph(n_intervals, play_clicks, next_clicks, previous_clicks, reverse_clicks, elements):
    global is_playing, current_time, direction
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
        elif button_id == 'previous-button':
            current_time = (current_time - 1) % len(graph_data)
            return graph_data[current_time]['elements'], f"Time: {current_time}"
        elif button_id == 'reverse-button':
            direction *= -1  # Toggle the direction
        elif button_id == 'interval-component' and is_playing:
            current_time = (current_time + direction) % len(graph_data)
        # Update the elements
        return graph_data[current_time]['elements'], f"Time: {current_time}"

if __name__ == '__main__':
    app.run_server(debug=True)