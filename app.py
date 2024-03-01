from dash import Dash, html, dash_table, callback, Input, Output
from dash.html import Div
import pandas as pd
from custom_dash import CustomDash

from main_service import start

# external JavaScript files
external_scripts = [
    {
        'src': 'https://code.jquery.com/jquery-3.7.1.min.js',
        'integrity': 'sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo=',
        'crossorigin': 'anonymous'
    }
]

# app = CustomDash(__name__)
app = Dash(__name__, external_scripts=external_scripts)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <div>My Custom header</div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <div>My Custom footer</div>
    </body>
</html>
'''


clustered_data = start()
# clustered_data.insert(loc=0, column="Date", value=clustered_data.index)
grouped_cluster = clustered_data.groupby("Cluster").count()
grouped_cluster.insert(loc=0, column="Cluster", value=grouped_cluster.index)

app.layout = Div([
    Div([
        Div([
            html.Span("Task: "),
            html.A([
                Div(children="Clustering", className="option-btn", id="btn-clustering")
            ]),
            html.A([
                Div(children="EDA", className="option-btn")
            ], id="btn-eda"),
            html.A([
                Div(children="Correlation", className="option-btn")
            ]),
            html.A([
                Div(children="Forecast", className="option-btn")
            ]),
        ], className="horizontal-panel"),

        Div([
            html.Span("Coins: "),
            html.A([
                Div(children="Coin 1", className="option-btn")
            ]),
            html.A([
                Div(children="Coin 2", className="option-btn")
            ])
        ], className="horizontal-panel ml-10"),

        Div([
            html.Span(children="Model: "),
            html.A([
                Div(children="ARIMA", className="option-btn")
            ]),
            html.A([
                Div(children="FBProphet", className="option-btn")
            ])
        ], className="horizontal-panel ml-10")

    ], className="horizontal-panel"),

    Div([
        Div(children="Clustered Dataset"),
        dash_table.DataTable(data=clustered_data.to_dict('records'), page_size=10, style_table={'overflowX': 'auto'}),
        Div([
            Div(children="Cluster Distribution"),
            dash_table.DataTable(data=grouped_cluster.to_dict('records'))
        ]),
    ], id="frg-cluster-distribution"),

    Div([
        Div(children="This is EDA")
    ], id="frg-eda"),


], className="container")


# @callback(
#         Output(component_id="frg-cluster-distribution", component_property="style"),
#         Output(component_id="frg-eda", component_property="style"),

#         Output(component_id="btn-clustering", component_property="style"),
#         Output(component_id="btn-eda", component_property="style"),

#         Input(component_id="btn-clustering", component_property="n_clicks"),
# )
# def show_clustering_panel(n):
#     print(n)
#     return (
#         {'display': 'block'},
#         {'display': 'none'},

#         {'border': '3px solid red'},
#         {'border': '1px solid black'},
#         )

# @callback(
#         Output(component_id="frg-cluster-distribution", component_property="style"),
#         Output(component_id="frg-eda", component_property="style"),

#         Output(component_id="btn-clustering", component_property="style"),
#         Output(component_id="btn-eda", component_property="style"),

#         Input(component_id="btn-eda", component_property="n_clicks"),
# )
# def show_eda_panel(n):
#     return (
#         {'display': 'none'},
#         {'display': 'block'},

#         {'border': '1px solid black'},
#         {'border': '3px solid red'},
#         )



if __name__ == '__main__':
    app.run(debug=True)
