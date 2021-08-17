import plotly.graph_objects as go


def make_animation(states):
    colors = ((0, "#ffba08"), (1, "#43aa8b"))
    first_plot = go.Heatmap(z=states[0], colorscale=colors)
    frames = [go.Frame(data=go.Heatmap(z=state)) for state in states[1:]]

    fig = go.Figure(data=first_plot, frames=frames)
    fig.update_traces(showscale=False)
    fig.update_layout(width=700,
                      height=700,
                      xaxis_visible=False,
                      yaxis_visible=False)

    buttons = [{"label": "play",
                "method": "animate",
                "args": [None]},
               {"label": "pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}]}]
    menus = [{"type": "buttons", "visible": True, "buttons": buttons}]
    fig.update_layout(updatemenus=menus)
    fig.show()
