# Standard Library
from typing import List
from random import choice, sample

# Third party
import gif
import numpy as np
import plotly.graph_objects as go


# TODO: add more colors
COLORS = [("#ffba08", "#ff8c61"),
          ("#ffba08", "#43aa8b"),
          ("#00d59e", "#ab63fa"),
          ("#ff928b", "#b392ac"),
          ("#ff928b", "#b392ac"),
          ("#ffba08", "#ff8c61"),
          ("#ffba08", "#43aa8b"),
          ("#00d59e", "#ab63fa")]


@gif.frame
def make_plot(state: np.ndarray, colors: List[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=state, colorscale=colors))
    fig.update_traces(showscale=False)
    fig.update_layout(width=700,
                      height=700,
                      xaxis_visible=False,
                      yaxis_visible=False,
                      margin_t=0,
                      margin_b=0,
                      margin_l=0,
                      margin_r=0)
    return fig


def make_animation(states: np.ndarray, filename: str):
    colors = sample(choice(COLORS), k=2)
    c0, c1 = [c.strip("#") for c in sample(choice(COLORS), k=2)]
    filename = f"{filename}_{c0}_{c1}.gif"
    frames = [make_plot(s, colors) for s in states]
    gif.save(frames, filename, duration=50)
