"""Methods and constants for creating animations"""

# Standard Library
from typing import List
from random import choice, sample

# Third party
import gif
import numpy as np
import plotly.graph_objects as go


# TODO: add more colors
COLORS = [("#155b92", "#15925e"),
          ("#9b76bc", "#618da7"),
          ("#ffba08", "#ff8c61"),
          ("#ffba08", "#43aa8b"),
          ("#00d59e", "#ab63fa"),
          ("#ff928b", "#b392ac"),
          ("#ff928b", "#b392ac"),
          ("#ffba08", "#ff8c61"),
          ("#ffba08", "#43aa8b"),
          ("#00d59e", "#ab63fa")]


def make_plot(state: np.ndarray, colors: List[str]) -> go.Figure:
    height, width = state.shape
    fig = go.Figure(go.Heatmap(z=state, colorscale=colors))
    fig.update_traces(showscale=False)
    fig.update_layout(width=10*width,
                      height=10*height,
                      xaxis_visible=False,
                      yaxis_visible=False,
                      margin=dict(t=0, b=0, l=0, r=0))
    return fig


@gif.frame
def animation_frame(fig: go.Figure) -> go.Figure:
    return fig


def make_animation(states: List[np.ndarray], filename: str):
    print("animating...")
    colors = sample(choice(COLORS), k=2)
    frames = [animation_frame(make_plot(s, colors)) for s in states]
    filename += "_" + "_".join(c.strip("#") for c in colors)
    filename += f"_{len(frames)}_frames.gif"
    gif.save(frames, filename, duration=75)
