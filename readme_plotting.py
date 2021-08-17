# std lib
from random import choice

# data processing
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# plotting
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# image processing
import gif


BLUE_GREEN = ["#d9ed92",
              "#b5e48c",
              "#99d98c",
              "#76c893",
              "#52b69a",
              "#34a0a4",
              "#168aad",
              "#1a759f",
              "#1e6091"]

BLUE_TURQ = ["#006895",
             "#097898",
             "#13879b",
             "#1c979e",
             "#25a7a1",
             "#2fb6a3",
             "#38c6a6",
             "#41d6a9",
             "#4be5ac",
             "#54f5af"]

RED_BLUE = ["#b7094c",
            "#a01a58",
            "#892b64",
            "#723c70",
            "#5c4d7d",
            "#455e89",
            "#2e6f95",
            "#1780a1",
            "#0091ad"]

BLUE_YELLOW = ["#1c7f93",
               "#398b80",
               "#55986e",
               "#71a45c",
               "#8eb049",
               "#aabc37",
               "#c6c925",
               "#e3d512",
               "#ffe100"]

RED_ORANGE = ["#a60000",
              "#b11b00",
              "#bc3500",
              "#c75000",
              "#d36b00",
              "#de8500",
              "#e9a000",
              "#f4ba00",
              "#ffd500"]


COLORS = [BLUE_GREEN, BLUE_YELLOW, RED_ORANGE, BLUE_TURQ, RED_BLUE]


dims = {
    0: {"x0": -0.5, "y0": 4.5, "x1": 2.5, "y1": 1.5},
    1: {"x0": 0.5, "y0": 4.5, "x1": 3.5, "y1": 1.5},
    2: {"x0": 1.5, "y0": 4.5, "x1": 4.5, "y1": 1.5},
    3: {"x0": -0.5, "y0": 3.5, "x1": 2.5, "y1": 0.5},
    4: {"x0": 0.5, "y0": 3.5, "x1": 3.5, "y1": 0.5},
    5: {"x0": 1.5, "y0": 3.5, "x1": 4.5, "y1": 0.5},
    6: {"x0": -0.5, "y0": 2.5, "x1": 2.5, "y1": -0.5},
    7: {"x0": 0.5, "y0": 2.5, "x1": 3.5, "y1": -0.5},
    8: {"x0": 1.5, "y0": 2.5, "x1": 4.5, "y1": -0.5},
}


refs = {
    0: {"xref": "x16", "yref": "y16"},
    1: {"xref": "x17", "yref": "y17"},
    2: {"xref": "x18", "yref": "y18"},
    3: {"xref": "x10", "yref": "y10"},
    4: {"xref": "x11", "yref": "y11"},
    5: {"xref": "x12", "yref": "y12"},
    6: {"xref": "x4", "yref": "y4"},
    7: {"xref": "x5", "yref": "y5"},
    8: {"xref": "x6", "yref": "y6"}
}


@gif.frame
def crazy_plot(big_fig, figs, *, dims, refs, add_zero=False):
    specs = [[{"colspan": 3, "rowspan": 3}] + [{}]*5, [{}]*6, [{}]*6]

    fig = make_subplots(rows=3,
                        cols=6,
                        specs=specs,
                        horizontal_spacing=0.01,
                        vertical_spacing=0.015,
                        start_cell="bottom-left",
                        print_grid=False)

    # big fig
    fig.add_trace(big_fig.data[0], row=1, col=1)

    # row 1
    fig.add_trace(figs[0].data[0], row=1, col=4)
    fig.add_trace(figs[1].data[0], row=1, col=5)
    fig.add_trace(figs[2].data[0], row=1, col=6)

    # row 2
    fig.add_trace(figs[3].data[0], row=2, col=4)
    fig.add_trace(figs[4].data[0], row=2, col=5)
    fig.add_trace(figs[5].data[0], row=2, col=6)

    # row 3
    fig.add_trace(figs[6].data[0], row=3, col=4)
    fig.add_trace(figs[7].data[0], row=3, col=5)
    fig.add_trace(figs[8].data[0], row=3, col=6)

    shapes = [{"type": "rect",
               "xref": "x",
               "yref": "y",
               "line_width": 5,
               "line_color": "white",
               **dims},
              {"type": "rect",
               "x0": -0.5,
               "y0": 2.5,
               "x1": 2.5,
               "y1": -0.5,
               "line_width": 5,
               "line_color": "white",
               **refs}]

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.update_layout(height=500,
                      width=950,
                      shapes=shapes,
                      margin_t=10,
                      margin_b=10,
                      margin_l=10,
                      margin_r=10,
                      showlegend=False,
                      paper_bgcolor="#aaaaaa",
                      plot_bgcolor="rgba(0, 0, 0, 0)",
                      coloraxis={"colorscale": ["#696969"]*add_zero + colors,
                                 "showscale": False})
    return fig


def get_annotations(x):
    n = len(x)
    annotation_lookup = dict(zip(range(10), [""] + list("ABCDEFGHI")))
    return [[annotation_lookup[x[j, i]] for i in range(n)] for j in range(n)]


def update_annotations(fig, x, y):
    for annotation in fig['layout']['annotations']:
        annotation['xref'] = x
        annotation['yref'] = y
    return fig


def make_heatmap(state, colors, draw_rects=False):
    fig = ff.create_annotated_heatmap(z=state,
                                      annotation_text=get_annotations(state),
                                      coloraxis="coloraxis")
    if draw_rects:
        fig.add_shape(name="game board",
                      type="rect",
                      x0=1.5,
                      y0=6.5,
                      x1=6.5,
                      y1=1.5,
                      line_color="black",
                      line_width = 4)
        fig.add_shape(name="original array",
                      type="rect",
                      x0=2.5,
                      y0=5.5,
                      x1=5.5,
                      y1=2.5,
                      line_color="white",
                      line_width=4)
    fig.update_layout(width=400,
                      height=400,
                      xaxis_visible=False,
                      yaxis_visible=False,
                      margin_t=0,
                      margin_b=0,
                      margin_l=0,
                      margin_r=0,
                      coloraxis={"colorscale": colors, "showscale": False})
#     fig.show()
    return fig


def make_window_plot(states, colors, add_zero=True):
    fig = make_subplots(rows=3,
                        cols=3,
                        print_grid=False,
                        shared_yaxes=True,
                        shared_xaxes=True,
                        horizontal_spacing=0.05,
                        vertical_spacing=0.05,
                        start_cell="bottom-left")
    all_annotations = []
    for i in range(1, 4):
        for j in range(1, 4):
            s = states[i-1, j-1]
            heatmap_fig = ff.create_annotated_heatmap(z=s,
                                                      annotation_text=get_annotations(s),
                                                      coloraxis="coloraxis")
            ref = (i-1)*3 + j
            heatmap_fig = update_annotations(heatmap_fig, f"x{ref}", f"y{ref}")
            all_annotations.extend(heatmap_fig.layout.annotations)
            fig.add_trace(heatmap_fig.data[0], row=i, col=j)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)

    fig.update_layout(width=700,
                      height=700,
                      margin_t=10,
                      margin_b=10,
                      margin_l=10,
                      margin_r=10,
                      paper_bgcolor='rgba(0,0,0,0)',
                      coloraxis={"colorscale": ["#696969"]*add_zero + colors, "showscale": False})
    fig.layout.annotations = all_annotations
    fig.show()
    return fig


big_fig = make_heatmap(np.pad(x, 1), ["#696969"] + colors)
figs = [make_heatmap(x, ["#696969"] + colors) for x in fixed_boundary_windows.reshape(9, 3, 3)]

frames = [crazy_plot(big_fig, figs, dims=dims[i], refs=refs[i], add_zero=True) for i in range(9)]
gif.save(frames, "sliding_window_fixed_boundary.gif", duration=750)

