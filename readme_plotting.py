# data processing
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# plotting
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# image processing
import gif


BG_COLOR = "#696969"
CELLS_COLOR = "#f08080"
BOARD_COLOR = "#696969"
SELECT_3x3_COLOR = "#70c2b4"
SELECT_1x1_COLOR = "#fbab3a"
PLOT_COLORS = ("#9b76bc", "#618da7")


DIMS = {
    0: {"x0": -0.5, "y0": 4.5, "x1": 2.5, "y1": 1.5},
    1: {"x0": 0.5, "y0": 4.5, "x1": 3.5, "y1": 1.5},
    2: {"x0": 1.5, "y0": 4.5, "x1": 4.5, "y1": 1.5},
    3: {"x0": -0.5, "y0": 3.5, "x1": 2.5, "y1": 0.5},
    4: {"x0": 0.5, "y0": 3.5, "x1": 3.5, "y1": 0.5},
    5: {"x0": 1.5, "y0": 3.5, "x1": 4.5, "y1": 0.5},
    6: {"x0": -0.5, "y0": 2.5, "x1": 2.5, "y1": -0.5},
    7: {"x0": 0.5, "y0": 2.5, "x1": 3.5, "y1": -0.5},
    8: {"x0": 1.5, "y0": 2.5, "x1": 4.5, "y1": -0.5}
}


CENTRAL_DIMS = {
    0: {"x0": 0.5, "y0": 3.5, "x1": 1.5, "y1": 2.5},
    1: {"x0": 1.5, "y0": 3.5, "x1": 2.5, "y1": 2.5},
    2: {"x0": 2.5, "y0": 3.5, "x1": 3.5, "y1": 2.5},
    3: {"x0": 0.5, "y0": 2.5, "x1": 1.5, "y1": 1.5},
    4: {"x0": 1.5, "y0": 2.5, "x1": 2.5, "y1": 1.5},
    5: {"x0": 2.5, "y0": 2.5, "x1": 3.5, "y1": 1.5},
    6: {"x0": 0.5, "y0": 1.5, "x1": 1.5, "y1": 0.5},
    7: {"x0": 1.5, "y0": 1.5, "x1": 2.5, "y1": 0.5},
    8: {"x0": 2.5, "y0": 1.5, "x1": 3.5, "y1": 0.5}
}


REFS = {
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


def get_annotations(x):
    n = len(x)
    annotation_lookup = dict(zip(range(10), [""] + list("ABCDEFGHI")))
    return [[annotation_lookup[x[j, i]] for i in range(n)] for j in range(n)]


def update_annotations(fig, x, y):
    for annotation in fig["layout"]["annotations"]:
        annotation["xref"] = x
        annotation["yref"] = y
    return fig


def make_heatmap(z):
    return ff.create_annotated_heatmap(z=z,
                                       font_colors=["white"],
                                       coloraxis="coloraxis")


@gif.frame
def make_sliding_window_plot(big_fig, windows, *, dims, central_dims, refs):
    specs = [[{"colspan": 3, "rowspan": 3}] + [{}]*5, [{}]*6, [{}]*6]
    fig = make_subplots(rows=3,
                        cols=6,
                        specs=specs,
                        horizontal_spacing=0.015,
                        vertical_spacing=0.015,
                        start_cell="bottom-left",
                        print_grid=False)
    line_width = 6
    opacity = 0.2
    annotations = []
    big_fig = update_annotations(big_fig, "x", "y")
    annotations.extend(big_fig.layout.annotations)
    fig.add_trace(big_fig.data[0], row=1, col=1)

    for i, row in enumerate((1, 2, 3)):
        for j, col in enumerate((4, 5, 6)):
            window_refs = REFS[6 - 3*i + j]
            xref, yref = window_refs["xref"], window_refs["yref"]
            window_fig = make_heatmap(windows[i, j])
            window_fig = update_annotations(window_fig, xref, yref)
            annotations.extend(window_fig.layout.annotations)
            fig.add_trace(window_fig.data[0], row=row, col=col)

    shapes = []
    for i in range(9):
        window_refs = REFS[i]
        xref, yref = window_refs["xref"], window_refs["yref"]
        shapes.append({"type": "rect",
                       "x0": -0.5,
                       "y0": 2.5,
                       "x1": 2.5,
                       "y1": -0.5,
                       "xref": xref,
                       "yref": yref,
                       "line_width": line_width,
                       "line_color": BOARD_COLOR})

    shapes += [{"type": "rect",
                "x0": -0.5,
                "y0": 4.5,
                "x1": 4.5,
                "y1": -0.5,
                "line_width": line_width,
                "line_color": BOARD_COLOR},
               {"type": "rect",
                "x0": 0.5,
                "y0": 3.5,
                "x1": 3.5,
                "y1": 0.5,
                "line_width": line_width,
                "line_color": CELLS_COLOR},
               {"type": "rect",
                "xref": "x",
                "yref": "y",
                **dims,
                "line_width": line_width,
                "line_color": SELECT_3x3_COLOR,
                "fillcolor": SELECT_3x3_COLOR,
                "opacity": opacity},
               {"type": "rect",
                "xref": "x",
                "yref": "y",
                **central_dims,
                "line_width": line_width,
                "line_color": SELECT_1x1_COLOR,
                "fillcolor": SELECT_1x1_COLOR,
                "opacity": opacity},
               {"type": "rect",
                "xref": "x",
                "yref": "y",
                **dims,
                "line_width": line_width,
                "line_color": SELECT_3x3_COLOR},
               {"type": "rect",
                "xref": "x",
                "yref": "y",
                **central_dims,
                "line_width": line_width,
                "line_color": SELECT_1x1_COLOR},
               {"type": "rect",
                "x0": -0.5,
                "y0": 2.5,
                "x1": 2.5,
                "y1": -0.5,
                **refs,
                "line_width": line_width,
                "line_color": SELECT_3x3_COLOR,
                "fillcolor": SELECT_3x3_COLOR,
                "opacity": opacity},
               {"type": "rect",
                "x0": 0.5,
                "y0": 1.5,
                "x1": 1.5,
                "y1": 0.5,
                **refs,
                "line_width": line_width,
                "line_color": SELECT_1x1_COLOR,
                "fillcolor": SELECT_1x1_COLOR,
                "opacity": opacity},
               {"type": "rect",
                "x0": -0.5,
                "y0": 2.5,
                "x1": 2.5,
                "y1": -0.5,
                **refs,
                "line_width": line_width,
                "line_color": SELECT_3x3_COLOR},
               {"type": "rect",
                "x0": 0.5,
                "y0": 1.5,
                "x1": 1.5,
                "y1": 0.5,
                **refs,
                "line_width": line_width,
                "line_color": SELECT_1x1_COLOR}]

    fig.update_layout(height=500,
                      width=950,
                      shapes=shapes,
                      margin_t=10,
                      margin_b=10,
                      margin_l=10,
                      margin_r=10,
                      showlegend=False,
                      paper_bgcolor=BG_COLOR,
                      plot_bgcolor=BG_COLOR,
                      coloraxis={"colorscale": PLOT_COLORS,
                                 "showscale": False})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.layout.annotations = annotations
    return fig


def make_sliding_window_animation(cells, boundary="fixed"):
    if boundary == "fixed":
        game_board = np.pad(cells, 1)
    else:
        n = len(cells) - 1
        game_board = np.tile(cells, (3, 3))[n:-n, n:-n]

    big_fig = make_heatmap(game_board)
    windows = sliding_window_view(game_board, (3, 3))

    frames = []
    for i in range(9):
        fig = make_sliding_window_plot(big_fig,
                                       windows,
                                       dims=DIMS[i],
                                       central_dims=CENTRAL_DIMS[i],
                                       refs=REFS[i])
        frames.append(fig)

    gif.save(frames, f"{boundary}_sliding_window_animation.gif", 750)


cells = np.random.randint(0, 2, (3, 3))

# n = len(cells) - 1
# tiled_cells = np.tile(cells, (3, 3))
# periodic_boundary = sliding_window_view(tiled_cells[n:-n, n:-n], (3, 3))
# game_board, windows = tiled_cells, periodic_boundary

# make_heatmap(game_board)
make_sliding_window_animation(cells, boundary="fixed")
make_sliding_window_animation(cells, boundary="periodic")
