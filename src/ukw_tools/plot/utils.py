import plotly.express as px
import pandas as pd
from ..stats.utils import analyse_stats
import numpy as np


COLOR_MAP = {
    0: "blue",
    1: "aqua",
    2: "red",
    3: "green",
    4: "darkgoldenrod",
    5: "hotpink",
    "outside": "blue",
    "insertion": "aqua",
    "caecum": "yellow",
    "withdrawal": "orange",
    "resection": "red",
    "low_quality": "grey",
    "water_jet": "blue",
    "snare": "green",
    "grasper": "hotpink",
    "nbi": "purple",
}

def segmentation_to_plot_df(segmentation, length):
    df = pd.DataFrame()
    n = [i for i in range(length)]
    _empty = [0 for i in range(length)]
    for key, value in segmentation.items():
        _df = pd.DataFrame()
        _df["n"] = n
        _df["label"] = [key]*len(n)
        _df["value"] = _empty.copy()

        pos = []
        for range_tuple in value:
            pos.extend(list(range(range_tuple[0], range_tuple[1])))
        _df.loc[_df.n.isin(pos), "value"] = 1


        _df = _df.loc[_df.value == 1, :]
        df = df.append(_df)

    return df

def get_plot(df,
    fps = None,
    width = 1100, 
    height = 300, 
    font_size = 18,
    title_font_size = 18,
    marker_size = 20,
    color_map = COLOR_MAP,
    ):
    if fps:
        df["minutes"] = df["n"]/fps/60
        select_x = "minutes"
    else:
        select_x = "n"
    plot = px.scatter(
        df,
        x=select_x,
        y="label",
        color="label",
        symbol="label",
        symbol_sequence=["line-ns", "line-ns"],
        width=width,
        height=height,
        category_orders={
                # "_type": ["annotations", "predictions", "predictions_smooth"],
                # "name": ["outside","caecum", "tool", "nbi"]
        },
        color_discrete_map= color_map,
        custom_data = ["n"]
        )
    plot.update_traces(
        marker=dict(size=marker_size),
        selector=dict(mode='markers'),
        hovertemplate = "Time: %{x}<br>Label: %{y}<br>Frame Number: %{customdata[0]}",    
        )

    plot.update_layout(
        showlegend= False,
        font_size = font_size,
        title_font_size=title_font_size
    )
    
    return plot

def get_x_ref(group_index, color_index, n_color):
    # First mid of first group is @ 0
    step_size = 1/(n_color+2)
    if n_color % 2 == 0:
        center = (n_color-1)/2
    else: 
        center = (n_color-0.5)/2
    x = group_index + (step_size * (color_index - center))
    return x


def add_p_value_annotation(plot, comparisons, x_labels, print_p = False, _format=dict(interline=0.0, text_height=1.07, color='black'), y_range_base = (.93, .94)):
    _dict = plot.to_dict()
    _data = _dict["data"]

    # Get y range
    y_range = np.zeros([len(comparisons), 2])
    for i in range(len(comparisons)):
        y_range[i] = [y_range_base[0]-i*_format['interline'], y_range_base[1]-i*_format['interline']]

    for y_range_index, comparison in enumerate(comparisons):
        # Add Significance for groups
        x_values = comparison["x_values"]
        paired = comparison["paired"]
        color_values = comparison["color_values"]
        subplot = comparison["subplot"]
        if subplot:
            x_axis = "x" + str(subplot)
            y_axis = "y"+str(subplot)+" domain"
        else:
            x_axis = "x"
            y_axis = "y domain"

        compare = []
        data_indices = []
        color_legend = []
        color_indices = []

        # Specify in what y_range to plot for each pair of columns

        for i in [0,1]:
            for data_index, e in enumerate(_data):
                if not e["legendgroup"] in color_legend: color_legend.append(e["legendgroup"])
                if (e["xaxis"] == x_axis) and (e["legendgroup"] == color_values[i]):
                    selected = e
                    values = [selected["y"][j] for j, _x in enumerate(selected["x"]) if _x == x_values[i]]
                    compare.append(values)
                    data_indices.append(data_index)
                    color_indices.append(
                        color_legend.index(e["legendgroup"])
                    )
        
        try:
            assert len(compare) == 2
        except:
            print("Error: Not enough data to compare")
            print(f"x_values: {x_values}")
            print(f"color_values: {color_values}")
            print(f"x_axis: {x_axis}")
            continue
        x0 = get_x_ref(x_labels.index(x_values[0]),color_indices[0],len(color_legend))
        x1 = get_x_ref(x_labels.index(x_values[0]),color_indices[1],len(color_legend))
        # st.write(compare)
        # try:
        try:
            assert len(compare[0]) == len(compare[1])
        except:
            print(comparison)
            print(len(compare[0]), len(compare[1]))
        r = analyse_stats(np.array(compare[0]), np.array(compare[1]), paired)
        p_value = r["p_value"]
        # except:
        #     p_value = 1.5

        if p_value >= 0.05:
            symbol = 'ns'
            continue
        elif p_value >= 0.01: 
            symbol = '*'
        elif p_value >= 0.001:
            symbol = '**'
        elif p_value < 0.001:
            symbol = '***'
        else:
            # symbol = "ns"
            continue
        if print_p:
            symbol = f"{symbol} ({p_value:.3f})"
            # symbol = symbol
        # Vertical line
         
        plot.add_shape(type="line",
            xref=x_axis, yref=y_axis,
            x0=x0, y0=y_range[y_range_index][0],
            x1=x0, y1=y_range[y_range_index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Horizontal line
        plot.add_shape(type="line",
            xref=x_axis, yref=y_axis,
            x0=x0, y0=y_range[y_range_index][1], 
            x1=x1, y1=y_range[y_range_index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Vertical line
        plot.add_shape(type="line",
            xref=x_axis, yref=y_axis,
            x0=x1, y0=y_range[y_range_index][0], 
            x1=x1, y1=y_range[y_range_index][1],
            line=dict(color=_format['color'], width=2,)
        )

        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        plot.add_annotation(
            dict(
                font=dict(color=_format['color'],size=14),
                x=(x0 + x1)/2,
                y=y_range[y_range_index][1]*_format['text_height'],
                showarrow=False,
                text=symbol,
                textangle=0,
                xref=x_axis,
                yref=y_axis
        ))

    return plot

    