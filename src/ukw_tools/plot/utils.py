import plotly.express as px
import pandas as pd

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
        color_discrete_map= color_map
        )
    plot.update_traces(
        marker=dict(size=marker_size),
        selector=dict(mode='markers'),    
        )

    plot.update_layout(
        showlegend= False,
        font_size = font_size,
        title_font_size=title_font_size
    )
    
    return plot