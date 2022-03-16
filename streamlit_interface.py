import streamlit as st
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.stats import rankdata
import numpy as np


def get_ranking(
    hyperopt_runtime_arr_mean,
    hyperopt_performance_arr_mean,
    hyperopt_performance_arr_min,
    hyperopt_performance_arr_max,
    hyperopt_performance_arr_std,
):
    interpolated_per_HPO_method = {}
    interpolator_per_HPO_method = {}
    data_per_HPO_method = {}

    data_per_HPO_method[hp_name] = [
        hyperopt_runtime_arr_mean,
        hyperopt_performance_arr_mean,
        hyperopt_performance_arr_min,
        hyperopt_performance_arr_max,
        hyperopt_performance_arr_std,
    ]
    interpolator_per_HPO_method[hp_name] = interp1d(
        hyperopt_runtime_arr_mean, hyperopt_performance_arr_mean, bounds_error=False
    )


@st.experimental_memo
def get_data_to_plot(temp_dir):
    print("accessing plot data")
    hyperopt_runtime_arr_mean = pickle.load(
        open(temp_dir + "/runtime_arr_mean.pkl", "rb",)
    )

    hyperopt_performance_arr_mean = pickle.load(
        open(temp_dir + "/performance_arr_mean.pkl", "rb",)
    )
    hyperopt_performance_arr_min = pickle.load(
        open(temp_dir + "/performance_arr_min.pkl", "rb",)
    )
    hyperopt_performance_arr_max = pickle.load(
        open(temp_dir + "/performance_arr_max.pkl", "rb",)
    )
    hyperopt_performance_arr_std = pickle.load(
        open(temp_dir + "/performance_arr_std.pkl", "rb",)
    )
    return (
        hyperopt_runtime_arr_mean,
        hyperopt_performance_arr_mean,
        hyperopt_performance_arr_min,
        hyperopt_performance_arr_max,
        hyperopt_performance_arr_std,
    )


@st.experimental_memo
def get_dataset_info(temp_dir):
    return pd.read_csv(temp_dir, sep=", ")


def get_file_structure(base_path):
    dir_names_per_depth = [x[0] for x in os.walk(base_path)]

    dir_structure_dict = {}
    for dir in dir_names_per_depth[1:]:
        broken_dir = dir.split("/")[1:]
        if not broken_dir[0] in dir_structure_dict:
            dir_structure_dict[broken_dir[0]] = {}
        if len(broken_dir) > 1:
            if not broken_dir[1] in dir_structure_dict[broken_dir[0]]:
                dir_structure_dict[broken_dir[0]][broken_dir[1]] = {}
            if len(broken_dir) > 2:
                if (
                    not broken_dir[2]
                    in dir_structure_dict[broken_dir[0]][broken_dir[1]]
                ):
                    dir_structure_dict[broken_dir[0]][broken_dir[1]][
                        broken_dir[2]
                    ] = set()
                if len(broken_dir) > 3:
                    dir_structure_dict[broken_dir[0]][broken_dir[1]][broken_dir[2]].add(
                        broken_dir[3]
                    )
    return dir_structure_dict


base_path = "streamlit_cached_data/"
visualize_error = "area"
error_type = "std"
st.set_page_config(layout="wide")

# get a dictionary of dictionaries with the file structure of all the data to be plotted
dir_structure_dict = get_file_structure(base_path)

st.sidebar.title("HPO benchmarks for DeepMTP")

# dataset selector
dataset_option = st.sidebar.selectbox(
    "Select a dataset", list(dir_structure_dict.keys())
)

st.header("Basic Dataset info")
dataset_info_df = get_dataset_info("dataset_info.csv")

st.dataframe(dataset_info_df[dataset_info_df["dataset_name"] == dataset_option])

st.markdown("""---""")

# performance metric selector, conditioned on the selected dataset
metric_option = st.sidebar.selectbox(
    "Select a metric", list(dir_structure_dict[dataset_option].keys())
)

# performance averaging selector, conditioned on the selected dataset and performance metric
averaging_option = st.sidebar.selectbox(
    "Averaging method option",
    list(dir_structure_dict[dataset_option][metric_option].keys()),
)


# checkboxes for dynamically applying log scaling the two axis
log_x_axis = st.sidebar.checkbox("Scale x-axis")
log_y_axis = st.sidebar.checkbox("Scale y-axis")
# add_x_axis_slider = st.checkbox("Enable custom range on x-axis")
header_cols = [c for c in st.columns(6)]
header_cols[1].header("Validation " + metric_option + "_" + averaging_option)
header_cols[4].header("Test " + metric_option + "_" + averaging_option)
plot_cols = [c for c in st.columns(2)]

for idx, mode in enumerate(["val", "test"]):

    fig = go.Figure()

    # fig.update_layout(
    #    autosize=False, width=2000, height=1000, paper_bgcolor="LightSteelBlue",
    # )
    fig.update_layout(
        xaxis=dict(title="Wall clock time (seconds)"),
        yaxis=dict(title=metric_option + "_" + averaging_option),
    )

    for hp_name in list(
        dir_structure_dict[dataset_option][metric_option][averaging_option]
    ):
        # print(hp_name)
        temp_dir = (
            base_path
            + dataset_option
            + "/"
            + metric_option
            + "/"
            + averaging_option
            + "/"
            + hp_name
            + "/"
            + mode
        )

        (
            hyperopt_runtime_arr_mean,
            hyperopt_performance_arr_mean,
            hyperopt_performance_arr_min,
            hyperopt_performance_arr_max,
            hyperopt_performance_arr_std,
        ) = get_data_to_plot(temp_dir)

        if visualize_error == "area":
            fig.add_trace(
                go.Scatter(
                    x=hyperopt_runtime_arr_mean,
                    y=hyperopt_performance_arr_mean,
                    mode="lines+markers",
                    name=hp_name,
                )
            )
            if error_type == "min_max":
                fig.add_trace(
                    go.Scatter(
                        x=hyperopt_runtime_arr_mean,
                        y=hyperopt_performance_arr_min,
                        mode="lines",
                        showlegend=False,
                        line=dict(width=0),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=hyperopt_runtime_arr_mean,
                        y=hyperopt_performance_arr_max,
                        mode="lines",
                        name=hp_name + "_area",
                        fill="tonexty",
                        showlegend=True,
                        line=dict(width=0),
                    )
                )

            elif error_type == "std":
                fig.add_trace(
                    go.Scatter(
                        x=hyperopt_runtime_arr_mean,
                        y=hyperopt_performance_arr_mean - hyperopt_performance_arr_std,
                        mode="lines",
                        showlegend=False,
                        line=dict(width=0),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=hyperopt_runtime_arr_mean,
                        y=hyperopt_performance_arr_mean + hyperopt_performance_arr_std,
                        mode="lines",
                        name=hp_name + "_area",
                        fill="tonexty",
                        showlegend=True,
                        line=dict(width=0),
                    )
                )
        else:
            # plt.errorbar(hyperopt_runtime_arr_mean, hyperopt_performance_arr_mean, yerr=[hyperopt_performance_arr_min, hyperopt_performance_arr_max], fmt=stamp_per_hyperopt_method[idx], label=label_per_hyperopt_method[idx])
            pass

        # fig.update_xaxes(
        #     exponentformat="power",
        #     dtick="D0",
        #     tickfont=dict(size=20),
        #     # rangeslider_visible=False,
        # )
        # fig.update_yaxes(exponentformat="power", dtick="D0", tickfont=dict(size=20))

    if log_x_axis:
        fig.update_xaxes(
            exponentformat="power", type="log", dtick="D0", tickfont=dict(size=20)
        )

    # if log_y_axis:
    #     fig.update_yaxes(
    #         exponentformat="power", type="log", dtick="D0", tickfont=dict(size=20)
    #     )

    if log_y_axis:
        fig.update_yaxes(exponentformat="power", type="log")
    # header_cols[idx].header(mode + " performance")
    plot_cols[idx].plotly_chart(fig, use_container_width=True)

resolution = st.slider("Define the sampling resolution: ", 10, 1000, 500)

rank_cols = [c for c in st.columns(2)]
for idx, mode in enumerate(["val", "test"]):
    interpolated_per_HPO_method = {}
    interpolator_per_HPO_method = {}
    data_per_HPO_method = {}
    for hp_name in list(
        dir_structure_dict[dataset_option][metric_option][averaging_option]
    ):
        # print(hp_name)
        temp_dir = (
            base_path
            + dataset_option
            + "/"
            + metric_option
            + "/"
            + averaging_option
            + "/"
            + hp_name
            + "/"
            + mode
        )

        (
            hyperopt_runtime_arr_mean,
            hyperopt_performance_arr_mean,
            hyperopt_performance_arr_min,
            hyperopt_performance_arr_max,
            hyperopt_performance_arr_std,
        ) = get_data_to_plot(temp_dir)

        data_per_HPO_method[hp_name] = [
            hyperopt_runtime_arr_mean,
            hyperopt_performance_arr_mean,
            hyperopt_performance_arr_min,
            hyperopt_performance_arr_max,
            hyperopt_performance_arr_std,
        ]
        interpolator_per_HPO_method[hp_name] = interp1d(
            hyperopt_runtime_arr_mean, hyperopt_performance_arr_mean, bounds_error=False
        )
    # resolution = 500
    # calculate the min and max time points across all HPO methods
    min_x = min([data[0][0] for HPO_name, data in data_per_HPO_method.items()])
    max_x = max([data[0][-1] for HPO_name, data in data_per_HPO_method.items()])
    global_x = np.linspace(min_x, max_x, resolution)

    hpo_names = list(
        dir_structure_dict[dataset_option][metric_option][averaging_option]
    )

    for hp_name in hpo_names:
        interpolated_per_HPO_method[hp_name] = interpolator_per_HPO_method[hp_name](
            global_x
        )

    raw_ratings_arr = np.array([d for d in interpolated_per_HPO_method.values()])
    rankings_arr = rankdata(-1 * raw_ratings_arr, axis=0, method="min")

    rankings_arr = rankings_arr.astype(float)
    rankings_arr[np.isnan(raw_ratings_arr)] = np.nan

    one_pos = np.where(rankings_arr == 1)
    two_pos = np.where(rankings_arr == 2)
    four_pos = np.where(rankings_arr == 4)
    five_pos = np.where(rankings_arr == 5)

    rankings_arr[one_pos] = 5
    rankings_arr[two_pos] = 4
    rankings_arr[four_pos] = 2
    rankings_arr[five_pos] = 1

    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(title="Wall clock time (seconds)"), yaxis=dict(title="ranking"),
    )

    for i in range(rankings_arr.shape[0]):
        nan_ids = np.isnan(rankings_arr[i, :])
        fig.add_trace(
            go.Scatter(
                x=global_x[~nan_ids],
                y=rankings_arr[i, ~nan_ids],
                mode="lines+markers",
                name=hpo_names[i],
            )
        )
    rank_cols[idx].plotly_chart(fig, use_container_width=True)
