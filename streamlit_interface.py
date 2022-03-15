import streamlit as st
import os
import pickle
import plotly.graph_objects as go


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

st.title("Hyperopt benchmarks for DeepMTP")

# dataset selector
dataset_option = st.selectbox("Select a dataset", list(dir_structure_dict.keys()))

# performance metric selector, conditioned on the selected dataset
metric_option = st.selectbox(
    "Select a metric", list(dir_structure_dict[dataset_option].keys())
)

# performance averaging selector, conditioned on the selected dataset and performance metric
averaging_option = st.selectbox(
    "Averaging method option",
    list(dir_structure_dict[dataset_option][metric_option].keys()),
)


# checkboxes for dynamically applying log scaling the two axis
log_x_axis = st.checkbox("Scale x-axis (time in seconds)")
log_y_axis = st.checkbox("Scale y-axis (time in seconds)")
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
