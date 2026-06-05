# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradio app to show the results embedded in the docs for each page.

The difference to `app.py` is that there are way less things displayed
and method-related data points can be highlighted via GET parameters.
"""

import os

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from processing import (
    get_model_ids,
    filter_data,
    compute_pareto_frontier,
    _get_metric_explanation,
    _TASK_PARETO_DEFAULTS,
    get_metric_preferences,
    format_df,
    load_task_results,
)


metric_preferences = {
    "accelerator_memory_reserved_avg": "lower",
    "accelerator_memory_max": "lower",
    "accelerator_memory_reserved_99th": "lower",
    "total_time": "lower",
    "train_time": "lower",
    "file_size": "lower",
    "test_accuracy": "higher",
    "train_loss": "lower",
    "num_trainable_params": "lower",
    "forgetting*": "lower",
}


def generate_pareto_plot(df, metric_x, metric_y, metric_preferences, highlight_type=""):
    """Generates a pareto frontier plot for the given metrics.
    If there is no highlight by (PEFT) type is given, the frontier
    points are individually colored and put into the legend.

    If a highlight by PEFT type is requested, all points are grayed
    out with the exception of the points matching the PEFT type.
    No points are added to the legend.

    This is useful when embedding the plot in the docs, first mode
    is good for general overviews while the second mode is good
    for highlighting one specific method.
    """
    if df.empty:
        return {}

    # Compute Pareto frontier and non-frontier points.
    pareto_df = compute_pareto_frontier(df, metric_x, metric_y, metric_preferences)
    non_pareto_df = df.drop(pareto_df.index)

    # Create an empty figure.
    fig = go.Figure()

    # Draw the line connecting Pareto frontier points.
    if not pareto_df.empty:
        # Sort the Pareto frontier points by metric_x for a meaningful connection.
        pareto_sorted = pareto_df.sort_values(by=metric_x)
        line_trace = go.Scatter(
            x=pareto_sorted[metric_x],
            y=pareto_sorted[metric_y],
            mode="lines",
            line={"color": "rgba(0,0,255,0.1)", "width": 4},
            name="Pareto Frontier",
        )
        fig.add_trace(line_trace)

    hover_data = {"experiment_name": True, "peft_type": True, metric_x: True, metric_y: True}

    # we want to highlight the pareto points and plot a legend in case we
    # don't highlight a specific method - this is useful when embedding the
    # benchmark as an overview to highlight the best methods.
    pareto_highlight_kwargs = {}
    if not highlight_type:
        pareto_highlight_kwargs = {"color": "experiment_name"}

    # Add non-frontier points in gray with semi-transparency.
    if not non_pareto_df.empty:
        highlight_mask = non_pareto_df["peft_type"].str.lower() == highlight_type
        no_pareto_df_no_highlight = non_pareto_df[~highlight_mask]
        no_pareto_df_highlight = non_pareto_df[highlight_mask]

        non_frontier_trace_no_highlight = go.Scatter(
            x=no_pareto_df_no_highlight[metric_x],
            y=no_pareto_df_no_highlight[metric_y],
            mode="markers",
            marker={"color": "rgba(128,128,128,0.5)", "size": 12},
            hoverinfo="text",
            text=no_pareto_df_no_highlight.apply(
                lambda row: (
                    f"experiment_name: {row['experiment_name']}<br>"
                    f"peft_type: {row['peft_type']}<br>"
                    f"{metric_x}: {row[metric_x]}<br>"
                    f"{metric_y}: {row[metric_y]}"
                ),
                axis=1,
            ),
            showlegend=False,
        )
        fig.add_trace(non_frontier_trace_no_highlight)

        if not no_pareto_df_highlight.empty:
            pareto_scatter = px.scatter(
                no_pareto_df_highlight,
                x=metric_x,
                y=metric_y,
                hover_data=hover_data,
            )
            for trace in pareto_scatter.data:
                trace.marker = {"size": 18, "color": "green"}
                fig.add_trace(trace)

    # Add Pareto frontier points with legend
    if not pareto_df.empty:
        highlight_mask = pareto_sorted["peft_type"].str.lower() == highlight_type
        pareto_df_no_highlight = pareto_sorted[~highlight_mask]
        pareto_df_highlight = pareto_sorted[highlight_mask]

        if not pareto_df_no_highlight.empty:
            pareto_scatter_no_highlight = px.scatter(
                pareto_df_no_highlight,
                x=metric_x,
                y=metric_y,
                hover_data=hover_data,
                **pareto_highlight_kwargs,
            )
            for trace in pareto_scatter_no_highlight.data:
                if pareto_highlight_kwargs:
                    trace.marker = {"size": 12}
                else:
                    trace.marker = {"size": 12, "color": "rgba(128,128,128,0.5)"}
                fig.add_trace(trace)

        if not pareto_df_highlight.empty:
            pareto_scatter_highlight = px.scatter(
                pareto_df_highlight,
                x=metric_x,
                y=metric_y,
                hover_data=hover_data,
                **pareto_highlight_kwargs,
            )
            for trace in pareto_scatter_highlight.data:
                trace.marker = {"size": 18, "color": "green"}
                fig.add_trace(trace)

    # Update layout with axes labels.
    fig.update_layout(
        title=f"{highlight_type} methods compared to Pareto Frontier for {metric_x} vs {metric_y}",
        template="seaborn",
        height=700,
        autosize=True,
        xaxis_title=metric_x,
        yaxis_title=metric_y,
    )

    return fig


def build_app(df):
    task_names = sorted(df["task_name"].unique())
    initial_task = "MetaMathQA" if "MetaMathQA" in task_names else task_names[0]
    initial_prefs = get_metric_preferences(initial_task)
    initial_x, initial_y = _TASK_PARETO_DEFAULTS.get(initial_task, (list(initial_prefs)[0], list(initial_prefs)[1]))

    with gr.Blocks() as demo:
        pareto_plot = gr.Plot(label="Pareto Frontier Plot")

        with gr.Row():
            metric_x_dropdown = gr.Dropdown(
                label="1st metric for Pareto plot",
                choices=list(initial_prefs.keys()),
                value=initial_x,
            )
            metric_y_dropdown = gr.Dropdown(
                label="2nd metric for Pareto plot",
                choices=list(initial_prefs.keys()),
                value=initial_y,
            )

        with gr.Row():
            task_dropdown = gr.Dropdown(
                label="Select Task",
                choices=task_names,
                value=initial_task,
            )
            model_dropdown = gr.Dropdown(label="Select Model ID", choices=get_model_ids(initial_task, df))

        # Make dataframe columns all equal in width so that they are good enough for numbers but don't
        # get hugely extended by columns like `train_config`.
        initial_filtered = filter_data(initial_task, get_model_ids(initial_task, df)[0], df)
        column_widths = ["150px" for _ in initial_filtered.columns]
        column2index = dict(zip(initial_filtered.columns, range(len(initial_filtered.columns))))
        column_widths[column2index["experiment_name"]] = "300px"

        def update_on_task(task_name):
            new_models = get_model_ids(task_name, df)
            filtered = filter_data(task_name, new_models[0] if new_models else "", df)

            prefs = get_metric_preferences(task_name)
            x_default, y_default = _TASK_PARETO_DEFAULTS.get(task_name, (list(prefs)[0], list(prefs)[1]))
            metric_choices = list(prefs.keys())

            return (
                gr.update(choices=new_models, value=new_models[0] if new_models else None),
                format_df(filtered),
                gr.update(choices=metric_choices, value=x_default),
                gr.update(choices=metric_choices, value=y_default),
            )

        task_dropdown.change(
            fn=update_on_task,
            inputs=[task_dropdown],
            outputs=[model_dropdown, metric_x_dropdown, metric_y_dropdown],
        )

        def update_pareto_plot(task_name, model_id, metric_x, metric_y, request: gr.Request):
            highlight_type = request.query_params.get("highlight[type]", "").lower()

            prefs = get_metric_preferences(task_name)
            filtered = filter_data(task_name, model_id, df)
            fig = generate_pareto_plot(filtered, metric_x, metric_y, prefs, highlight_type)
            return fig

        for comp in [model_dropdown, metric_x_dropdown, metric_y_dropdown]:
            comp.change(
                fn=update_pareto_plot,
                inputs=[task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown],
                outputs=[pareto_plot],
            )

        demo.load(
            fn=update_pareto_plot,
            inputs=[task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown],
            outputs=[pareto_plot],
        )

    return demo


base_dir = os.path.dirname(__file__)
_TASK_CONFIGS = {
    "MetaMathQA": os.path.join(base_dir, "MetaMathQA", "results"),
    "image-gen": os.path.join(base_dir, "image-gen", "results"),
}

df = load_task_results(_TASK_CONFIGS)
demo = build_app(df)
demo.launch(theme=gr.themes.Soft())
