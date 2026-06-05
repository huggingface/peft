# Copyright 2025-present the HuggingFace Inc. team.
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

"""Gradio app to show the results"""

import logging
import os
import tempfile

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from processing import (
    filter_data,
    get_model_ids,
    get_metric_preferences,
    _get_metric_explanation,
    _TASK_PARETO_DEFAULTS,
    compute_pareto_frontier,
    format_df,
    load_task_results,
)
from sanitizer import parse_and_filter


logger = logging.getLogger(__name__)


def generate_pareto_plot(df, metric_x, metric_y, metric_preferences):
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
            line={"color": "rgba(0,0,255,0.3)", "width": 4},
            name="Pareto Frontier",
        )
        fig.add_trace(line_trace)

    # Add non-frontier points in gray with semi-transparency.
    if not non_pareto_df.empty:
        non_frontier_trace = go.Scatter(
            x=non_pareto_df[metric_x],
            y=non_pareto_df[metric_y],
            mode="markers",
            marker={"color": "rgba(128,128,128,0.5)", "size": 12},
            hoverinfo="text",
            text=non_pareto_df.apply(
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
        fig.add_trace(non_frontier_trace)

    # Add Pareto frontier points with legend
    if not pareto_df.empty:
        pareto_scatter = px.scatter(
            pareto_df,
            x=metric_x,
            y=metric_y,
            color="experiment_name",
            hover_data={"experiment_name": True, "peft_type": True, metric_x: True, metric_y: True},
        )
        for trace in pareto_scatter.data:
            trace.marker = {"size": 12}
            fig.add_trace(trace)

    # Update layout with axes labels.
    fig.update_layout(
        title=f"Pareto Frontier for {metric_x} vs {metric_y}",
        template="seaborn",
        height=700,
        autosize=True,
        xaxis_title=metric_x,
        yaxis_title=metric_y,
    )

    return fig


def compute_pareto_summary(filtered, pareto_df, metric_x, metric_y):
    if filtered.empty:
        return "No data available."

    stats = filtered[[metric_x, metric_y]].agg(["min", "max", "mean"]).to_string()
    total_points = len(filtered)
    pareto_points = len(pareto_df)
    excluded_points = total_points - pareto_points
    summary_text = (
        f"{stats}\n\n"
        f"Total points: {total_points}\n"
        f"Pareto frontier points: {pareto_points}\n"
        f"Excluded points: {excluded_points}"
    )
    return summary_text


def export_csv(df):
    if df.empty:
        return None
    csv_data = df.to_csv(index=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp:
        tmp.write(csv_data)
        tmp_path = tmp.name
    return tmp_path


def build_app(df):
    task_names = sorted(df["task_name"].unique())
    initial_task = "MetaMathQA" if "MetaMathQA" in task_names else task_names[0]
    initial_prefs = get_metric_preferences(initial_task)
    initial_x, initial_y = _TASK_PARETO_DEFAULTS.get(initial_task, (list(initial_prefs)[0], list(initial_prefs)[1]))

    with gr.Blocks() as demo:
        gr.Markdown("# PEFT method comparison")
        gr.Markdown(
            "Find more information [on the PEFT GitHub repo](https://github.com/huggingface/peft/tree/main/method_comparison)"
        )

        # Hidden state to store the current filter query.
        filter_state = gr.State("")

        gr.Markdown("## Choose the task and base model")
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

        data_table = gr.DataFrame(
            label="Results",
            value=format_df(initial_filtered),
            interactive=False,
            max_chars=100,
            wrap=False,
            column_widths=column_widths,
        )

        with gr.Row():
            filter_textbox = gr.Textbox(
                label="Filter DataFrame",
                placeholder="Enter filter (e.g.: peft_type=='LORA')",
                interactive=True,
            )
            apply_filter_button = gr.Button("Apply Filter")
            reset_filter_button = gr.Button("Reset Filter")

        metric_explanation = gr.Markdown(
            _get_metric_explanation(initial_task),
        )

        gr.Markdown("## Pareto plot")
        gr.Markdown(
            "Select 2 criteria to plot the Pareto frontier. This will show the best PEFT methods along this axis and "
            "the trade-offs with the other axis. The PEFT methods that Pareto-dominate are shown in colors. All other "
            "methods are inferior with regard to these two metrics. Hover over a point to show details."
        )

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

        pareto_plot = gr.Plot(label="Pareto Frontier Plot")
        summary_box = gr.Textbox(label="Summary Statistics", lines=6)
        csv_output = gr.File(label="Export Filtered Data as CSV")

        def update_on_task(task_name, current_filter):
            new_models = get_model_ids(task_name, df)
            filtered = filter_data(task_name, new_models[0] if new_models else "", df)
            if current_filter.strip():
                try:
                    mask = parse_and_filter(filtered, current_filter)
                    df_queried = filtered[mask]
                    if not df_queried.empty:
                        filtered = df_queried
                except Exception as exc:
                    # invalid filter query
                    logger.debug("Ignoring invalid filter query: %s", exc)

            prefs = get_metric_preferences(task_name)
            x_default, y_default = _TASK_PARETO_DEFAULTS.get(task_name, (list(prefs)[0], list(prefs)[1]))
            metric_choices = list(prefs.keys())
            explanation = _get_metric_explanation(task_name)

            return (
                gr.update(choices=new_models, value=new_models[0] if new_models else None),
                format_df(filtered),
                gr.update(choices=metric_choices, value=x_default),
                gr.update(choices=metric_choices, value=y_default),
                explanation,
            )

        task_dropdown.change(
            fn=update_on_task,
            inputs=[task_dropdown, filter_state],
            outputs=[model_dropdown, data_table, metric_x_dropdown, metric_y_dropdown, metric_explanation],
        )

        def update_on_model(task_name, model_id, current_filter):
            filtered = filter_data(task_name, model_id, df)
            if current_filter.strip():
                try:
                    mask = parse_and_filter(filtered, current_filter)
                    filtered = filtered[mask]
                except Exception as exc:
                    logger.debug("Ignoring invalid filter query: %s", exc)
            return format_df(filtered)

        model_dropdown.change(
            fn=update_on_model, inputs=[task_dropdown, model_dropdown, filter_state], outputs=data_table
        )

        def update_pareto_plot_and_summary(task_name, model_id, metric_x, metric_y, current_filter):
            prefs = get_metric_preferences(task_name)
            filtered = filter_data(task_name, model_id, df)
            if current_filter.strip():
                try:
                    mask = parse_and_filter(filtered, current_filter)
                    filtered = filtered[mask]
                except Exception as e:
                    return generate_pareto_plot(filtered, metric_x, metric_y, prefs), f"Filter error: {e}"

            pareto_df = compute_pareto_frontier(filtered, metric_x, metric_y, prefs)
            fig = generate_pareto_plot(filtered, metric_x, metric_y, prefs)
            summary = compute_pareto_summary(filtered, pareto_df, metric_x, metric_y)
            return fig, summary

        for comp in [model_dropdown, metric_x_dropdown, metric_y_dropdown]:
            comp.change(
                fn=update_pareto_plot_and_summary,
                inputs=[task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown, filter_state],
                outputs=[pareto_plot, summary_box],
            )

        def apply_filter(filter_query, task_name, model_id, metric_x, metric_y):
            prefs = get_metric_preferences(task_name)
            filtered = filter_data(task_name, model_id, df)
            if filter_query.strip():
                try:
                    mask = parse_and_filter(filtered, filter_query)
                    filtered = filtered[mask]
                except Exception as e:
                    # Update the table, plot, and summary even if there is a filter error.
                    return (
                        filter_query,
                        filtered,
                        generate_pareto_plot(filtered, metric_x, metric_y, prefs),
                        f"Filter error: {e}",
                    )

            pareto_df = compute_pareto_frontier(filtered, metric_x, metric_y, prefs)
            fig = generate_pareto_plot(filtered, metric_x, metric_y, prefs)
            summary = compute_pareto_summary(filtered, pareto_df, metric_x, metric_y)
            return filter_query, format_df(filtered), fig, summary

        apply_filter_button.click(
            fn=apply_filter,
            inputs=[filter_textbox, task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown],
            outputs=[filter_state, data_table, pareto_plot, summary_box],
        )

        def reset_filter(task_name, model_id, metric_x, metric_y):
            prefs = get_metric_preferences(task_name)
            filtered = filter_data(task_name, model_id, df)
            pareto_df = compute_pareto_frontier(filtered, metric_x, metric_y, prefs)
            fig = generate_pareto_plot(filtered, metric_x, metric_y, prefs)
            summary = compute_pareto_summary(filtered, pareto_df, metric_x, metric_y)
            # Return empty strings to clear the filter state and textbox.
            return "", "", format_df(filtered), fig, summary

        reset_filter_button.click(
            fn=reset_filter,
            inputs=[task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown],
            outputs=[filter_state, filter_textbox, data_table, pareto_plot, summary_box],
        )

        gr.Markdown("## Export data")
        # Export button for CSV download.
        export_button = gr.Button("Export Filtered Data")
        export_button.click(
            fn=lambda task, model: export_csv(filter_data(task, model, df)),
            inputs=[task_dropdown, model_dropdown],
            outputs=csv_output,
        )

        demo.load(
            fn=update_pareto_plot_and_summary,
            inputs=[task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown, filter_state],
            outputs=[pareto_plot, summary_box],
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
