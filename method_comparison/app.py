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

import os
import tempfile

import gradio as gr
import plotly.express as px

from processing import load_df


metric_preferences = {
    "cuda_memory_reserved_avg": "lower",
    "cuda_memory_max": "lower",
    "cuda_memory_reserved_99th": "lower",
    "total_time": "lower",
    "train_time": "lower",
    "file_size": "lower",
    "test_accuracy": "higher",
    "test_loss": "lower",
}


def get_model_ids(task_name, df):
    filtered = df[df["task_name"] == task_name]
    return sorted(filtered["model_id"].unique())


def filter_data(task_name, model_id, df):
    filtered = df[(df["task_name"] == task_name) & (df["model_id"] == model_id)]
    return filtered


# Compute the Pareto frontier for two selected metrics.
def compute_pareto_frontier(df, metric_x, metric_y):
    if df.empty:
        return df

    df = df.copy()
    points = df[[metric_x, metric_y]].values
    selected_indices = []

    def dominates(a, b, metric_x, metric_y):
        # Check for each metric whether b is as good or better than a
        if metric_preferences[metric_x] == "higher":
            cond_x = b[0] >= a[0]
            better_x = b[0] > a[0]
        else:
            cond_x = b[0] <= a[0]
            better_x = b[0] < a[0]
        if metric_preferences[metric_y] == "higher":
            cond_y = b[1] >= a[1]
            better_y = b[1] > a[1]
        else:
            cond_y = b[1] <= a[1]
            better_y = b[1] < a[1]
        return cond_x and cond_y and (better_x or better_y)

    for i, point in enumerate(points):
        dominated = False
        for j, other_point in enumerate(points):
            if i == j:
                continue
            if dominates(point, other_point, metric_x, metric_y):
                dominated = True
                break
        if not dominated:
            selected_indices.append(i)
    pareto_df = df.iloc[selected_indices]
    return pareto_df


def generate_pareto_plot(df, metric_x, metric_y):
    if df.empty:
        return {}
    fig = px.scatter(
        df,
        x=metric_x,
        y=metric_y,
        hover_data={"experiment_name": True, "peft_type": True, metric_x: True, metric_y: True},
        title=f"Pareto Frontier for {metric_x} vs {metric_y}",
        template="seaborn",
    )
    fig.update_traces(marker={"size": 12})
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
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PEFT method comparison")

        gr.Markdown("## Choose the task and base model")
        with gr.Row():
            task_dropdown = gr.Dropdown(
                label="Select Task",
                choices=sorted(df["task_name"].unique()),
                value=sorted(df["task_name"].unique())[0],
            )
            model_dropdown = gr.Dropdown(
                label="Select Model ID", choices=get_model_ids(sorted(df["task_name"].unique())[0], df)
            )

        data_table = gr.DataFrame(label="Results", value=df, interactive=False)

        gr.Markdown("## Pareto plot")
        gr.Markdown(
            "Select 2 criteria to plot the Pareto frontier. This will show the best PEFT methods along this axis and "
            "the trade-offs with the other axis."
        )

        with gr.Row():
            x_default = (
                "cuda_memory_max" if "cuda_memory_max" in metric_preferences else list(metric_preferences.keys())[0]
            )
            y_default = (
                "test_accuracy" if "test_accuracy" in metric_preferences else list(metric_preferences.keys())[1]
            )
            metric_x_dropdown = gr.Dropdown(
                label="1st metric for Pareto plot",
                choices=list(metric_preferences.keys()),
                value=x_default,
            )
            metric_y_dropdown = gr.Dropdown(
                label="2nd metric for Pareto plot",
                choices=list(metric_preferences.keys()),
                value=y_default,
            )

        pareto_plot = gr.Plot(label="Pareto Frontier Plot")
        summary_box = gr.Textbox(label="Summary Statistics", lines=6)
        csv_output = gr.File(label="Export Filtered Data as CSV")

        # When task is changed, update model_id dropdown choices and table.
        def update_on_task(task_name):
            new_models = get_model_ids(task_name, df)
            filtered = filter_data(task_name, new_models[0] if new_models else "", df)
            return gr.update(choices=new_models, value=new_models[0] if new_models else None), filtered

        task_dropdown.change(fn=update_on_task, inputs=task_dropdown, outputs=[model_dropdown, data_table])

        # When model_id changes, update the table.
        def update_on_model(task_name, model_id):
            filtered = filter_data(task_name, model_id, df)
            return filtered

        model_dropdown.change(fn=update_on_model, inputs=[task_dropdown, model_dropdown], outputs=data_table)

        # Update Pareto plot and summary when metrics or filtering change.
        def update_pareto_plot_and_summary(task_name, model_id, metric_x, metric_y):
            filtered = filter_data(task_name, model_id, df)
            pareto_df = compute_pareto_frontier(filtered, metric_x, metric_y)
            fig = generate_pareto_plot(pareto_df, metric_x, metric_y)
            summary = compute_pareto_summary(filtered, pareto_df, metric_x, metric_y)
            return fig, summary

        inputs = [task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown]
        for comp in [model_dropdown, metric_x_dropdown, metric_y_dropdown]:
            comp.change(fn=update_pareto_plot_and_summary, inputs=inputs, outputs=[pareto_plot, summary_box])

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
            inputs=[task_dropdown, model_dropdown, metric_x_dropdown, metric_y_dropdown],
            outputs=[pareto_plot, summary_box],
        )

    return demo


# TODO only 1 task, using temporary results for now
path = os.path.join(os.path.dirname(__file__), "MetaMathQA", "temporary_results")
df = load_df(path, task_name="MetaMathQA")
demo = build_app(df)
demo.launch()
