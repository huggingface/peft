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

import functools
import json
import logging
import os
import re
import tempfile
from io import BytesIO
from typing import Any

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from PIL import Image
from processing import (
    filter_data,
    get_model_ids,
    get_metric_preferences,
    get_task_columns,
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


def format_df(df):
    return df.style.format(precision=3, thousands=",", decimal=".")


IMAGE_GEN_TASK = "image-gen"
SAMPLE_IMAGE_BUCKET = "peft-internal-testing/image-gen-benchmark"
SAMPLE_IMAGE_BUCKET_DIR = f"hf://buckets/{SAMPLE_IMAGE_BUCKET}/sample-images/results"
SAMPLE_IMAGE_BUCKET_URL = f"https://huggingface.co/buckets/{SAMPLE_IMAGE_BUCKET}"
GENERATED_VIEW = "Generated samples"
DATASET_VIEW = "Training dataset"


def _load_default_train_config_image_gen() -> dict[str, Any]:
    # The default training params define the prompts and dataset used by the benchmark; load them once
    # to caption generated images and to show the dataset images before an experiment is selected.
    path = os.path.join(os.path.dirname(__file__), "image-gen", "default_training_params.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Could not load default training params from %r: %s", path, exc)
        return {}


DEFAULT_TRAIN_CONFIG_IMAGE_GEN = _load_default_train_config_image_gen()
SAMPLE_IMAGE_PROMPTS = DEFAULT_TRAIN_CONFIG_IMAGE_GEN.get("sample_image_prompts", [])


@functools.lru_cache(maxsize=1)
def _get_bucket_fs() -> HfFileSystem:
    # Anonymous read access to the public bucket. The listing cache is disabled so that newly uploaded sample images
    # show up on a page refresh without having to redeploy the app.
    return HfFileSystem(use_listings_cache=False)


def get_sample_images(experiment_name: str) -> list[tuple[Image.Image, str]]:
    """Fetch the sample images of an experiment from the storage bucket.

    Returns a list of (PIL image, caption) tuples suitable for a gr.Gallery, or an empty list if no images are
    found. Each image is captioned with the prompt that was used to generate it.
    """
    stem = experiment_name.replace("/", "--")
    fs = _get_bucket_fs()
    try:
        paths = sorted(fs.glob(f"{SAMPLE_IMAGE_BUCKET_DIR}/{stem}_*.png"))
    except Exception as exc:
        logger.warning("Could not list sample images for %r: %s", experiment_name, exc)
        return []

    gallery = []
    for path in paths:
        try:
            with fs.open(path, "rb") as f:
                image = Image.open(BytesIO(f.read()))
                image.load()
        except Exception as exc:
            logger.warning("Could not load sample image %r: %s", path, exc)
            continue
        match = re.search(r"_(\d+)\.png$", path)
        prompt_idx = int(match.group(1)) - 1 if match else len(gallery)
        caption = (
            SAMPLE_IMAGE_PROMPTS[prompt_idx]
            if 0 <= prompt_idx < len(SAMPLE_IMAGE_PROMPTS)
            else os.path.basename(path)
        )
        gallery.append((image, caption))
    return gallery


@functools.lru_cache(maxsize=1)
def _load_dataset_images(dataset_id: str, split: str, image_column: str) -> list[Image.Image]:
    ds = load_dataset(dataset_id, split=split)
    images = []
    for image in ds[image_column]:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        images.append(image.convert("RGB"))
    return images


def get_dataset_images(config: dict[str, Any]) -> list[tuple[Image.Image, str]]:
    """Fetch the training dataset images for a training configuration from the Hugging Face Hub."""
    dataset_id = config.get("dataset_id") if config else None
    if not dataset_id:
        return []
    split = config.get("dataset_split", "train")
    image_column = config.get("image_column", "image")
    try:
        images = _load_dataset_images(dataset_id, split, image_column)
    except Exception as exc:
        logger.warning("Could not load dataset images for %r: %s", dataset_id, exc)
        return []

    prompts = config.get("instance_prompts", [])
    if isinstance(prompts, str):
        prompts = [prompts] * len(images)
    gallery = []
    for idx, image in enumerate(images):
        gallery.append((image, prompts[idx]))
    return gallery


def render_image_gallery(image_view, selected):
    """Return a gallery update with the contents for the selected experiment and image source view.

    The dataset view falls back to the default dataset when no experiment is selected, so its images can be shown before
    the user clicks a row. When generated samples are shown, the gallery label names the selected experiment.
    """
    if image_view == DATASET_VIEW:
        if selected:
            try:
                config = json.loads(selected["train_config"])
            except (TypeError, ValueError):
                config = {}
        else:
            config = DEFAULT_TRAIN_CONFIG_IMAGE_GEN
        return gr.update(value=get_dataset_images(config), label="Images")
    if not selected:
        return gr.update(value=None, label="Images")
    return gr.update(
        value=get_sample_images(selected["experiment_name"]),
        label=f"Generated samples for {selected['experiment_name']}",
    )


def load_gallery_deferred(task_name, image_view, selected):
    """Populate the image gallery in a chained event.

    Fetching the images can take a while, so the event handlers that update multiple components only clear the
    gallery and the images are loaded here in a follow-up event. Otherwise, the other components (e.g. the results
    table) would not be updated until the images are loaded.
    """
    if task_name != IMAGE_GEN_TASK:
        return gr.update()
    return render_image_gallery(image_view, selected)


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
        # Hidden state to store the experiment selected for the image gallery.
        selected_state = gr.State(None)

        gr.Markdown("## Choose the task and base model")
        with gr.Row():
            task_dropdown = gr.Dropdown(
                label="Select Task",
                choices=task_names,
                value=initial_task,
            )
            model_dropdown = gr.Dropdown(label="Select Model ID", choices=get_model_ids(initial_task, df))

        task_info = gr.Markdown(_get_task_info(initial_task))

        # Make dataframe columns all equal in width so that they are good enough for numbers but don't get hugely
        # extended by columns like `train_config`. Tasks can have different column counts, so size the widths to the
        # widest task; experiment_name is always the first column (see _TASK_IMPORTANT_COLUMNS) and holds long names, so
        # it gets extra width.
        initial_filtered = filter_data(initial_task, get_model_ids(initial_task, df)[0], df)
        num_columns = max(len(get_task_columns(task)) for task in task_names)
        column_widths = ["150px"] * num_columns
        column_widths[0] = "300px"

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

        with gr.Group(visible=initial_task == IMAGE_GEN_TASK) as sample_images_group:
            gr.Markdown("## Images")
            gr.Markdown(
                "The training dataset images are shown by default. Click a row in the results table above to see the "
                "sample images generated by that experiment, and use the selector to switch between the generated "
                "samples and the training dataset. Each image is captioned with its prompt. The generated images are "
                f"stored in [this bucket]({SAMPLE_IMAGE_BUCKET_URL})."
            )
            image_view_radio = gr.Radio(
                choices=[GENERATED_VIEW, DATASET_VIEW],
                value=DATASET_VIEW,
                label="Image source",
            )
            # The gallery starts empty and is populated by load_gallery_deferred on page load so that fetching the
            # images doesn't block the app startup.
            sample_gallery = gr.Gallery(
                label="Images",
                value=None,
                columns=3,
                object_fit="contain",
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

            is_image_gen = task_name == IMAGE_GEN_TASK
            return (
                gr.update(choices=new_models, value=new_models[0] if new_models else None),
                _get_task_info(task_name),
                format_df(filtered),
                gr.update(choices=metric_choices, value=x_default),
                gr.update(choices=metric_choices, value=y_default),
                explanation,
                gr.update(visible=is_image_gen),
                gr.update(value=DATASET_VIEW),
                gr.update(value=None, label="Images"),
                None,
            )

        task_dropdown.change(
            fn=update_on_task,
            inputs=[task_dropdown, filter_state],
            outputs=[
                model_dropdown,
                task_info,
                data_table,
                metric_x_dropdown,
                metric_y_dropdown,
                metric_explanation,
                sample_images_group,
                image_view_radio,
                sample_gallery,
                selected_state,
            ],
        ).then(
            fn=load_gallery_deferred,
            inputs=[task_dropdown, image_view_radio, selected_state],
            outputs=sample_gallery,
        )

        def update_on_model(task_name, model_id, current_filter):
            filtered = filter_data(task_name, model_id, df)
            if current_filter.strip():
                try:
                    mask = parse_and_filter(filtered, current_filter)
                    filtered = filtered[mask]
                except Exception as exc:
                    logger.debug("Ignoring invalid filter query: %s", exc)
            return format_df(filtered), gr.update(value=DATASET_VIEW), gr.update(value=None, label="Images"), None

        model_dropdown.change(
            fn=update_on_model,
            inputs=[task_dropdown, model_dropdown, filter_state],
            outputs=[data_table, image_view_radio, sample_gallery, selected_state],
        ).then(
            fn=load_gallery_deferred,
            inputs=[task_dropdown, image_view_radio, selected_state],
            outputs=sample_gallery,
        )

        def show_sample_images(task_name, model_id, evt: gr.SelectData):
            if task_name != IMAGE_GEN_TASK or evt.index is None:
                return None, gr.update(), gr.update()
            # Look up the clicked row by its experiment name (always the first column) instead of by the row index:
            # sorting the table happens client-side only, so the row index refers to the displayed order, not the order
            # of the dataframe on the server.
            experiment_name = evt.row_value[0]
            rows = filter_data(task_name, model_id, df)
            rows = rows[rows["experiment_name"] == experiment_name]
            if rows.empty:
                return None, gr.update(), gr.update()
            row = rows.iloc[0]
            selected = {"experiment_name": row["experiment_name"], "train_config": row["train_config"]}
            # Clicking a row switches the view to the experiment's generated samples.
            return selected, gr.update(value=GENERATED_VIEW), render_image_gallery(GENERATED_VIEW, selected)

        data_table.select(
            fn=show_sample_images,
            inputs=[task_dropdown, model_dropdown],
            outputs=[selected_state, image_view_radio, sample_gallery],
        )

        def update_image_view(image_view, selected):
            return render_image_gallery(image_view, selected)

        # Use the input event (user-only) so the programmatic radio updates above don't re-trigger this.
        image_view_radio.input(
            fn=update_image_view,
            inputs=[image_view_radio, selected_state],
            outputs=sample_gallery,
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
        demo.load(
            fn=load_gallery_deferred,
            inputs=[task_dropdown, image_view_radio, selected_state],
            outputs=sample_gallery,
        )

    return demo


_METRIC_EXPLANATIONS = {
    "MetaMathQA": (
        "*forgetting: This is the reduction in CE loss on a sample of Wikipedia data and reflects how much the "
        "model 'forgot' during training. The lower the number, the better."
    ),
    "image-gen": (
        "*drift: This measures how much the generated images drift from the base model's outputs on unrelated "
        "prompts, reflecting how much the model 'forgot' during training. The lower the number, the better."
    ),
}


def _get_metric_explanation(task_name):
    return _METRIC_EXPLANATIONS.get(task_name, "")


_TASK_DESCRIPTIONS = {
    "MetaMathQA": (
        "Trains on the MetaMathQA dataset and validates/tests on GSM8K, comparing how well PEFT methods teach "
        "mathematical chain-of-thought reasoning."
    ),
    "image-gen": (
        "DreamBooth-style fine-tuning on a "
        "[cat plushy dataset](https://huggingface.co/datasets/peft-internal-testing/cat-image-dataset) image dataset."
    ),
}

_TASK_CHECKPOINT_URLS = {
    "MetaMathQA": "https://huggingface.co/buckets/peft-internal-testing/metamathqa-checkpoints",
    "image-gen": "https://huggingface.co/buckets/peft-internal-testing/image-gen-benchmark/tree/checkpoints",
}


def _get_task_info(task_name):
    description = _TASK_DESCRIPTIONS.get(task_name, "")
    url = _TASK_CHECKPOINT_URLS.get(task_name)
    if url:
        description = f"{description} The trained PEFT checkpoints are available in [this bucket]({url})."
    return description


base_dir = os.path.dirname(__file__)
_TASK_CONFIGS = {
    "MetaMathQA": os.path.join(base_dir, "MetaMathQA", "results"),
    "image-gen": os.path.join(base_dir, "image-gen", "results"),
}

df = load_task_results(_TASK_CONFIGS)
demo = build_app(df)
demo.launch(theme=gr.themes.Soft())
