# FRoD fine-tuning examples

These examples show minimal FRoD fine-tuning with the Transformers `Trainer`.

Install the example dependencies and run either script directly:

```bash
pip install -r examples/frod_finetuning/requirements.txt
python examples/frod_finetuning/frod_text_classification.py
python examples/frod_finetuning/frod_image_classification.py
```

The text example fine-tunes `google-bert/bert-base-uncased` on `nyu-mll/glue` with the `sst2` configuration. The image
example fine-tunes `openai/clip-vit-base-patch32` on the train and test parquet splits from `tanganke/stanford_cars`.

Both scripts use separate optimizer learning rates for FRoD diagonal coefficients, FRoD sparse coefficients, and the
classification head. FRoD dropout is set to `0.0` because the sparse rotational parameterization is the main
regularizer in these examples.

To use local mirrors of the image model or dataset, pass the paths as CLI arguments:

```bash
python examples/frod_finetuning/frod_image_classification.py \
  --model_name_or_path /path/to/local/clip-vit-model \
  --data_dir /path/to/local/stanford_cars \
  --output_dir clip-vit-local-frod-stanford-cars
```
