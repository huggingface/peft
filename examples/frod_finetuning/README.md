# FRoD fine-tuning examples

These examples show minimal FRoD fine-tuning with the Transformers `Trainer`.

Install the example dependencies and run either script directly:

```bash
pip install -r examples/frod_finetuning/requirements.txt
python examples/frod_finetuning/frod_text_classification.py
python examples/frod_finetuning/frod_image_classification.py
```

The text example fine-tunes `google-bert/bert-base-uncased` on `nyu-mll/glue` with the `sst2` configuration. The image
example fine-tunes `google/vit-base-patch16-224` on the train and test parquet splits from `tanganke/stanford_cars`.

Both scripts use separate optimizer learning rates for FRoD diagonal coefficients, FRoD sparse coefficients, and the
classification head. FRoD dropout is set to `0.0` because the sparse rotational parameterization is the main
regularizer in these examples.

To use local mirrors of the image model or dataset, override the image example paths with environment variables:

```bash
FROD_IMAGE_MODEL_NAME=/path/to/local/vit-model \
FROD_STANFORD_CARS_DATA_DIR=/path/to/local/stanford_cars \
FROD_IMAGE_OUTPUT_DIR=vit-local-frod-stanford-cars \
python examples/frod_finetuning/frod_image_classification.py
```
