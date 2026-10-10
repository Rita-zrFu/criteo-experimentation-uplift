# Data

Raw and processed data are intentionally excluded from Git.

Expected raw file:

```text
data/raw/criteo-uplift-v2.1.csv.gz
```

Create it with `python src/data.py download`, or provide a local compressed CSV to `python src/data.py prepare --input /path/to/file.csv.gz`. The downloader tries Criteo's storage first and Criteo's Hugging Face dataset mirror second.

The pipeline creates:

```text
data/processed/criteo_uplift.parquet
data/processed/data_quality.json
```

Source: [Criteo AI Lab Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). The dataset is distributed under CC BY-NC-SA 4.0; review the official terms before use.
