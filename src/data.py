"""Download, validate, and convert the Criteo uplift dataset to Parquet."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "criteo-uplift-v2.1.csv.gz"
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "processed" / "criteo_uplift.parquet"
DEFAULT_QUALITY = PROJECT_ROOT / "data" / "processed" / "data_quality.json"
DATA_URLS = (
    (
        "https://criteostorage.blob.core.windows.net/"
        "criteo-research-datasets/criteo-uplift-v2.1.csv.gz"
    ),
    (
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
        "criteo-research-uplift-v2.1.csv.gz?download=true"
    ),
)
FEATURES = [f"f{i}" for i in range(12)]
EXPECTED_COLUMNS = FEATURES + ["treatment", "conversion", "visit", "exposure"]
PROCESSED_COLUMNS = ["row_id"] + EXPECTED_COLUMNS
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]


def sql_path(path: Path) -> str:
    """Return a safely quoted path value for a DuckDB SQL string."""
    return path.resolve().as_posix().replace("'", "''")


def download_dataset(output: Path, force: bool = False) -> None:
    """Stream the official corrected dataset to disk without loading it in memory."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    partial = output.with_suffix(output.suffix + ".part")
    errors = []
    for url in DATA_URLS:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            partial.replace(output)
            break
        except Exception as error:
            partial.unlink(missing_ok=True)
            errors.append(f"{url}: {error}")
    else:
        raise RuntimeError("All download sources failed:\n" + "\n".join(errors))
    print(f"Downloaded dataset to {output} ({output.stat().st_size / 1e6:,.1f} MB)")


def csv_columns(connection: duckdb.DuckDBPyConnection, raw_path: Path) -> list[str]:
    query = "SELECT * FROM read_csv_auto(?, header = true, sample_size = 100000) LIMIT 0"
    cursor = connection.execute(query, [str(raw_path.resolve())])
    return [description[0] for description in cursor.description]


def validate_columns(columns: list[str]) -> None:
    missing = sorted(set(EXPECTED_COLUMNS) - set(columns))
    unexpected = sorted(set(columns) - set(EXPECTED_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "Unexpected dataset schema. "
            f"Missing columns: {missing or 'none'}; unexpected columns: {unexpected or 'none'}"
        )


def prepare_dataset(raw_path: Path, output: Path, quality_path: Path, force: bool = False) -> None:
    """Validate the compressed CSV, cast fields, and write compressed Parquet."""
    raw_path = raw_path.resolve()
    output = output.resolve()
    quality_path = quality_path.resolve()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run `python src/data.py download` first."
        )
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    output.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        columns = csv_columns(connection, raw_path)
        validate_columns(columns)

        feature_select = ",\n                ".join(
            f"CAST({feature} AS DOUBLE) AS {feature}" for feature in FEATURES
        )
        binary_select = ",\n                ".join(
            f"CAST({column} AS UTINYINT) AS {column}" for column in BINARY_COLUMNS
        )
        copy_sql = f"""
            COPY (
                SELECT
                    ROW_NUMBER() OVER () - 1 AS row_id,
                    {feature_select},
                    {binary_select}
                FROM read_csv_auto(
                    '{sql_path(raw_path)}',
                    header = true,
                    sample_size = 100000
                )
            ) TO '{sql_path(output)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
        connection.execute(copy_sql)

        null_expressions = ", ".join(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
            for column in PROCESSED_COLUMNS
        )
        null_row = connection.execute(
            f"SELECT {null_expressions} FROM read_parquet(?)", [str(output)]
        ).fetchone()
        row_count = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(output)]
        ).fetchone()[0]
        invalid_binary = {}
        for column in BINARY_COLUMNS:
            invalid_binary[column] = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet(?) WHERE {column} NOT IN (0, 1)",
                [str(output)],
            ).fetchone()[0]

        quality = {
            "source_file": str(raw_path),
            "processed_file": str(output),
            "rows": int(row_count),
            "columns": PROCESSED_COLUMNS,
            "null_counts": dict(zip(PROCESSED_COLUMNS, map(int, null_row))),
            "invalid_binary_counts": {key: int(value) for key, value in invalid_binary.items()},
        }
        quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    finally:
        connection.close()

    print(f"Created {output} with {row_count:,} rows")
    print(f"Wrote validation summary to {quality_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download the corrected v2.1 dataset")
    download.add_argument("--output", type=Path, default=DEFAULT_RAW)
    download.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare", help="Validate CSV and convert it to Parquet")
    prepare.add_argument("--input", type=Path, default=DEFAULT_RAW)
    prepare.add_argument("--output", type=Path, default=DEFAULT_PARQUET)
    prepare.add_argument("--quality-output", type=Path, default=DEFAULT_QUALITY)
    prepare.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "download":
            download_dataset(args.output, args.force)
        else:
            prepare_dataset(args.input, args.output, args.quality_output, args.force)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
