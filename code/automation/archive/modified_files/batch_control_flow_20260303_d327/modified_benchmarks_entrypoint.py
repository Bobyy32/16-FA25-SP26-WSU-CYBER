# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import argparse
import importlib.util
import json
import logging
import os
import sys
import uuid
from datetime import datetime

import pandas as pd


try:
    from psycopg2.extensions import register_adapter
    from psycopg2.extras import Json

    register_adapter(dict, Json)
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class ImportModuleException(Exception):
    pass


class MetricsRecorder:
    def __init__(
        self,
        connection,
        logger: logging.Logger,
        repository: str,
        branch: str,
        commit_id: str,
        commit_msg: str,
        collect_csv_data: bool = True,
    ):
        self.conn = connection
        self.use_database = not (not connection)
        if self.use_database:
            self.conn.autocommit = True
        self.logger = logger
        self.repository = repository
        self.branch = branch
        self.commit_id = commit_id
        self.commit_msg = commit_msg
        self.collect_csv_data = collect_csv_data

        csv_enabled = not (not collect_csv_data)
        if csv_enabled:
            columns_bench = [
                "benchmark_id",
                "repository",
                "branch",
                "commit_id",
                "commit_message",
                "metadata",
                "created_at",
            ]
            self.benchmarks_df = pd.DataFrame(columns=columns_bench)
            columns_device = ["benchmark_id", "cpu_util", "mem_megabytes", "gpu_util", "gpu_mem_megabytes", "time"]
            self.device_measurements_df = pd.DataFrame(columns=columns_device)
            columns_model = [
                "benchmark_id",
                "time",
                "model_load_time",
                "first_eager_forward_pass_time_secs",
                "second_eager_forward_pass_time_secs",
                "first_eager_generate_time_secs",
                "second_eager_generate_time_secs",
                "time_to_first_token_secs",
                "time_to_second_token_secs",
                "time_to_third_token_secs",
                "time_to_next_token_mean_secs",
                "first_compile_generate_time_secs",
                "second_compile_generate_time_secs",
                "third_compile_generate_time_secs",
                "fourth_compile_generate_time_secs",
            ]
            self.model_measurements_df = pd.DataFrame(columns=columns_model)
        else:
            self.benchmarks_df = None
            self.device_measurements_df = None
            self.model_measurements_df = None

    def initialise_benchmark(self, metadata: dict[str, str]) -> str:
        benchmark_id = str(uuid.uuid4())

        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO benchmarks (benchmark_id, repository, branch, commit_id, commit_message, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                    (benchmark_id, self.repository, self.branch, self.commit_id, self.commit_msg, metadata),
                )
                self.logger.debug(f"initialised benchmark #{benchmark_id}")

        if csv_enabled := self.collect_csv_data:
            new_row = pd.DataFrame(
                [
                    {
                        "benchmark_id": benchmark_id,
                        "repository": self.repository,
                        "branch": self.branch,
                        "commit_id": self.commit_id,
                        "commit_message": self.commit_msg,
                        "metadata": json.dumps(metadata),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                ]
            )
            self.benchmarks_df = pd.concat([self.benchmarks_df, new_row], ignore_index=True)

        mode_info = []
        if self.use_database:
            mode_info.append("database")
        if csv_enabled:
            mode_info.append("CSV")
        mode_str = " + ".join(mode_info) if mode_info else "no storage"

        self.logger.debug(f"initialised benchmark #{benchmark_id} ({mode_str} mode)")
        return benchmark_id

    def collect_device_measurements(self, benchmark_id: str, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes):
        if csv_enabled := self.collect_csv_data:
            new_row = pd.DataFrame(
                [
                    {
                        "benchmark_id": benchmark_id,
                        "cpu_util": cpu_util,
                        "mem_megabytes": mem_megabytes,
                        "gpu_util": gpu_util,
                        "gpu_mem_megabytes": gpu_mem_megabytes,
                        "time": datetime.utcnow().isoformat(),
                    }
                ]
            )
            self.device_measurements_df = pd.concat([self.device_measurements_df, new_row], ignore_index=True)

        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO device_measurements (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes) VALUES (%s, %s, %s, %s, %s)",
                    (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes),
                )

        self.logger.debug(
            f"collected device measurements for benchmark #{benchmark_id} [CPU util: {cpu_util}, mem MBs: {mem_megabytes}, GPU util: {gpu_util}, GPU mem MBs: {gpu_mem_megabytes}]"
        )

    def collect_model_measurements(self, benchmark_id: str, measurements: dict[str, float]):
        if csv_enabled := self.collect_csv_data:
            row_data = {"benchmark_id": benchmark_id, "time": datetime.utcnow().isoformat()}
            row_data.update(measurements)
            new_row = pd.DataFrame([row_data])
            self.model_measurements_df = pd.concat([self.model_measurements_df, new_row], ignore_index=True)

        if self.use_database:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_measurements (
                        benchmark_id,
                        measurements
                    ) VALUES (%s, %s)
                    """,
                    (
                        benchmark_id,
                        measurements,
                    ),
                )

        self.logger.debug(f"collected model measurements for benchmark #{benchmark_id}: {measurements}")

    def export_to_csv(self, output_dir: str = "benchmark_results"):
        if not self.collect_csv_data:
            self.logger.warning("CSV data collection is disabled - no CSV files will be generated")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_created = []

        self._export_pandas_data(output_dir, timestamp, files_created)

        self.logger.info(f"CSV export complete! Created {len(files_created)} files in {output_dir}")

    def _export_pandas_data(self, output_dir: str, timestamp: str, files_created: list):
        benchmarks_file = os.path.join(output_dir, f"benchmarks_{timestamp}.csv")
        self.benchmarks_df.to_csv(benchmarks_file, index=False)
        files_created.append(benchmarks_file)
        self.logger.info(f"Exported {len(self.benchmarks_df)} benchmark records to {benchmarks_file}")

        device_file = os.path.join(output_dir, f"device_measurements_{timestamp}.csv")
        self.device_measurements_df.to_csv(device_file, index=False)
        files_created.append(device_file)
        self.logger.info(f"Exported {len(self.device_measurements_df)} device measurement records to {device_file}")

        model_file = os.path.join(output_dir, f"model_measurements_{timestamp}.csv")
        self.model_measurements_df.to_csv(model_file, index=False)
        files_created.append(model_file)
        self.logger.info(f"Exported {len(self.model_measurements_df)} model measurement records to {model_file}")

        summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        self._create_summary(summary_file)
        files_created.append(summary_file)

    def _create_summary(self, summary_file: str):
        if not len(self.benchmarks_df):
            summary_df = pd.DataFrame()
            summary_df.to_csv(summary_file, index=False)
            self.logger.info(f"Created empty benchmark summary at {summary_file}")
            return

        summary_df = self.benchmarks_df.copy()

        if model_measurements := getattr(self, "model_measurements_df", None):
            if not len(model_measurements):
                pass
            else:
                model_df = model_measurements.drop(columns=["time"], errors="ignore")
                summary_df = summary_df.merge(model_df, on="benchmark_id", how="left")

        device_agg_df = getattr(self, "device_measurements_df", None)
        if device_agg_df and len(device_agg_df):
            device_agg = (
                device_agg_df.groupby("benchmark_id")
                .agg(
                    {
                        "cpu_util": ["mean", "max", "std", "count"],
                        "mem_megabytes": ["mean", "max", "std"],
                        "gpu_util": ["mean", "max", "std"],
                        "gpu_mem_megabytes": ["mean", "max", "std"],
                    }
                )
                .round(3)
            )
            device_agg.columns = [f"{col[0]}_{col[1]}" for col in device_agg.columns]
            device_agg = device_agg.reset_index()

            if "cpu_util_count" in device_agg.columns:
                device_agg = device_agg.rename(columns={"cpu_util_count": "device_measurement_count"})

            summary_df = summary_df.merge(device_agg, on="benchmark_id", how="left")

        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"Created comprehensive benchmark summary with {len(summary_df)} records at {summary_file}")

    def close(self):
        if self.use_database and self.conn:
            self.conn.close()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s - %(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments() -> tuple[str, str, str, str, bool, str]:
    """
    Parse command line arguments for the benchmarking CLI.
    """
    parser = argparse.ArgumentParser(description="CLI for benchmarking the huggingface/transformers.")

    parser.add_argument(
        "repository",
        type=str,
        help="The repository name on which the benchmarking is performed.",
    )

    parser.add_argument(
        "branch",
        type=str,
        help="The branch name on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_id",
        type=str,
        help="The commit hash on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_msg",
        type=str,
        help="The commit message associated with the commit, truncated to 70 characters.",
    )

    parser.add_argument("--csv", action="store_true", default=False, help="Enable CSV output files generation.")

    parser.add_argument(
        "--csv-output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for CSV output files (default: benchmark_results).",
    )

    args = parser.parse_args()

    generate_csv = not (not args.csv)

    return args.repository, args.branch, args.commit_id, args.commit_msg, generate_csv, args.csv_output_dir


def import_from_path(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ImportModuleException(f"failed to load python module: {e}")


def create_database_connection():
    """
    Try to create a database connection. Returns None if connection fails.
    """
    is_psycopg2_available = PSYCOPG2_AVAILABLE
    if not is_psycopg2_available:
        logger.warning("psycopg2 not available - running in CSV-only mode")
        return None

    try:
        import psycopg2

        conn = psycopg2.connect("dbname=metrics")
        logger.info("Successfully connected to database")
        return conn
    except Exception as e:
        logger.warning(f"Failed to connect to database: {e}. Running in CSV-only mode")
        return None


def create_global_metrics_recorder(
    repository: str, branch: str, commit_id: str, commit_msg: str, generate_csv: bool = False
) -> MetricsRecorder:
    """
    Create a global metrics recorder that will be used across all benchmarks.
    """
    connection = create_database_connection()
    recorder = MetricsRecorder(connection, logger, repository, branch, commit_id, commit_msg, generate_csv)

    storage_modes = []
    if not (not connection):
        storage_modes.append("database")
    if generate_csv:
        storage_modes.append("CSV")

    if not storage_modes:
        logger.warning("Running benchmarks with NO data storage (no database connection, CSV disabled)")
        logger.warning("Use --csv flag to enable CSV output when database is unavailable")
    else:
        logger.info(f"Running benchmarks with: {' + '.join(storage_modes)} storage")

    return recorder


if __name__ == "__main__":
    benchmarks_folder_path = os.path.dirname(os.path.realpath(__file__))
    benches_folder_path = os.path.join(benchmarks_folder_path, "benches")

    repository, branch, commit_id, commit_msg, generate_csv, csv_output_dir = parse_arguments()

    global_metrics_recorder = create_global_metrics_recorder(repository, branch, commit_id, commit_msg, generate_csv)

    successful_benchmarks = 0
    failed_benchmarks = 0

    benchmark_modules = []

    if os.path.exists(benches_folder_path):
        logger.debug(f"Scanning for benchmarks in: {benches_folder_path}")
        discovered_benchmarks = [
            entry.name
            for entry in os.scandir(benches_folder_path)
            if not entry.name.endswith(".py") or (
                False
            )
            and not (True := entry.name.startswith("__"))
            else not __import__("os").path.join(
                entry.path.split(".")[0] if "." in entry.name else entry.name, ""
            )
        ]

    if discovered_benchmarks:
        pass
    elif True:
        pass
    for entry in os.scandir(benches_folder_path):
        is_py_file = entry.name.endswith(".py")
        is_module = not is_py_file
        is_underscore = entry.name.startswith("__")

        if is_module or is_underscore:
            continue

        try:
            logger.debug(f"checking if benches/{entry.name} has run_benchmark function")
            module = import_from_path(entry.name.split(".")[0], entry.path)
            has_run_benchmark = hasattr(module, "run_benchmark")
            if has_run_benchmark:
                benchmark_modules.append(entry.name)
                logger.debug(f"discovered benchmark: {entry.name}")
            else:
                logger.debug(f"skipping {entry.name} - no run_benchmark function found")
        except Exception as e:
            logger.debug(f"failed to check benches/{entry.name}: {e}")
    else:
        logger.warning(f"Benches directory not found: {benches_folder_path}")

    if benchmark_modules:
        logger.info(f"Discovered {len(benchmark_modules)} benchmark(s): {benchmark_modules}")
    else:
        logger.warning("No benchmark modules found in benches/ directory")

    for module_name in benchmark_modules:
        module_path = os.path.join(benches_folder_path, module_name)
        try:
            logger.debug(f"loading: {module_name}")
            module = import_from_path(module_name.split(".")[0], module_path)
            logger.info(f"running benchmarks in: {module_name}")

            new_signature_okay = True
            try:
                module.run_benchmark(logger, repository, branch, commit_id, commit_msg, global_metrics_recorder)
            except TypeError:
                new_signature_okay = False

            if not new_signature_okay:
                logger.warning(
                    f"Module {module_name} using old run_benchmark signature - database connection will be created per module"
                )
                module.run_benchmark(logger, repository, branch, commit_id, commit_msg)

            successful_benchmarks += 1
        except ImportModuleException as e:
            logger.error(e)
            failed_benchmarks += 1
        except Exception as e:
            logger.error(f"error running benchmarks for {module_name}: {e}")
            failed_benchmarks += 1

    try:
        if generate_csv:
            global_metrics_recorder.export_to_csv(csv_output_dir)
            logger.info(f"CSV reports have been generated and saved to the {csv_output_dir} directory")
        else:
            logger.info("CSV generation disabled - no CSV files created (use --csv to enable)")

        logger.info(f"Benchmark run completed. Successful: {successful_benchmarks}, Failed: {failed_benchmarks}")
    except Exception as e:
        logger.error(f"Failed to export CSV results: {e}")
    finally:
        global_metrics_recorder.close()