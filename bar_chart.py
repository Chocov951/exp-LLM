from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_FILE = Path("results_trec20.csv")
OUTPUT_TIME = Path("bar_time_comparison.png")
OUTPUT_ENERGY_CO2 = Path("bar_energy_co2_comparison.png")
OUTPUT_SIDE_BY_SIDE = Path("bar_side_by_side_comparison.png")

BM25_VALUES = [100, 200, 300, 500]

COLOR_DIRECT = "#D79B00"
# COLOR_FILTER = "#82B366"
COLOR_FILTER = "#B4D1A3"
# COLOR_RERANK = "#6C8EBF"
COLOR_RERANK = "#82B366"
COLOR_CO2_DIRECT = "#B85450"
COLOR_CO2_BERT = "#6C8EBF"
# COLOR_CO2_BERT = "#9673A6"

FONT_TITLE = 26
FONT_LABEL = 24
FONT_TICK = 20
FONT_LEGEND = 20
FONT_TITLE_SIDE = 34
FONT_LABEL_SIDE = 28
FONT_TICK_SIDE = 24
FONT_LEGEND_SIDE = 24


def _prepare_data(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)

	bert_mask = (
		(df["filter_method"] == "BERT-bge-m3")
		& (df["rerank_method"] == "Listwise-qwen30-Wfull-S10")
		& (df["filter_size"] == 30)
		& (df["bm25_k"].isin(BM25_VALUES))
	)

	direct_mask = (
		(df["filter_method"] == "Direct")
		& (df["rerank_method"] == "Listwise-qwen30-W20-S10")
		& (df["bm25_k"].isin(BM25_VALUES))
	)

	selected = df[bert_mask | direct_mask].copy()
	if selected.empty:
		raise ValueError("No matching rows found for requested configurations.")

	# Normalize into a comparable model label.
	selected["pipeline"] = np.where(
		selected["filter_method"] == "Direct",
		"Direct + qwen30",
		"BERT-bge-m3 + qwen30 + fs=30",
	)

	needed_rows = len(BM25_VALUES) * 2
	if len(selected) != needed_rows:
		raise ValueError(
			f"Expected {needed_rows} rows (2 pipelines x {len(BM25_VALUES)} bm25_k), got {len(selected)}."
		)

	selected = selected.sort_values(["bm25_k", "pipeline"]).reset_index(drop=True)
	return selected


def _series_for_pipeline(df: pd.DataFrame, pipeline: str, column: str) -> np.ndarray:
	subset = df[df["pipeline"] == pipeline].set_index("bm25_k").reindex(BM25_VALUES)
	return subset[column].to_numpy(dtype=float)


def plot_time(df: pd.DataFrame, output_path: Path) -> None:
	x = np.arange(len(BM25_VALUES))
	width = 0.36

	bert_filter = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_filter_time"
	)
	bert_rerank = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_ranking_time"
	)
	direct_time = _series_for_pipeline(df, "Direct + qwen30", "avg_total_time")

	fig, ax = plt.subplots(figsize=(10, 8))

	ax.bar(
		x - width / 2,
		bert_filter,
		width,
		label="BERT filter time",
		color=COLOR_FILTER,
	)
	ax.bar(
		x - width / 2,
		bert_rerank,
		width,
		bottom=bert_filter,
		label="Qwen3-30B rerank time",
		color=COLOR_RERANK,
	)
	ax.bar(
		x + width / 2,
		direct_time,
		width,
		label="Sliding Window time",
		color=COLOR_DIRECT,
	)

	ax.set_title(
		"Time Comparison",
		fontsize=FONT_TITLE,
	)
	ax.set_xlabel("bm25_k", fontsize=FONT_LABEL)
	ax.set_ylabel("Time (s)", fontsize=FONT_LABEL)
	max_time = max(np.nanmax(bert_filter + bert_rerank), np.nanmax(direct_time))
	upper_time = max(5, int(np.ceil(max_time / 5.0) * 5))
	ax.set_ylim(0, upper_time)
	ax.set_yticks(np.arange(0, upper_time + 1, 5))
	ax.set_xticks(x)
	ax.set_xticklabels([str(v) for v in BM25_VALUES])
	ax.tick_params(axis="both", labelsize=FONT_TICK)
	ax.grid(axis="y", alpha=0.3)
	ax.legend(loc="upper left", fontsize=FONT_LEGEND)

	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def plot_energy_and_co2(df: pd.DataFrame, output_path: Path) -> None:
	x = np.arange(len(BM25_VALUES))
	width = 0.36

	bert_filter_cons = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_filter_consumption_wh"
	)
	bert_rerank_cons = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_ranking_consumption_wh"
	)
	direct_cons = _series_for_pipeline(df, "Direct + qwen30", "avg_total_consumption_wh")

	bert_filter_co2 = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_filter_emissions_g"
	)
	bert_rerank_co2 = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_ranking_emissions_g"
	)
	direct_co2 = _series_for_pipeline(df, "Direct + qwen30", "avg_total_emissions_g")

	fig, ax_left = plt.subplots(figsize=(10, 8))
	ax_right = ax_left.twinx()

	# Left axis: energy bars (with stacked BERT decomposition).
	ax_left.bar(
		x - width / 2,
		bert_filter_cons,
		width,
		label="BERT filter consumption",
		color=COLOR_FILTER,
	)
	ax_left.bar(
		x - width / 2,
		bert_rerank_cons,
		width,
		bottom=bert_filter_cons,
		label="Qwen3-30B rerank consumption",
		color=COLOR_RERANK,
	)
	ax_left.bar(
		x + width / 2,
		direct_cons,
		width,
		label="Sliding Window consumption",
		color=COLOR_DIRECT,
	)

	# Right axis: CO2 lines for readability with dual-axis scale.
	ax_right.plot(
		x - width / 2,
		bert_filter_co2 + bert_rerank_co2,
		marker="o",
		linewidth=2,
		color=COLOR_CO2_BERT,
		label="BERT filter + Qwen3-30B rerank CO2",
	)
	ax_right.plot(
		x + width / 2,
		direct_co2,
		marker="s",
		linewidth=2,
		color=COLOR_CO2_DIRECT,
		label="Sliding Window CO2",
	)

	ax_left.set_title(
		"Energy Consumption and\nCO2 Emissions Comparison",
		fontsize=FONT_TITLE,
	)
	ax_left.set_xlabel("bm25_k", fontsize=FONT_LABEL)
	ax_left.set_ylabel("Consumption (Wh)", fontsize=FONT_LABEL)
	ax_right.set_ylabel("CO2 Emissions (g)", fontsize=FONT_LABEL)

	ax_left.set_xticks(x)
	ax_left.set_xticklabels([str(v) for v in BM25_VALUES])
	ax_left.tick_params(axis="both", labelsize=FONT_TICK)
	ax_right.tick_params(axis="y", labelsize=FONT_TICK)
	ax_left.grid(axis="y", alpha=0.3)

	# Merge legends from both axes.
	left_handles, left_labels = ax_left.get_legend_handles_labels()
	right_handles, right_labels = ax_right.get_legend_handles_labels()
	ax_left.legend(
		left_handles + right_handles,
		left_labels + right_labels,
		loc="upper left",
		fontsize=FONT_LEGEND,
	)

	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def plot_side_by_side(df: pd.DataFrame, output_path: Path) -> None:
	x = np.arange(len(BM25_VALUES))
	width = 0.4

	bert_filter_time = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_filter_time"
	)
	bert_rerank_time = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_ranking_time"
	)
	direct_time = _series_for_pipeline(df, "Direct + qwen30", "avg_total_time")

	bert_filter_cons = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_filter_consumption_wh"
	)
	bert_rerank_cons = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_ranking_consumption_wh"
	)
	direct_cons = _series_for_pipeline(df, "Direct + qwen30", "avg_total_consumption_wh")

	bert_filter_co2 = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_filter_emissions_g"
	)
	bert_rerank_co2 = _series_for_pipeline(
		df, "BERT-bge-m3 + qwen30 + fs=30", "avg_ranking_emissions_g"
	)
	direct_co2 = _series_for_pipeline(df, "Direct + qwen30", "avg_total_emissions_g")

	fig, (ax_time, ax_cons) = plt.subplots(1, 2, figsize=(24, 11), constrained_layout=True)
	ax_co2 = ax_cons.twinx()

	# Enforce same plotting-box geometry on both panels.
	ax_time.set_box_aspect(0.8)
	ax_cons.set_box_aspect(0.8)

	# Left panel: time
	ax_time.bar(
		x - width / 2,
		bert_filter_time,
		width,
		label="BERT filter time",
		color=COLOR_FILTER,
	)
	ax_time.bar(
		x - width / 2,
		bert_rerank_time,
		width,
		bottom=bert_filter_time,
		label="Qwen3-30B rerank time",
		color=COLOR_RERANK,
	)
	ax_time.bar(
		x + width / 2,
		direct_time,
		width,
		label="Sliding Window time",
		color=COLOR_DIRECT,
	)

	ax_time.set_title("Time Comparison", fontsize=FONT_TITLE_SIDE)
	ax_time.set_xlabel("bm25_k", fontsize=FONT_LABEL_SIDE)
	ax_time.set_ylabel("Time (s)", fontsize=FONT_LABEL_SIDE)
	max_time = max(np.nanmax(bert_filter_time + bert_rerank_time), np.nanmax(direct_time))
	upper_time = max(5, int(np.ceil(max_time / 5.0) * 5))
	ax_time.set_ylim(0, upper_time)
	ax_time.set_yticks(np.arange(0, upper_time + 1, 5))
	ax_time.set_xticks(x)
	ax_time.set_xticklabels([str(v) for v in BM25_VALUES])
	ax_time.tick_params(axis="both", labelsize=FONT_TICK_SIDE)
	ax_time.grid(axis="y", alpha=0.3)
	ax_time.legend(loc="upper left", fontsize=FONT_LEGEND_SIDE)

	# Right panel: consumption + CO2
	ax_cons.bar(
		x - width / 2,
		bert_filter_cons,
		width,
		label="BERT filter consumption",
		color=COLOR_FILTER,
	)
	ax_cons.bar(
		x - width / 2,
		bert_rerank_cons,
		width,
		bottom=bert_filter_cons,
		label="Qwen3-30B rerank consumption",
		color=COLOR_RERANK,
	)
	ax_cons.bar(
		x + width / 2,
		direct_cons,
		width,
		label="Sliding Window consumption",
		color=COLOR_DIRECT,
	)

	ax_co2.plot(
		x - width / 2,
		bert_filter_co2 + bert_rerank_co2,
		marker="o",
		linewidth=4,
        markersize=12,
		color=COLOR_CO2_BERT,
		label="BERT filter + Qwen3-30B rerank CO2",
	)
	ax_co2.plot(
		x + width / 2,
		direct_co2,
		marker="s",
		linewidth=4,
		markersize=12,
		color=COLOR_CO2_DIRECT,
		label="Sliding Window CO2",
	)

	ax_cons.set_title(
		"Energy Consumption and\nCO2 Emissions Comparison",
		fontsize=FONT_TITLE_SIDE,
	)
	ax_cons.set_xlabel("bm25_k", fontsize=FONT_LABEL_SIDE)
	ax_cons.set_ylabel("Consumption (Wh)", fontsize=FONT_LABEL_SIDE)
	ax_co2.set_ylabel("CO2 Emissions (g)", fontsize=FONT_LABEL_SIDE)
	ax_cons.set_xticks(x)
	ax_cons.set_xticklabels([str(v) for v in BM25_VALUES])
	ax_cons.tick_params(axis="both", labelsize=FONT_TICK_SIDE)
	ax_co2.tick_params(axis="y", labelsize=FONT_TICK_SIDE)
	ax_cons.grid(axis="y", alpha=0.3)

	left_handles, left_labels = ax_cons.get_legend_handles_labels()
	right_handles, right_labels = ax_co2.get_legend_handles_labels()
	ax_cons.legend(
		left_handles + right_handles,
		left_labels + right_labels,
		loc="upper left",
		fontsize=FONT_LEGEND_SIDE,
	)

	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def main() -> None:
	df = _prepare_data(CSV_FILE)
	plot_time(df, OUTPUT_TIME)
	plot_energy_and_co2(df, OUTPUT_ENERGY_CO2)
	plot_side_by_side(df, OUTPUT_SIDE_BY_SIDE)
	print(f"Saved: {OUTPUT_TIME}")
	print(f"Saved: {OUTPUT_ENERGY_CO2}")
	print(f"Saved: {OUTPUT_SIDE_BY_SIDE}")


if __name__ == "__main__":
	main()
