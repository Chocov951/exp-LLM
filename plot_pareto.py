from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Paths
# ============================================================
path_trec19 = Path('results_trec19.csv')
path_trec20 = Path('results_trec20.csv')
OUTPUT_TREC19 = Path('pareto_trec19.png')
OUTPUT_TREC20 = Path('pareto_trec20.png')
OUTPUT_SIDE_BY_SIDE = Path('pareto_side_by_side.png')

# ============================================================
# Visual style (mirrors bar_chart.py)
# ============================================================
FONT_TITLE  = 28
FONT_LABEL  = 26
FONT_TICK   = 22
FONT_LEGEND = 22
FONT_ANNOT  = 20

FONT_TITLE_SIDE  = 34
FONT_LABEL_SIDE  = 28
FONT_TICK_SIDE   = 24
FONT_ANNOT_SIDE  = 22

COLOR_DIRECT  = "#D79B00"   # SW / Sliding Window
COLOR_DIRECT_FONT  = "#A17400"   # SW / Sliding Window
COLOR_BGE     = "#82B366"   # FtR-BGE
COLOR_BGE_FONT     = "#608D46"   # FtR-BGE
COLOR_T5      = "#6C8EBF"   # FtR-T5 / Pointwise
COLOR_MINILM  = "#B85450"   # FtR-MiniLM
COLOR_PARETO  = "#444444"

# ============================================================
# 3. Fonctions utilitaires
# ============================================================

def load_results(path):
    return pd.read_csv(path)


def to_numeric_if_exists(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def filter_config(df, filter_method=None, rerank_method=None, filter_size=None):
    out = df.copy()

    if filter_method is not None:
        out = out[out['filter_method'] == filter_method]

    if rerank_method is not None:
        out = out[out['rerank_method'] == rerank_method]

    if filter_size is not None:
        out = out[out['filter_size'] == filter_size]

    return out


def sort_for_curve(df):
    if 'bm25_k' in df.columns and df['bm25_k'].notna().any():
        return df.sort_values('bm25_k')
    return df.sort_values('avg_total_time')


def pareto_front(df, x_col='avg_total_time', y_col='ndcg@10'):
    valid = df.dropna(subset=[x_col, y_col]).copy()
    if valid.empty:
        return valid

    non_dominated = []
    rows = valid.to_dict('records')

    for i, p in enumerate(rows):
        dominated = False
        for j, q in enumerate(rows):
            if i == j:
                continue
            if (
                q[x_col] <= p[x_col] and
                q[y_col] >= p[y_col] and
                (q[x_col] < p[x_col] or q[y_col] > p[y_col])
            ):
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)

    return pd.DataFrame(non_dominated).sort_values(x_col)


def annotate_selected_k(ax, df, x_col='avg_total_time', y_col='ndcg@10',
                        k_col='bm25_k', selected_k=(100, 500),
                        fontsize=8, color='black', y_offset=6):
    """
    N'annote que certains k (par défaut 100 et 500) pour éviter la surcharge.
    """
    if k_col not in df.columns:
        return

    for _, row in df.iterrows():
        if pd.notna(row.get(k_col)) and int(row[k_col]) in selected_k:
            ax.annotate(
                f'k={int(row[k_col])}',
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha='center',
                fontsize=fontsize,
                color=color
            )


def label_last_point(ax, df, text, x_col='avg_total_time', y_col='ndcg@10',
                     color='black', fontsize=12, dx=5, dy=0):
    if df.empty:
        return

    last = df.iloc[-1]
    ax.annotate(
        text,
        (last[x_col], last[y_col]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha='left',
        va='center',
        fontsize=fontsize,
        color=color
    )


def label_single_point(ax, df, text, x_col='avg_total_time', y_col='ndcg@10',
                       color='black', fontsize=12, dx=5, dy=0):
    if df.empty:
        return

    row = df.iloc[0]
    ax.annotate(
        text,
        (row[x_col], row[y_col]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha='left',
        va='center',
        fontsize=fontsize,
        color=color
    )


# ============================================================
# 4. Figure LNCS-friendly : un dataset par figure
# ============================================================

def _fill_ax(ax, df, title, show_pareto=True, font_title=FONT_TITLE,
             font_label=FONT_LABEL, font_tick=FONT_TICK, font_annot=FONT_ANNOT):
    """Draw the pareto trade-off chart onto an existing Axes object."""
    styles = {
        'SW':       {'color': COLOR_DIRECT, 'font_color': COLOR_DIRECT_FONT,  'marker': 's', 'ls': '--', 'lw': 5, 'ms': 15},
        'FtR-BGE':  {'color': COLOR_BGE, 'font_color': COLOR_BGE_FONT,     'marker': 'o', 'ls': '-',  'lw': 5, 'ms': 15},
        'FtR-T5':   {'color': COLOR_T5,      'marker': 'X', 's': 300},
        'FtR-MiniLM': {'color': COLOR_MINILM, 'marker': '^', 's': 300},
        'Pareto':   {'color': COLOR_PARETO,  'ls': ':',     'lw': 5}
    }

    # Sliding Window
    sw = filter_config(
        df,
        filter_method='Direct',
        rerank_method='Listwise-qwen30-W20-S10'
    )
    sw = sort_for_curve(sw)

    ax.plot(
        sw['avg_total_time'], sw['ndcg@10'],
        color=styles['SW']['color'],
        marker=styles['SW']['marker'],
        linestyle=styles['SW']['ls'],
        linewidth=styles['SW']['lw'],
        markersize=styles['SW']['ms'],
        label='Sliding Window',
        zorder=3
    )
    if not sw.empty and 'bm25_k' in sw.columns:
        for row in (sw.iloc[0], sw.iloc[-1]):
            if pd.notna(row.get('bm25_k')):
                k_val = int(row['bm25_k'])
                if k_val in (100, 500):
                    y_offset = -30 if k_val == 100 else 14
                    ax.annotate(
                        f'k={k_val}',
                        (row['avg_total_time'], row['ndcg@10']),
                        textcoords='offset points',
                        xytext=(0, y_offset),
                        ha='center',
                        fontsize=font_annot + 4,
                        color=styles['SW']['font_color'],
                    )

    # FtR-BGE
    ftr_bge = filter_config(
        df,
        filter_method='BERT-bge-m3',
        rerank_method='Listwise-qwen30-Wfull-S10',
        filter_size=30
    )
    ftr_bge = sort_for_curve(ftr_bge)

    ax.plot(
        ftr_bge['avg_total_time'], ftr_bge['ndcg@10'],
        color=styles['FtR-BGE']['color'],
        marker=styles['FtR-BGE']['marker'],
        linestyle=styles['FtR-BGE']['ls'],
        linewidth=styles['FtR-BGE']['lw'],
        markersize=styles['FtR-BGE']['ms'],
        label='FtR-BGE',
        zorder=4
    )
    if not ftr_bge.empty and 'bm25_k' in ftr_bge.columns:
        for row in (ftr_bge.iloc[0], ftr_bge.iloc[-1]):
            if pd.notna(row.get('bm25_k')):
                k_val = int(row['bm25_k'])
                if k_val in (100, 500):
                    y_offset = -30 if k_val == 100 else 14
                    ax.annotate(
                        f'k={k_val}',
                        (row['avg_total_time'], row['ndcg@10']),
                        textcoords='offset points',
                        xytext=(0, y_offset),
                        ha='center',
                        fontsize=font_annot + 4,
                        color=styles['FtR-BGE']['font_color'],
                    )

    # FtR-T5
    ftr_t5 = filter_config(
        df,
        filter_method='Pointwise',
        rerank_method='Listwise-qwen30-Wfull-S10',
        filter_size=20
    )
    ax.scatter(
        ftr_t5['avg_total_time'], ftr_t5['ndcg@10'],
        color=styles['FtR-T5']['color'],
        marker=styles['FtR-T5']['marker'],
        s=styles['FtR-T5']['s'],
        label='FtR-T5',
        zorder=5
    )

    # FtR-MiniLM
    ftr_fast = filter_config(
        df,
        filter_method='BERT-all-MiniLM-L12-v2',
        rerank_method='Listwise-qwen4-Wfull-S10',
        filter_size=20
    )
    ax.scatter(
        ftr_fast['avg_total_time'], ftr_fast['ndcg@10'],
        color=styles['FtR-MiniLM']['color'],
        marker=styles['FtR-MiniLM']['marker'],
        s=styles['FtR-MiniLM']['s'],
        label='FtR-MiniLM',
        zorder=5
    )

    # Pareto calculé sur les points affichés
    if show_pareto:
        all_points = pd.concat([
            sw[['avg_total_time', 'ndcg@10']].copy(),
            ftr_bge[['avg_total_time', 'ndcg@10']].copy(),
            ftr_t5[['avg_total_time', 'ndcg@10']].copy(),
            ftr_fast[['avg_total_time', 'ndcg@10']].copy()
        ], ignore_index=True)

        pf = pareto_front(all_points)

        if not pf.empty and len(pf) >= 2:
            ax.plot(
                pf['avg_total_time'], pf['ndcg@10'],
                color=styles['Pareto']['color'],
                linestyle=styles['Pareto']['ls'],
                linewidth=styles['Pareto']['lw'],
                label='Pareto Frontier',
                zorder=2
            )

    # Expand visible value ranges a bit beyond data extents.
    all_x = pd.concat([
        sw['avg_total_time'],
        ftr_bge['avg_total_time'],
        ftr_t5['avg_total_time'],
        ftr_fast['avg_total_time'],
    ], ignore_index=True).dropna()
    all_y = pd.concat([
        sw['ndcg@10'],
        ftr_bge['ndcg@10'],
        ftr_t5['ndcg@10'],
        ftr_fast['ndcg@10'],
    ], ignore_index=True).dropna()

    if not all_x.empty:
        x_min = float(all_x.min())
        x_max = float(all_x.max())
        ax.set_xlim(x_min * 0.9, x_max * 1.12)

    if not all_y.empty:
        y_min = float(all_y.min())
        y_max = float(all_y.max())
        y_pad = 0.004
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Cosmétique
    ax.set_xscale('log')
    ax.set_title(title, fontsize=font_title)
    ax.set_xlabel('Mean latency / query (s, log scale)', fontsize=font_label)
    ax.set_ylabel('NDCG@10', fontsize=font_label)
    ax.tick_params(axis='both', labelsize=font_tick)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=font_annot, loc='lower right')
    ax.set_box_aspect(0.8)


def generate_lncs_tradeoff_plot(csv_path, title, save_path, show_pareto=True):
    df = load_results(csv_path)
    df = to_numeric_if_exists(
        df,
        ['avg_total_time', 'ndcg@10', 'bm25_k', 'filter_size', 'avg_total_emissions_g']
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    _fill_ax(ax, df, title, show_pareto=show_pareto)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_side_by_side(csv_path_left, csv_path_right, title_left, title_right,
                          save_path, show_pareto=True):
    df19 = load_results(csv_path_left)
    df19 = to_numeric_if_exists(
        df19, ['avg_total_time', 'ndcg@10', 'bm25_k', 'filter_size', 'avg_total_emissions_g']
    )
    df20 = load_results(csv_path_right)
    df20 = to_numeric_if_exists(
        df20, ['avg_total_time', 'ndcg@10', 'bm25_k', 'filter_size', 'avg_total_emissions_g']
    )

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(23, 9), constrained_layout=True
    )

    _fill_ax(ax_left,  df19, title_left,  show_pareto=show_pareto,
             font_title=FONT_TITLE_SIDE, font_label=FONT_LABEL_SIDE,
             font_tick=FONT_TICK_SIDE,   font_annot=FONT_ANNOT_SIDE)
    _fill_ax(ax_right, df20, title_right, show_pareto=show_pareto,
             font_title=FONT_TITLE_SIDE, font_label=FONT_LABEL_SIDE,
             font_tick=FONT_TICK_SIDE,   font_annot=FONT_ANNOT_SIDE)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================================
# 5. Génération des deux figures séparées
# ============================================================

generate_lncs_tradeoff_plot(
    csv_path=path_trec19,
    title='TREC-DL 2019',
    save_path=OUTPUT_TREC19,
    show_pareto=True
)

generate_lncs_tradeoff_plot(
    csv_path=path_trec20,
    title='TREC-DL 2020',
    save_path=OUTPUT_TREC20,
    show_pareto=True
)

generate_side_by_side(
    csv_path_left=path_trec19,
    csv_path_right=path_trec20,
    title_left='TREC-DL 2019',
    title_right='TREC-DL 2020',
    save_path=OUTPUT_SIDE_BY_SIDE,
    show_pareto=True
)
