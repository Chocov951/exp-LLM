import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Parameters to vary
filter_numbers = [10, 20, 30]
bm25_topks = [100, 200, 300]

# Data structures to store results
data = {
    'filter': [],
    'topk': [],
    'duration': [],
    'emissions': [],
    'energy_consumed': []
}

# Loop through all parameter combinations
for filter_number in filter_numbers:
    for bm25_topk in bm25_topks:
        # Construct file path
        filepath = f'paper_res/results_trec20_qwen72_reject{filter_number}_topk{bm25_topk}_carbon_final.json'
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist. Skipping...")
            continue
        
        # Load the data
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Sum data from tracker_rerank and tracker_filter
        duration = results['tracker_rerank']['duration'] + results['tracker_reject']['duration']
        # Convert emissions from kg to grams
        emissions = (results['tracker_rerank']['emissions'] + results['tracker_reject']['emissions']) * 1000
        energy = results['tracker_rerank']['energy_consumed'] + results['tracker_reject']['energy_consumed']
        
        # Store the data
        data['filter'].append(filter_number)
        data['topk'].append(bm25_topk)
        data['duration'].append(duration)
        data['emissions'].append(emissions)
        data['energy_consumed'].append(energy)

# Additional data structures for window-based approach
window_data = {
    'topk': [],
    'duration': [],
    'emissions': [],
    'energy_consumed': []
}

# Load window-based data
for bm25_topk in bm25_topks:
    filepath = f'paper_res/results_trec20_qwen72_rerank_window_topk{bm25_topk}_carbon.json'
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist. Skipping...")
        continue
    
    # Load the data
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    duration = results['tracker_rerank']['duration'] 
    # Convert emissions from kg to grams
    emissions = results['tracker_rerank']['emissions'] * 1000  # Convert kg to g
    energy = results['tracker_rerank']['energy_consumed'] 
    
    # Store the data
    window_data['topk'].append(bm25_topk)
    window_data['duration'].append(duration)
    window_data['emissions'].append(emissions)
    window_data['energy_consumed'].append(energy)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Plot 1: Duration
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))

for filter in filter_numbers:
    filter_data = [(t, d) for r, t, d in zip(data['filter'], data['topk'], data['duration']) if r == filter]
    if filter_data:
        topk_vals, duration_vals = zip(*filter_data)
        ax1.plot(topk_vals, duration_vals, marker='o', label=f'Filter={filter}')

if window_data['topk']:
    sorted_indices = sorted(range(len(window_data['topk'])), key=lambda i: window_data['topk'][i])
    sorted_topk = [window_data['topk'][i] for i in sorted_indices]
    sorted_duration = [window_data['duration'][i] for i in sorted_indices]
    ax1.plot(sorted_topk, sorted_duration, marker='s', linestyle='--', 
             color='red', linewidth=2, label='RankGPT')

ax1.set_title('Mean Duration per query (seconds)', fontsize=16)
ax1.set_xlabel('BM25 K', fontsize=14)
ax1.set_ylabel('Duration (s)', fontsize=14)
ax1.legend(fontsize=12)
ax1.set_xticks(bm25_topks)
ax1.set_xticklabels(bm25_topks)

plt.tight_layout()
plt.savefig('plots/duration_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Emissions
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

for filter in filter_numbers:
    filter_data = [(t, e) for r, t, e in zip(data['filter'], data['topk'], data['emissions']) if r == filter]
    if filter_data:
        topk_vals, emission_vals = zip(*filter_data)
        ax2.plot(topk_vals, emission_vals, marker='o', label=f'Filter={filter}')

if window_data['topk']:
    sorted_emissions = [window_data['emissions'][i] for i in sorted_indices]
    ax2.plot(sorted_topk, sorted_emissions, marker='s', linestyle='--', 
             color='red', linewidth=2, label='RankGPT')

ax2.set_title('Mean CO$_2$ Emissions per query (g)', fontsize=16)
ax2.set_xlabel('BM25 K', fontsize=14)
ax2.set_ylabel('CO$_2$ Emissions (g)', fontsize=14)
ax2.legend(fontsize=12)
ax2.set_xticks(bm25_topks)
ax2.set_xticklabels(bm25_topks)

plt.tight_layout()
plt.savefig('plots/emissions_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Energy
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))

for filter in filter_numbers:
    filter_data = [(t, e) for r, t, e in zip(data['filter'], data['topk'], data['energy_consumed']) if r == filter]
    if filter_data:
        topk_vals, energy_vals = zip(*filter_data)
        ax3.plot(topk_vals, energy_vals, marker='o', label=f'Filter={filter}')

if window_data['topk']:
    sorted_energy = [window_data['energy_consumed'][i] for i in sorted_indices]
    ax3.plot(sorted_topk, sorted_energy, marker='s', linestyle='--', 
             color='red', linewidth=2, label='RankGPT')

ax3.set_title('Mean Energy Consumed per query (kWh)', fontsize=16)
ax3.set_xlabel('BM25 K', fontsize=14)
ax3.set_ylabel('Energy Consumed (kWh)', fontsize=14)
ax3.legend(fontsize=12)
ax3.set_xticks(bm25_topks)
ax3.set_xticklabels(bm25_topks)

plt.tight_layout()
plt.savefig('plots/energy_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots saved successfully!")