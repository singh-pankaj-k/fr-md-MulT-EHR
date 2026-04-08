import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def aggregate_results(checkpoints_dir='checkpoints'):
    results = []
    checkpoints_path = Path(checkpoints_dir)
    
    # Traverse checkpoints directory
    for stats_file in checkpoints_path.rglob('training_stats.json'):
        folder = stats_file.parent
        # Try to infer model, dataset, task/ablation from path
        # Expected paths: 
        # checkpoints/{Model_Type}/training_stats.json (where Model_Type contains Dataset)
        # checkpoints/{Ablation_Type}/{Dataset}/{Detail}/training_stats.json
        
        parts = folder.relative_to(checkpoints_path).parts
        model_type = parts[0]
        
        # Determine dataset and detail from path parts
        if model_type in ['GNN_ablation', 'Hidden_Dim_ablation', 'Dropout_ablation']:
            dataset = parts[1] if len(parts) > 1 else 'unknown'
            detail = parts[2] if len(parts) > 2 else 'default'
        else:
            # For HGT_Causal_MIMIC4_RMDL etc.
            dataset = 'mimiciv' if 'mimic4' in model_type.lower() else 'mimiciii' if 'mimic3' in model_type.lower() else 'unknown'
            detail = 'default'
            
        # Load stats
        try:
            with open(stats_file, 'r') as f:
                lines = f.readlines()
                epochs_stats = [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            print(f"Error reading {stats_file}: {e}")
            continue
            
        if not epochs_stats:
            continue
            
        summary = {
            'path': str(folder),
            'model_type': model_type,
            'dataset': dataset,
            'detail': detail
        }
        
        tasks = ['mort_pred', 'los', 'drug_rec', 'readm']
        for task in tasks:
            best_auc = 0
            best_acc = 0
            auc_key = f"{task}_roc_auc"
            if task == 'los': auc_key = "los_roc_auc_weighted_ovo"
            if task == 'drug_rec': auc_key = "drug_rec_roc_auc_samples"
            
            acc_key = f"{task}_accuracy"
            
            for epoch in epochs_stats:
                auc = epoch.get(auc_key, 0)
                if isinstance(auc, str): auc = 0 # Handle potential NaN or errors
                if auc > best_auc:
                    best_auc = auc
                    best_acc = epoch.get(acc_key, 0)
            
            summary[f"{task}_auc"] = best_auc
            summary[f"{task}_acc"] = best_acc
            
        results.append(summary)
        
    return pd.DataFrame(results)

def plot_ablation(df, ablation_type, x_col, benchmark_dir):
    ablation_df = df[df['model_type'] == ablation_type].copy()
    if ablation_df.empty:
        return
        
    # Convert x_col to numeric if possible
    ablation_df[x_col] = pd.to_numeric(ablation_df['detail'], errors='coerce')
    ablation_df = ablation_df.sort_values(x_col)
    
    tasks = ['mort_pred', 'los', 'drug_rec', 'readm']
    plt.figure(figsize=(10, 6))
    for task in tasks:
        plt.plot(ablation_df[x_col], ablation_df[f"{task}_auc"], marker='o', label=f"{task} AUC")
        
    plt.title(f"Ablation Study: {ablation_type}")
    plt.xlabel(x_col)
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(benchmark_dir, f"plots/{ablation_type.lower()}_auc.png"))
    plt.close()

def plot_model_comparison(df, benchmark_dir):
    # Filter for main models or GNN ablation
    # We now have more models trained sequentially
    main_model_types = ['HGT_Causal_MIMIC4_RMDL', 'HGT_MTCausal_MIMIC4_RMDL', 'HGT_Causal_MIMIC3_RMDL', 'HGT_MTCausal_MIMIC3_RMDL',
                        'GNN_ablation', 'HGT_MIMIC4_Readm', 'HGT_MIMIC3_Readm', 
                        'HGT_ST_MIMIC4_RMDL_L3', 'HGT_ST_MIMIC3_RMLD', 'AdaCare']
    
    comparison_df = df[df['model_type'].isin(main_model_types) | df['model_type'].str.contains('HGT_')].copy()
    if comparison_df.empty:
        comparison_df = df.copy()

    # Map detail to model name for GNN_ablation, and include dataset if needed
    comparison_df['model_name'] = comparison_df.apply(
        lambda x: f"{x['dataset']}_{x['detail']}" if x['model_type'] == 'GNN_ablation' else f"{x['dataset']}_{x['model_type']}", axis=1
    )
    
    tasks = ['mort_pred', 'los', 'drug_rec', 'readm']
    
    # Melt for easier plotting
    melted = []
    for _, row in comparison_df.iterrows():
        for task in tasks:
            melted.append({
                'model_name': row['model_name'],
                'Task': task,
                'AUC': row[f"{task}_auc"]
            })
    melted_df = pd.DataFrame(melted)
    
    if melted_df.empty:
        return

    # Deduplicate in case of multiple runs
    melted_df = melted_df.groupby(['model_name', 'Task'])['AUC'].max().reset_index()

    plt.figure(figsize=(12, 7))
    models = sorted(melted_df['model_name'].unique())
    tasks_unique = melted_df['Task'].unique()
    
    x = range(len(tasks_unique))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = melted_df[melted_df['model_name'] == model]
        # Ensure tasks are in order
        model_data = model_data.set_index('Task').reindex(tasks_unique).reset_index()
        plt.bar([pos + i * width for pos in x], model_data['AUC'], width, label=model)
        
    plt.xticks([pos + (len(models)-1) * width / 2 for pos in x], tasks_unique)
    plt.title("Model Performance Comparison (ROC AUC)")
    plt.ylabel("ROC AUC")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_dir, "plots/model_comparison_auc.png"))
    plt.close()

def main():
    benchmark_dir = 'benchmark'
    os.makedirs(os.path.join(benchmark_dir, 'plots'), exist_ok=True)
    
    print("Aggregating results from checkpoints...")
    df = aggregate_results()
    
    if df.empty:
        print("No results found in checkpoints directory.")
        return
        
    # Save raw aggregation
    df.to_csv(os.path.join(benchmark_dir, 'aggregated_results.csv'), index=False)
    
    # Generate Markdown table
    with open(os.path.join(benchmark_dir, 'summary_report.md'), 'w') as f:
        f.write("# MulT-EHR Benchmark Summary Report\n\n")
        f.write("## Performance Metrics (ROC AUC)\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            # Fallback if tabulate is not installed
            f.write(df.to_csv(sep="|", index=False))
        f.write("\n\n## Visualizations\n\n")
        f.write("### Model Comparison\n")
        f.write("![Model Comparison](plots/model_comparison_auc.png)\n\n")
        
        if any(df['model_type'] == 'Hidden_Dim_ablation'):
            plot_ablation(df, 'Hidden_Dim_ablation', 'Hidden Dimension', benchmark_dir)
            f.write("### Hidden Dimension Ablation\n")
            f.write("![Hidden Dim Ablation](plots/hidden_dim_ablation_auc.png)\n\n")
            
        if any(df['model_type'] == 'Dropout_ablation'):
            plot_ablation(df, 'Dropout_ablation', 'Dropout Rate', benchmark_dir)
            f.write("### Dropout Ablation\n")
            f.write("![Dropout Ablation](plots/dropout_ablation_auc.png)\n\n")

    # Generate plots
    print("Generating plots...")
    plot_model_comparison(df, benchmark_dir)
    
    print(f"Benchmark report generated in {benchmark_dir}/")

if __name__ == "__main__":
    main()
