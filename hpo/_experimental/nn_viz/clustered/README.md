# Elise Clustered Network Visualizations

Checkpoint: `G:\Meine Ablage\rl_lab\hpo\best_checkpoints\solar_system_lander_10d_elise_stp\best_eval_checkpoint.pt`
Score in metadata: `267.4799686753183`
Hidden size: `128`
Total weights: `18176`

Generated files:

- `elise_clustered_layer2_heatmap.png`: complete `H2 x H1` layer2 matrix, rows and columns clustered by absolute connection profile.
- `elise_clustered_bipartite_top_edges.png` / `.svg`: H1 and H2 reordered by cluster, with strongest incoming H1 edges per H2.
- `elise_clustered_top_neurons.png` / `.html`: top 30 H1/H2 neurons by connection strength, clustered and labeled.

`H1-80` is highlighted because it is the dominant dv/popometer-coupled hidden-1 neuron.
