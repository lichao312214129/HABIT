# Cluster Validity — Reading the Curves

`habit get-habitat` writes a validation curve when `plot_curves: true`. This
guide explains how to interpret each metric and pick the optimal number of
habitats.

## Metric cheat sheet

| Metric | Range | Better when | Use case |
|---|---|---|---|
| `silhouette` | -1 to +1 | **higher** | Default; balances cohesion and separation |
| `davies_bouldin` | 0 to ∞ | **lower** | Penalizes overlapping clusters |
| `calinski_harabasz` | 0 to ∞ | **higher** | Sensitive to large k; can over-cluster |
| `inertia` (kmeans only) | 0 to ∞ | **lower** | Used for elbow method (look at the bend, not the minimum) |

## How to choose the metric

```yaml
HabitatsSegmention:
  habitat:
    habitat_cluster_selection_method: silhouette   # recommended default
```

- **`silhouette`** for most studies. Picks the k with the highest silhouette
  score in `[min_clusters, max_clusters]`.
- **`davies_bouldin`** when you suspect clusters overlap (e.g. DCE habitats
  with similar wash-out behavior).
- **`calinski_harabasz`** when you want to maximize between-cluster variance;
  tends to choose larger k.
- **`inertia`** (one_step or kmeans) → use the **elbow method**: look at the
  curve, find the "knee" where the rate of decrease slows abruptly. The
  selection picks the minimum-inertia point, but you should override
  `fixed_n_clusters` based on the visual bend.

## Reading the validation plot

Generated to `<out_dir>/visualizations/.../cluster_validation_scores.png`:

```
       silhouette
   ^   .---.
   |  /     \
   | /       \___
   |/             \____
   +---+---+---+---+---+--> k (number of habitats)
       2   3   4   5   6
                    ^
                    pick this k (peak silhouette)
```

For inertia (elbow):

```
       inertia
   ^   .
   |    \
   |     `.
   |       *<-- elbow at k=4
   |          ----.____
   +---+---+---+---+---+--> k
       2   3   4   5   6
```

## Per-subject one_step curves

In `one_step` mode, each subject gets its OWN validation curve under:
```
out_dir/visualizations/supervoxel_clustering/<subject>_*_validation.png
```

The agent should not try to set a single optimal k for all subjects. The
algorithm picks per-subject optimal automatically when `fixed_n_clusters: null`.

## Common gotchas

1. **Silhouette gives k=2 for everyone**: ROI is too homogeneous; intensity
   distribution has only one mode. Try `concat()` with more modalities or
   switch to `voxel_radiomics()`.
2. **Validation curve is monotonically improving**: typical when `max_clusters`
   is too small. Increase `max_clusters` to 12 or 15.
3. **Plot says k=8 but tumors look over-segmented**: a metric thinks more
   clusters are better but biology disagrees. Manually override with
   `fixed_n_clusters: 4` for biological plausibility.

## How to fix `fixed_n_clusters` post-hoc

You looked at the curve and decided k=4 looks right? Edit the YAML:

```yaml
HabitatsSegmention:
  habitat:
    fixed_n_clusters: 4   # was: null
```

Re-run only the population-level step is not currently supported via CLI; you
must rerun `habit get-habitat` with the updated config. To save time, set
`HabitatsSegmention.habitat.mode: testing` if you already have a saved
`supervoxel2habitat_clustering_model.pkl`.
