"""Main-paper figures, one script each, written to runs/figures/ ($PGNM_RUNS).

    python -m figures.teaser --photo PHOTO      # Fig. 1 (bring your own photo)
    python -m figures.residual_density          # Fig. 3, SIDD-Medium fitting crops
    python -m figures.quantile_mapping          # Fig. 4, schematic
    python -m figures.visual_comparison sidd    # Fig. 5, generated SIDD outputs
    python -m figures.visual_comparison film    # Fig. 6, generated film-grain outputs
    python -m figures.joint_pca                 # Fig. 7, the 24 fitted presets

Fig. 2 (inference pipeline) is a hand-drawn diagram and has no script.
The scripts use Matplotlib, SciPy and Pillow besides NumPy.
"""
