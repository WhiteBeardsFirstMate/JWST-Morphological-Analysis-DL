Galaxy Morphology ML Notebook
Overview
This notebook implements an end‑to‑end machine learning workflow to study galaxy morphology using Galaxy Zoo data. It loads and preprocesses image and label data, applies t‑SNE and UMAP for structure discovery, and trains classifiers to predict Galaxy Zoo attributes.

Runtime and Dependencies
Designed for a Colab‑style environment with optional GPU acceleration (Tesla T4, CUDA 12.2).

Main Python libraries:

NumPy, SciPy, pandas

scikit‑learn (datasets, model_selection, manifold, metrics, SVM)

umap‑learn (UMAP dimensionality reduction)

Matplotlib and Seaborn for plots.

Utility Functions
The notebook defines helper functions for:

Saving text summaries to disk (save_string_to_file).

Computing average accuracy and loss over training runs (create_performance_summary).

Summarizing model histories with confidence intervals using Student’s t‑distribution (review_model_history, confidence_interval).

Visualizing training vs. validation accuracy and loss, and saving performance plots as PNG files (visualize_model_performance).

Data and Feature Engineering
Creates DataFrames from Galaxy Zoo label arrays with named columns (e.g., smooth-or-featured-gz2_*, disk-edge-on-gz2_*, spiral-arm-count-gz2_*).

Works with subsets of 1000 samples and 30 target columns for tractable experimentation.

Constructs X_subset_array from image channels (primarily the red channel) to exploit astrophysical redshift properties while acknowledging loss of fine‑grained information.

Dimensionality Reduction and Visualization
t‑SNE:

2D embeddings with perplexity 30 and random_state 42 on the red channel.

Generates a 6×5 grid of scatter plots (30 subplots), one per target column, colored by label values, saved as r_image_tsne_plot.png.

Notes which attributes show visible clusters or gradients (e.g., smooth-or-featured-gz2_featured-or-disk, disk-edge-on-gz2_no, how-rounded-gz2_*).

UMAP:

Additional structure discovery using UMAP to complement t‑SNE analysis.

Model Training and Evaluation
Uses scikit‑learn classifiers (including SVM) with standard scaling for predictive modeling on galaxy labels.

Reports accuracy, precision, recall, F1, classification reports, confusion matrices, and confidence intervals to quantify performance.

Stores visual and text summaries to support comparison across models and hyperparameters.

Outputs
Performance plots (<model_name>_performance_plot.png).

t‑SNE visualization PNG (r_image_tsne_plot.png).

Text files with model performance summaries and confidence interval analyses.

How to Use
Run the runtime and dependency cells to verify RAM/GPU and install umap-learn.

Load the Galaxy Zoo data and construct X_subset and y_subset as shown in the notebook.

Execute the dimensionality‑reduction sections (t‑SNE, UMAP) to explore structures in the labels.

Train models using the provided utilities; use the summary and visualization functions to interpret and compare results.

Export plots and summaries for inclusion in reports, publications, or patent‑related documentation.

