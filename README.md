# Fake News Detection

This project implements a fake news detection pipeline using graph-based learning on the GossipCop benchmark, where news spread is modeled as a propagation graph.

## Project Highlights

- Developed a fake news detection model using the UPFD architecture on the GossipCop dataset, modeling news propagation as graph-structured data.
- Built and trained Graph Neural Networks (GNNs) with GraphSAGE layers in PyTorch Geometric, utilizing NetworkX for propagation graph construction.
- Fused BERT-derived textual embeddings with topological graph features to jointly capture semantic meaning and structural diffusion patterns.
- Engineered an end-to-end PyTorch pipeline for graph processing, model training, and evaluation, optimizing for accuracy and F1-score.

## Approach

The workflow combines content and structure signals:

1. Text representation: BERT embeddings are used to encode article-level semantics.
2. Propagation modeling: user interaction and diffusion information are represented as graphs.
3. Graph learning: GraphSAGE-based message passing learns node and graph-level patterns.
4. Multimodal fusion: textual and graph features are combined for robust fake news classification.

## Tech Stack

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- Transformers (BERT)
- Jupyter Notebook

## Dataset

- UPFD (User Preference-aware Fake News Detection)
- GossipCop split

## Results

The following scores are taken from the saved output logs in `fake_news.ipynb`.

- Train samples: 1092
- Test samples: 3826
- Epochs: 40 (0-39)

| Metric | Best Value | Final Epoch (39) |
| --- | --- | --- |
| Test Accuracy | 0.93 | 0.93 |
| Test F1-score | 0.93 | 0.93 |
| Test Loss | 0.30 (lowest) | 0.30 |

### Training Snapshot

- Early training starts near random baseline (TestAcc ~0.50, TestF1 ~0.00).
- Performance improves significantly after mid-training and stabilizes in later epochs.
- Strong checkpoints include epochs 32, 35, and 39 with high Accuracy/F1.

## Repository Structure

- fake_news.ipynb: end-to-end experimentation notebook for data processing, model training, and evaluation.

## Running the Project

1. Open the notebook fake_news.ipynb.
2. Install required dependencies in your environment.
3. Run cells in order to preprocess data, construct graphs, train the model, and evaluate performance.

## Notes

- The model is designed to leverage both linguistic cues and social diffusion behavior.
- Accuracy and F1-score are treated as primary evaluation metrics.