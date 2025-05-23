# SimCLR + Caption Retrieval on Flickr30k

This project investigates the application of **SimCLR** (a self-supervised contrastive learning model) for **vision-language representation learning** using object crops from the Flickr30k dataset. It explores how visual embeddings and natural language captions align in the latent space.

## 🖼️ Dataset

- **Flickr30k Object Crops**
- Preprocessed and embedded using a pretrained SimCLR model
- Captions cleaned and paired with image crop IDs

## 🔧 Project Highlights

- Loads and processes pretrained embeddings (`simclr_epoch5.pt`)
- Retrieves and pairs embeddings with captions
- Potential tasks:
  - Image-to-caption retrieval
  - Caption-to-image matching
  - Embedding visualization or clustering

## 🧠 Methods

- Contrastive Representation Learning (SimCLR)
- Embedding analysis using:
  - Cosine similarity
  - k-NN
  - PCA/TSNE (for visualization)

## 📁 Artifacts

- `simclr_epoch5.pt`: pretrained SimCLR checkpoint
- `embeddings.npy`: NumPy array of visual features
- `embedding_filenames.txt`: order of images in embeddings
- `cleaned_captions.txt`: paired captions for retrieval/analysis

## 📈 Visualization

- Sample crop visualization
- Embedding similarity heatmaps
- TSNE/PCA plots of embedding clusters

## 🧰 Dependencies

```bash
pip install pandas numpy matplotlib patool pyunpack
