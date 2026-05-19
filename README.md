# Artist Reference Asset Organizer

A local desktop tool that helps digital artists organize large folders of reference images.

The app scans an image folder, suggests practical categories, and sorts files into organized output folders. If a category is wrong, the user can fix it. Corrections are saved locally so future sorting can better match the artist’s reference library.

## Features

- Sorts images into categories such as Hands, Body Poses, Clothing, Accessories, Creatures and Enemies, and Review Needed
- Supports copy or move mode
- Saves scan history and category fixes in SQLite
- Uses CLIP image embeddings for local image classification
- Uses corrected examples to improve future sorting behavior

## Tech Stack

Python, Tkinter, PyTorch, Hugging Face Transformers, CLIP, SQLite, Pillow, NumPy

## Run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py