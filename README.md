# Artist Reference Asset Organizer

A local desktop tool that helps digital artists turn messy reference folders into organized, usable image libraries.

![Hand reference sorting example](demonstration/Sample%20Screenshot%20Hand.jpg)

Artists often collect hundreds or thousands of reference images for anatomy, clothing, accessories, and creatures. This application saves valuable time by sorting the reference images automatically.

[Watch the demonstration video](demonstration/Artist%20Reference%20Organizer%20Demonstration%20Video.mp4)

## Built for Real Artist Workflows

The app sorts images into practical categories such as Hands, Body Poses, Clothing, Accessories, and Creatures/Enemies. If the program is uncertain then it will move the image into the Review Needed category. It supports copy or move mode, so the artist can either preserve the original folder or reorganize it directly.

![Clothing reference sorting example](demonstration/Sample%20Screenshot%20Clothing.jpg)

## Local Sorting With Saved Corrections

The sorter uses local image classification to suggest categories, then saves category fixes in SQLite. When the user corrects an image, that correction becomes an example the app can compare against in future scans.

![Category correction screen](demonstration/Sample%20Screenshot%20Correction.jpg)

## Designed to Reduce Manual Sorting
Accessories, body poses, clothing, hands, and creature references can be tedious to organize one by one. This tool gives artists a faster way to clean up their reference folders so they can spend more time creating.

![Accessory reference sorting example](demonstration/Sample%20Screenshot%20Accessories.jpg)

## Features

* Sorts image folders into artist-focused categories
* Supports copy or move mode
* Sends uncertain images to Review Needed
* Saves scan history and category fixes locally
* Uses CLIP image embeddings for image classification
* Uses corrected examples to improve future sorting behavior

## Tech Stack

Python, Tkinter, PyTorch, Hugging Face Transformers, CLIP, SQLite, Pillow, NumPy

## Run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The first model load may take a while because the image model has to download.

## Status

Working desktop application built to reduce manual sorting work for digital artists and demonstrate practical use of computer vision, file automation, local storage, and user-corrected classification.
