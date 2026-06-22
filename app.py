"""
Artist Reference Asset Organizer

A local desktop tool that helps digital artists organize large folders of
reference images. The app scans an image folder, suggests categories, sorts
files into output folders, and saves category fixes for future sorting.

Run:
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageTk, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_NAME = "Artist Reference Asset Organizer"
DB_FILE = "artist_reference_asset_organizer.db"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CATEGORIES: Dict[str, List[str]] = {
    "Hands": [
        "artist reference image focused on hands, fingers, palms, wrists, hand gestures, or hand anatomy",
        "close-up hand pose reference for artists",
        "human hands reference image for illustration",
    ],
    "Body Poses": [
        "a full body pose reference for drawing, anatomy, action pose, standing pose, sitting pose, or gesture drawing",
        "human figure pose reference for artists",
        "dynamic body pose reference image",
    ],
    "Clothing": [
        "clothing reference image, outfit design, costume, fabric folds, armor, jacket, dress, shoes, or fashion reference",
        "outfit and costume reference for character design",
        "fabric and clothing detail reference image",
    ],
    "Accessories": [
        "small accessory reference image, jewelry, belt, bag, weapon prop, ornament, tool, buckle, charm, or decorative item",
        "small props and accessories reference for character design",
        "artist reference for objects, ornaments, tools, bags, belts, or jewelry",
    ],
    "Creatures and Enemies": [
        "creature reference, monster design, enemy concept, beast, demon, alien, fantasy creature, or horror creature",
        "fantasy monster creature design reference image",
        "enemy creature reference for game art or illustration",
    ],
}

NEEDS_REVIEW = "Review Needed"


@dataclass
class ClassificationResult:
    source_path: str
    file_hash: str
    predicted_category: str
    final_category: str
    confidence: float
    destination_path: str
    used_feedback: bool


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    final_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    destination_path TEXT,
                    was_corrected INTEGER NOT NULL DEFAULT 0,
                    used_feedback INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    corrected_category TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.commit()

    def save_result(self, result: ClassificationResult):
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO images (
                    source_path, file_hash, predicted_category, final_category,
                    confidence, destination_path, was_corrected, used_feedback, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.source_path,
                    result.file_hash,
                    result.predicted_category,
                    result.final_category,
                    float(result.confidence),
                    result.destination_path,
                    0,
                    1 if result.used_feedback else 0,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            con.commit()

    def save_correction(
        self,
        file_hash: str,
        original_path: str,
        predicted_category: str,
        corrected_category: str,
        embedding: np.ndarray,
    ):
        emb = embedding.astype(np.float32).tobytes()
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO corrections (
                    file_hash, original_path, predicted_category,
                    corrected_category, embedding, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_hash,
                    original_path,
                    predicted_category,
                    corrected_category,
                    emb,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            con.execute(
                """
                UPDATE images
                SET final_category = ?, was_corrected = 1
                WHERE file_hash = ?
                """,
                (corrected_category, file_hash),
            )
            con.commit()

    def load_corrections(self) -> List[Tuple[str, np.ndarray]]:
        rows: List[Tuple[str, bytes]] = []
        with self.connect() as con:
            cur = con.execute("SELECT corrected_category, embedding FROM corrections")
            rows = cur.fetchall()

        corrections: List[Tuple[str, np.ndarray]] = []
        for category, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            corrections.append((category, emb))
        return corrections


class ArtRefClassifier:
    def __init__(self, db: Database, model_name: str = "openai/clip-vit-base-patch32"):
        self.db = db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.category_names = list(CATEGORIES.keys())
        self.prompt_texts = []
        self.prompt_to_category = []

        for category, prompts in CATEGORIES.items():
            for prompt in prompts:
                self.prompt_texts.append(prompt)
                self.prompt_to_category.append(category)

        self.text_features = self._encode_text_prompts(self.prompt_texts)

    def _encode_text_prompts(self, prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features

    def image_embedding(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features[0].detach().cpu().numpy().astype(np.float32)

    def classify(
        self,
        image_path: Path,
        confidence_threshold: float = 0.26,
        feedback_weight: float = 0.10,
    ) -> Tuple[str, float, np.ndarray, bool]:
        embedding = self.image_embedding(image_path)
        image_tensor = torch.tensor(embedding, device=self.device).unsqueeze(0)

        with torch.no_grad():
            similarities = (image_tensor @ self.text_features.T).squeeze(0)
            prompt_scores = similarities.detach().cpu().numpy()

        category_scores: Dict[str, float] = {cat: -999.0 for cat in self.category_names}
        for score, category in zip(prompt_scores, self.prompt_to_category):
            category_scores[category] = max(category_scores[category], float(score))

        used_feedback = False
        corrections = self.db.load_corrections()
        if corrections:
            used_feedback = True
            feedback_scores: Dict[str, List[float]] = {cat: [] for cat in self.category_names}
            for corrected_category, corrected_embedding in corrections:
                if corrected_category not in feedback_scores:
                    continue
                denom = (np.linalg.norm(embedding) * np.linalg.norm(corrected_embedding))
                sim = float(np.dot(embedding, corrected_embedding) / denom) if denom else 0.0
                feedback_scores[corrected_category].append(sim)

            for category, values in feedback_scores.items():
                if values:
                    # Estimate how confident the sorter is by checking the gap between the best match and the next best match.
                    top_values = sorted(values, reverse=True)[:3]
                    category_scores[category] += feedback_weight * float(np.mean(top_values))

        sorted_scores = sorted(category_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_category, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1

        # Confidence is not a literal probability. It is a pragmatic margin-based score.
        margin = best_score - second_score
        confidence = float(max(0.0, min(1.0, 0.5 + margin * 5.0)))

        if best_score < confidence_threshold or confidence < 0.53:
            return NEEDS_REVIEW, confidence, embedding, used_feedback

        return best_category, confidence, embedding, used_feedback


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_images(input_folder: Path) -> List[Path]:
    images: List[Path] = []
    for root, _, files in os.walk(input_folder):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(path)
    return sorted(images)


def safe_destination(output_folder: Path, category: str, source_path: Path) -> Path:
    category_folder = output_folder / category
    category_folder.mkdir(parents=True, exist_ok=True)

    candidate = category_folder / source_path.name
    if not candidate.exists():
        return candidate

    stem = source_path.stem
    suffix = source_path.suffix
    counter = 2
    while True:
        candidate = category_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1240x790")
        self.minsize(1040, 690)

        self.project_dir = Path.cwd()
        self.db = Database(self.project_dir / DB_FILE)
        self.classifier: Optional[ArtRefClassifier] = None

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.mode = tk.StringVar(value="copy")
        self.confidence_threshold = tk.DoubleVar(value=0.26)
        self.threshold_display = tk.StringVar(value="0.26")
        self.status_text = tk.StringVar(value="Ready. Load the model, choose folders, then scan.")

        self.results: List[ClassificationResult] = []
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.selected_preview: Optional[ImageTk.PhotoImage] = None

        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        self.colors = {
            "ink": "#f3f1ec",
            "muted": "#a7a3ae",
            "line": "#34333c",
            "outline": "#000000",
            "paper": "#2b2a33",
            "surface": "#36343e",
            "workspace": "#1f1e25",
            "selected": "#41404a",
            "scale_thumb": "#000000",
            "primary": "#16151b",
            "grid": "#302f38",
        }
        self.configure(background=self.colors["workspace"])

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 9), background=self.colors["paper"], foreground=self.colors["ink"])
        style.configure("Sidebar.TFrame", background=self.colors["paper"])
        style.configure("Workspace.TFrame", background=self.colors["workspace"])
        style.configure(
            "Card.TFrame",
            background=self.colors["paper"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            relief="solid",
            borderwidth=1,
        )
        style.configure("Title.TLabel", background=self.colors["paper"], font=("Segoe UI Semibold", 12))
        style.configure(
            "Section.TLabel",
            background=self.colors["paper"],
            foreground=self.colors["muted"],
            font=("Segoe UI Semibold", 8),
        )
        style.configure("Body.TLabel", background=self.colors["paper"], font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=self.colors["paper"], foreground=self.colors["muted"], font=("Segoe UI", 8))
        style.configure("CardTitle.TLabel", background=self.colors["paper"], font=("Segoe UI Semibold", 11))
        style.configure("WorkspaceTitle.TLabel", background=self.colors["workspace"], font=("Segoe UI Semibold", 13))
        style.configure("Status.TLabel", background=self.colors["paper"], foreground=self.colors["muted"], wraplength=270)
        style.configure(
            "TEntry",
            fieldbackground=self.colors["paper"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            padding=(8, 7),
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["paper"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            padding=(7, 6),
        )
        style.map("TCombobox", fieldbackground=[("readonly", self.colors["paper"])])
        style.configure(
            "TButton",
            background=self.colors["paper"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            padding=(12, 8),
            relief="solid",
        )
        style.map("TButton", background=[("active", self.colors["surface"])])
        style.configure(
            "Primary.TButton",
            background=self.colors["primary"],
            foreground=self.colors["ink"],
            bordercolor=self.colors["primary"],
            padding=(12, 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["workspace"]), ("pressed", self.colors["primary"])],
            foreground=[("active", self.colors["ink"])],
        )
        style.configure("Small.TButton", font=("Segoe UI Semibold", 8), padding=(8, 7))
        style.configure(
            "Segment.TRadiobutton",
            background=self.colors["paper"],
            foreground=self.colors["ink"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            borderwidth=1,
            padding=(20, 8),
            relief="solid",
        )
        style.layout(
            "Segment.TRadiobutton",
            [
                (
                    "Button.border",
                    {
                        "sticky": "nswe",
                        "children": [
                            (
                                "Radiobutton.padding",
                                {"sticky": "nswe", "children": [("Radiobutton.label", {"sticky": "nswe"})]},
                            )
                        ],
                    },
                )
            ],
        )
        style.map(
            "Segment.TRadiobutton",
            background=[("selected", self.colors["primary"]), ("active", self.colors["surface"])],
            foreground=[("selected", self.colors["ink"])],
            bordercolor=[("selected", self.colors["line"])],
        )
        style.configure(
            "Strictness.Horizontal.TScale",
            background=self.colors["scale_thumb"],
            troughcolor=self.colors["line"],
            bordercolor=self.colors["scale_thumb"],
            lightcolor=self.colors["scale_thumb"],
            darkcolor=self.colors["scale_thumb"],
        )
        style.map("Strictness.Horizontal.TScale", background=[("active", self.colors["scale_thumb"])])
        style.configure(
            "Treeview",
            background=self.colors["paper"],
            fieldbackground=self.colors["paper"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            borderwidth=1,
            relief="solid",
            rowheight=31,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=self.colors["surface"],
            foreground=self.colors["muted"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            font=("Segoe UI Semibold", 8),
            padding=(8, 9),
        )
        style.map("Treeview", background=[("selected", self.colors["selected"])], foreground=[("selected", self.colors["ink"])])

    def _sidebar_section(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        tk.Frame(parent, height=1, background=self.colors["outline"], borderwidth=0, highlightthickness=0).pack(fill="x")
        section = ttk.Frame(parent, style="Sidebar.TFrame", padding=(20, 13, 20, 15))
        section.pack(fill="x")
        ttk.Label(section, text=title.upper(), style="Section.TLabel").pack(anchor="w", pady=(0, 11))
        return section

    def _folder_field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, command):
        ttk.Label(parent, text=label.upper(), style="Muted.TLabel").pack(anchor="w", pady=(0, 4))
        row = ttk.Frame(parent, style="Sidebar.TFrame")
        row.pack(fill="x", pady=(0, 10))
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="...", command=command, style="Small.TButton", width=3).pack(side="left", padx=(5, 0))

    def _build_ui(self):
        shell = ttk.Frame(self, style="Workspace.TFrame")
        shell.pack(fill="both", expand=True)

        sidebar = ttk.Frame(shell, style="Sidebar.TFrame", width=322)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        header = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(20, 14, 20, 13))
        header.pack(fill="x")
        ttk.Label(header, text="REFERENCE SORTER", style="Title.TLabel").pack(side="left")

        folders = self._sidebar_section(sidebar, "Folders")
        self._folder_field(folders, "Input", self.input_folder, self.browse_input)
        self._folder_field(folders, "Output", self.output_folder, self.browse_output)

        actions = self._sidebar_section(sidebar, "Actions")
        action_buttons = ttk.Frame(actions, style="Sidebar.TFrame")
        action_buttons.pack(fill="x", pady=(0, 13))
        ttk.Button(action_buttons, text="LOAD MODEL", command=self.load_model_thread).pack(side="left", fill="x", expand=True)
        ttk.Button(action_buttons, text="SCAN + SORT", command=self.scan_thread, style="Primary.TButton").pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )
        ttk.Label(actions, text="FILE MODE", style="Muted.TLabel").pack(anchor="w", pady=(0, 5))
        mode_row = ttk.Frame(actions, style="Sidebar.TFrame")
        mode_row.pack(fill="x")
        ttk.Radiobutton(mode_row, text="Copy", value="copy", variable=self.mode, style="Segment.TRadiobutton").pack(
            side="left", fill="x", expand=True
        )
        ttk.Radiobutton(mode_row, text="Move", value="move", variable=self.mode, style="Segment.TRadiobutton").pack(
            side="left", fill="x", expand=True
        )

        sort = self._sidebar_section(sidebar, "Sorting")
        threshold_row = ttk.Frame(sort, style="Sidebar.TFrame")
        threshold_row.pack(fill="x", pady=(0, 8))
        ttk.Label(threshold_row, text="STRICTNESS", style="Muted.TLabel").pack(side="left")
        ttk.Label(threshold_row, textvariable=self.threshold_display, style="Body.TLabel").pack(side="right")
        ttk.Scale(
            sort,
            from_=0.15,
            to=0.40,
            variable=self.confidence_threshold,
            orient="horizontal",
            command=self._update_threshold_label,
            style="Strictness.Horizontal.TScale",
        ).pack(fill="x", pady=(0, 7))
        ttk.Label(sort, text="Higher values send uncertain files to review.", style="Muted.TLabel").pack(anchor="w")

        status = self._sidebar_section(sidebar, "Status")
        ttk.Label(status, textvariable=self.status_text, style="Status.TLabel", justify="left").pack(anchor="w")

        workspace = ttk.Frame(shell, style="Workspace.TFrame")
        workspace.pack(side="left", fill="both", expand=True)
        self.workspace_canvas = tk.Canvas(workspace, background=self.colors["workspace"], highlightthickness=0)
        self.workspace_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.workspace_canvas.bind("<Configure>", self._draw_workspace_grid)

        work_header = ttk.Frame(workspace, style="Workspace.TFrame", padding=(28, 20, 28, 16))
        work_header.pack(fill="x")
        ttk.Label(work_header, text="SORTED REFERENCES", style="WorkspaceTitle.TLabel").pack(side="left")

        results_card = ttk.Frame(workspace, style="Card.TFrame", padding=(15, 12, 15, 15))
        results_card.pack(fill="both", expand=True, padx=(28, 16))
        ttk.Label(results_card, text="Results", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))

        columns = ("file", "predicted", "final", "confidence", "feedback", "destination")
        table = ttk.Frame(results_card, style="Sidebar.TFrame")
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=14)
        for col in columns:
            self.tree.heading(col, text=col.upper())
        self.tree.column("file", width=210)
        self.tree.column("predicted", width=130)
        self.tree.column("final", width=130)
        self.tree.column("confidence", width=88)
        self.tree.column("feedback", width=80)
        self.tree.column("destination", width=260)
        tree_scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        detail_card = ttk.Frame(workspace, style="Card.TFrame", padding=(15, 12, 15, 15))
        detail_card.pack(fill="x", padx=28, pady=(16, 24))
        preview = ttk.Frame(detail_card, style="Sidebar.TFrame", width=196, height=146)
        preview.pack(side="left", fill="y", padx=(0, 18))
        preview.pack_propagate(False)
        ttk.Label(preview, text="PREVIEW", style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        self.preview_label = ttk.Label(preview, text="Select a result", anchor="center", style="Muted.TLabel")
        self.preview_label.pack(fill="both", expand=True)

        correction = ttk.Frame(detail_card, style="Sidebar.TFrame")
        correction.pack(side="left", fill="both", expand=True)
        ttk.Label(correction, text="CORRECTION", style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        ttk.Label(correction, text="Correct Category", style="Body.TLabel").pack(anchor="w")
        self.correct_category = tk.StringVar(value="Hands")
        ttk.Combobox(
            correction,
            textvariable=self.correct_category,
            values=list(CATEGORIES.keys()) + [NEEDS_REVIEW],
            state="readonly",
            width=30,
        ).pack(anchor="w", pady=(5, 10))

        ttk.Button(correction, text="SAVE CORRECTION", command=self.save_selected_correction, style="Primary.TButton").pack(
            anchor="w"
        )

        explanation = (
            "Corrections are saved locally.\n"
            "Future scans compare images against examples you have fixed."
        )
        ttk.Label(correction, text=explanation, style="Muted.TLabel", justify="left").pack(anchor="w", pady=(10, 0))

    def _draw_workspace_grid(self, event):
        self.workspace_canvas.delete("grid")
        for x in range(18, event.width, 24):
            for y in range(18, event.height, 24):
                self.workspace_canvas.create_oval(
                    x,
                    y,
                    x + 1,
                    y + 1,
                    fill=self.colors["grid"],
                    outline=self.colors["grid"],
                    tags="grid",
                )

    def _update_threshold_label(self, _value=None):
        self.threshold_display.set(f"{self.confidence_threshold.get():.2f}")

    def browse_input(self):
        folder = filedialog.askdirectory(title="Choose input folder")
        if folder:
            self.input_folder.set(folder)
            if not self.output_folder.get():
                self.output_folder.set(str(Path(folder).parent / "Sorted Output"))

    def browse_output(self):
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self.output_folder.set(folder)

    def load_model_thread(self):
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            self.status_text.set("Loading image sorter. This may take a moment the first time.")
            self.classifier = ArtRefClassifier(self.db)
            self.status_text.set(f"Model loaded on {self.classifier.device}.")
        except Exception as exc:
            self.status_text.set("Model failed to load.")
            messagebox.showerror("Model Load Error", str(exc))

    def scan_thread(self):
        threading.Thread(target=self.scan_and_sort, daemon=True).start()

    def scan_and_sort(self):
        if self.classifier is None:
            self.status_text.set("Load the AI model first.")
            return

        input_folder = Path(self.input_folder.get())
        output_folder = Path(self.output_folder.get())

        if not input_folder.exists() or not input_folder.is_dir():
            messagebox.showerror("Input Error", "Choose a valid input folder.")
            return

        output_folder.mkdir(parents=True, exist_ok=True)

        image_paths = find_images(input_folder)
        if not image_paths:
            self.status_text.set("No supported images found.")
            return

        self.results.clear()
        self.embedding_cache.clear()
        self.tree.delete(*self.tree.get_children())

        for idx, image_path in enumerate(image_paths, start=1):
            try:
                self.status_text.set(f"Processing {idx}/{len(image_paths)}: {image_path.name}")
                file_hash = compute_file_hash(image_path)
                category, confidence, embedding, used_feedback = self.classifier.classify(
                    image_path,
                    confidence_threshold=self.confidence_threshold.get(),
                )

                destination = safe_destination(output_folder, category, image_path)

                if self.mode.get() == "move":
                    shutil.move(str(image_path), str(destination))
                    source_for_record = str(destination)
                else:
                    shutil.copy2(str(image_path), str(destination))
                    source_for_record = str(image_path)

                result = ClassificationResult(
                    source_path=source_for_record,
                    file_hash=file_hash,
                    predicted_category=category,
                    final_category=category,
                    confidence=confidence,
                    destination_path=str(destination),
                    used_feedback=used_feedback,
                )
                self.results.append(result)
                self.embedding_cache[file_hash] = embedding
                self.db.save_result(result)
                self.add_result_to_table(result)

            except UnidentifiedImageError:
                continue
            except Exception as exc:
                print(f"Error processing {image_path}: {exc}")

        self.status_text.set(f"Complete. Sorted {len(self.results)} image(s).")

    def add_result_to_table(self, result: ClassificationResult):
        self.tree.insert(
            "",
            "end",
            iid=result.file_hash,
            values=(
                Path(result.source_path).name,
                result.predicted_category,
                result.final_category,
                f"{result.confidence:.2f}",
                "yes" if result.used_feedback else "no",
                result.destination_path,
            ),
        )

    def selected_result(self) -> Optional[ClassificationResult]:
        selection = self.tree.selection()
        if not selection:
            return None
        file_hash = selection[0]
        for result in self.results:
            if result.file_hash == file_hash:
                return result
        return None

    def on_select(self, _event=None):
        result = self.selected_result()
        if result is None:
            return

        path = Path(result.destination_path)
        if not path.exists():
            path = Path(result.source_path)

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((184, 108))
            self.selected_preview = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.selected_preview, text="")
            self.correct_category.set(result.final_category)
        except Exception:
            self.preview_label.configure(image="", text="Preview unavailable")

    def save_selected_correction(self):
        result = self.selected_result()
        if result is None:
            messagebox.showinfo("No Selection", "Select an image row first.")
            return

        corrected = self.correct_category.get()
        if not corrected:
            return

        embedding = self.embedding_cache.get(result.file_hash)
        if embedding is None:
            path = Path(result.destination_path)
            if not path.exists():
                path = Path(result.source_path)
            if self.classifier is None:
                return
            embedding = self.classifier.image_embedding(path)

        self.db.save_correction(
            file_hash=result.file_hash,
            original_path=result.source_path,
            predicted_category=result.predicted_category,
            corrected_category=corrected,
            embedding=embedding,
        )

        result.final_category = corrected
        self.tree.item(
            result.file_hash,
            values=(
                Path(result.source_path).name,
                result.predicted_category,
                result.final_category,
                f"{result.confidence:.2f}",
                "yes" if result.used_feedback else "no",
                result.destination_path,
            ),
        )
        self.status_text.set(f"Correction saved: {Path(result.source_path).name} -> {corrected}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
