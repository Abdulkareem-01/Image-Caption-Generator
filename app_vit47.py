# app_vit47.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch

from transformers import ViTImageProcessor, T5Tokenizer
from model_vit47 import QFormerCaptionModel

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# PROCESSOR & TOKENIZER
# ===============================
image_processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

tokenizer = T5Tokenizer.from_pretrained(
    "google/flan-t5-base"
)

# ===============================
# LOAD TRAINED MODEL (Q=16)
# ===============================
CHECKPOINT_PATH = r"C:\Users\kittu\Desktop\Image caption generator\epoch_vit_4.pth"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

model = QFormerCaptionModel(num_query_tokens=16).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(" QFormer-ViT model (q16) loaded")

# ===============================
# CAPTION FUNCTION
# ===============================
@torch.no_grad()
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    pixel_values = image_processor(
        image, return_tensors="pt"
    ).pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        tokenizer=tokenizer,
        max_length=40
    )

    caption = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return caption, image

# ===============================
# TKINTER UI
# ===============================
root = tk.Tk()
root.title("Image Caption Generator (ViT-QFormer)")
root.geometry("900x850")
root.configure(bg="white")

HEADER_COLOR = "#51A2FF"

header = tk.Frame(root, bg=HEADER_COLOR, height=85)
header.pack(fill="x")

title_label = tk.Label(
    header,
    text="Image Caption Generator",
    font=("Helvetica", 28, "bold"),
    bg=HEADER_COLOR,
    fg="white"
)
title_label.pack(pady=20)

main_frame = tk.Frame(root, bg="white")
main_frame.pack(pady=30)

img_label = tk.Label(main_frame, bg="white")
img_label.pack(pady=20)

caption_frame = tk.Frame(
    main_frame,
    bg="white",
    highlightthickness=2,
    highlightbackground="#d9d9d9",
    padx=20,
    pady=10
)
caption_frame.pack(pady=10)

caption_title = tk.Label(
    caption_frame,
    text="Generated Caption",
    font=("Helvetica", 18, "bold"),
    bg="white",
    fg="#333"
)
caption_title.pack()

caption_box = tk.Text(
    caption_frame,
    wrap="word",
    font=("Helvetica", 15),
    bg="white",
    fg="#444",
    height=4,
    width=60,
    bd=0
)
caption_box.pack(pady=10)
caption_box.config(state="disabled")

def choose_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")]
    )
    if not path:
        return

    caption, image = generate_caption(path)

    preview = image.copy()
    preview.thumbnail((550, 350))
    tk_img = ImageTk.PhotoImage(preview)

    img_label.config(image=tk_img)
    img_label.image = tk_img

    caption_box.config(state="normal")
    caption_box.delete("1.0", tk.END)
    caption_box.insert(tk.END, caption)
    caption_box.config(state="disabled")

upload_button = tk.Button(
    root,
    text="Select Image",
    font=("Helvetica", 18, "bold"),
    bg="#2ecc71",
    fg="white",
    padx=20,
    pady=12,
    bd=0,
    activebackground="#27ae60",
    cursor="hand2",
    command=choose_image
)
upload_button.pack(pady=20)

root.mainloop()
