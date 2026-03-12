import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict import predict_image


def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if file_path:

        # Show image preview
        img = Image.open(file_path)
        img = img.resize((240,240))
        img = ImageTk.PhotoImage(img)

        image_label.config(image=img)
        image_label.image = img

        # Prediction
        result = predict_image(file_path)

        if result == "Malignant":
            result_label.config(
                text="⚠ Malignant Detected",
                bg="#e74c3c",
                fg="white"
            )

        elif result == "Benign":
            result_label.config(
                text="✔ Benign Tumor",
                bg="#27ae60",
                fg="white"
            )

        else:
            result_label.config(
                text="ℹ Normal Tissue",
                bg="#3498db",
                fg="white"
            )


def clear_ui():
    image_label.config(image="")
    result_label.config(
        text="Prediction will appear here",
        bg="#ecf0f1",
        fg="black"
    )


root = tk.Tk()
root.title("Breast Cancer Detection System")
root.geometry("620x620")
root.configure(bg="#f5f6fa")

# ================= TITLE =================

title = tk.Label(
    root,
    text="Breast Cancer Detection System",
    font=("Arial",22,"bold"),
    bg="#f5f6fa",
    fg="#2c3e50"
)
title.pack(pady=15)

desc = tk.Label(
    root,
    text="Upload a breast ultrasound image for prediction",
    font=("Arial",11),
    bg="#f5f6fa"
)
desc.pack(pady=5)

# ================= BUTTONS =================

btn_frame = tk.Frame(root, bg="#f5f6fa")
btn_frame.pack(pady=15)

upload_btn = tk.Button(
    btn_frame,
    text="Upload Image",
    font=("Arial",12,"bold"),
    bg="#0984e3",
    fg="white",
    padx=20,
    pady=10,
    command=upload_image
)
upload_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(
    btn_frame,
    text="Clear",
    font=("Arial",12,"bold"),
    bg="#636e72",
    fg="white",
    padx=20,
    pady=10,
    command=clear_ui
)
clear_btn.grid(row=0, column=1, padx=10)

# ================= IMAGE PREVIEW =================

image_label = tk.Label(root, bg="#f5f6fa")
image_label.pack(pady=20)

# ================= RESULT =================

result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial",16,"bold"),
    bg="#ecf0f1",
    width=28,
    pady=12
)
result_label.pack(pady=20)

# ================= FOOTER =================

footer = tk.Label(
    root,
    text="Deep Learning Model for Breast Cancer Detection",
    font=("Arial",9),
    bg="#f5f6fa",
    fg="#7f8c8d"
)
footer.pack(side="bottom", pady=10)

root.mainloop()