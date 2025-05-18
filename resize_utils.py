import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import os

def resize_image(edge, new_width, new_height):
    """
    Zmienia rozmiar obrazu PIL do nowych wymiarów.
    Zwraca nowy obraz PIL.
    """
    resized = edge.resize((new_width, new_height), Image.LANCZOS)
    return resized

def ask_resize(root, edge, img_width, img_height, update_callback):
    """
    Pokazuje okno dialogowe do wpisania nowych wymiarów.
    Po zatwierdzeniu wywołuje update_callback z nowym obrazem i wymiarami.
    """
    if edge is None:
        messagebox.showinfo("Brak obrazu", "Najpierw otwórz obraz.")
        return

    resize_win = tk.Toplevel(root)
    resize_win.title("Resize Image")

    tk.Label(resize_win, text="Szerokość:").grid(row=0, column=0, padx=5, pady=5)
    width_entry = tk.Entry(resize_win)
    width_entry.insert(0, str(img_width))
    width_entry.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(resize_win, text="Wysokość:").grid(row=1, column=0, padx=5, pady=5)
    height_entry = tk.Entry(resize_win)
    height_entry.insert(0, str(img_height))
    height_entry.grid(row=1, column=1, padx=5, pady=5)

    def on_resize():
        try:
            w = int(width_entry.get())
            h = int(height_entry.get())
            if w <= 0 or h <= 0:
                raise ValueError("Wymiary muszą być dodatnie")
            new_edge = resize_image(edge, w, h)
            update_callback(new_edge, w, h)
            resize_win.destroy()
        except Exception as e:
            messagebox.showerror("Błąd", f"Niepoprawne wymiary: {e}")

    tk.Button(resize_win, text="Zmień rozmiar", command=on_resize).grid(row=2, column=0, columnspan=2, pady=10)

def batch_resize_images():
    """
    Okno dialogowe pozwala wybrać wiele obrazów.
    Zmienia ich rozmiar według podanych wymiarów.
    Zapisuje nowe pliki z dopiskiem _RESIZED w nazwie.
    """
    file_paths = filedialog.askopenfilenames(
        title="Select Images to Resize",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )
    if not file_paths:
        return

    resize_win = tk.Toplevel()
    resize_win.title("Batch Resize Images")

    tk.Label(resize_win, text="Szerokość:").grid(row=0, column=0, padx=5, pady=5)
    width_entry = tk.Entry(resize_win)
    width_entry.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(resize_win, text="Wysokość:").grid(row=1, column=0, padx=5, pady=5)
    height_entry = tk.Entry(resize_win)
    height_entry.grid(row=1, column=1, padx=5, pady=5)

    def on_batch_resize():
        try:
            w = int(width_entry.get())
            h = int(height_entry.get())
            if w <= 0 or h <= 0:
                raise ValueError("Wymiary muszą być dodatnie")

            success_count = 0
            error_files = []

            for path in file_paths:
                try:
                    img = Image.open(path)
                    resized = resize_image(img, w, h)
                    base, ext = os.path.splitext(path)
                    new_path = base + f"_RESIZED{ext}"
                    resized.save(new_path)
                    success_count += 1
                except Exception as e:
                    error_files.append((os.path.basename(path), str(e)))

            summary = f"{success_count} plików zmieniono rozmiar."
            if error_files:
                summary += "\n\nBłędy dla plików:\n"
                for fname, err in error_files:
                    summary += f"{fname}: {err}\n"

            messagebox.showinfo("Batch Resize Completed", summary)
            resize_win.destroy()

        except Exception as e:
            messagebox.showerror("Błąd", f"Niepoprawne wymiary: {e}")

    tk.Button(resize_win, text="Resize All", command=on_batch_resize).grid(row=2, column=0, columnspan=2, pady=10)