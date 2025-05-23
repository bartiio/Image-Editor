# main.py
import tkinter as tk
from tkinter import ttk 
from tkinter import Menu, filedialog, messagebox, font
from PIL import Image, ImageTk
import cv2
import numpy as np
import tempfile
import os
from Watermark import Water  
from histogram import HistogramViewer
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from resize_utils import ask_resize, batch_resize_images
from siec_osoby import run_detection

undo_stack = []
redo_stack = []

root = tk.Tk()
root.geometry("1000x800+300+150")
root.resizable(width=True, height=True)
root.title('Image Editor')


# Styl menu (czcionka, kolory)
default_font = font.nametofont("TkMenuFont")
default_font.configure(size=11, family="Segoe UI")  # lub inna nowoczesna czcionka

root.option_add("*Menu*Font", default_font)
root.option_add("*Menu*Background", "#f0f0f0")
root.option_add("*Menu*Foreground", "#333333")
root.option_add("*Menu*ActiveBackground", "#007acc")
root.option_add("*Menu*ActiveForeground", "#ffffff")
root.option_add("*Menu*BorderWidth", 0)


canvas = tk.Canvas(root, bg='white')
canvas.pack(fill=tk.BOTH, expand=True)
hist_frame = tk.Frame(root, bg='white', height=200)
hist_frame.pack(fill=tk.X, side=tk.BOTTOM)
hist_label = tk.Label(hist_frame, bg='white')
hist_label.pack()

image_id = None
img = None
edge = None
show_hist = tk.BooleanVar(value=True)
img_width = 0
img_height = 0

def push_undo(image_pil):
    global undo_stack, redo_stack
    undo_stack.append(image_pil.copy())
    redo_stack.clear()  

def undo():
    global edge, img, image_id, undo_stack, redo_stack
    if not undo_stack:
        messagebox.showinfo("Undo", "Brak dalszych cofnięć")
        return
    redo_stack.append(edge.copy())  # zapisujemy obecny stan do redo
    edge = undo_stack.pop()
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def redo():
    global edge, img, image_id, undo_stack, redo_stack
    if not redo_stack:
        messagebox.showinfo("Redo", "Brak dalszych powtórzeń")
        return
    undo_stack.append(edge.copy())  # zapisujemy obecny stan do undo
    edge = redo_stack.pop()
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def update_resized_image(new_edge, new_width, new_height):
    global edge, img, image_id, img_width, img_height
    edge = new_edge
    img_width, img_height = new_width, new_height
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def openfn():
    return filedialog.askopenfilename(title='Open')

def open_img():
    global img, image_id, edge, img_width, img_height
    filename = openfn()
    if not filename:
        return
    pil_img = Image.open(filename)
    edge = pil_img
    img_width, img_height = pil_img.size
    img = ImageTk.PhotoImage(pil_img)
    canvas.delete("all")
    image_id = canvas.create_image(0, 0, anchor="nw", image=img)
    if show_hist.get():
        update_histogram()


start_x = 0
start_y = 0

def on_button_press(event):
    global start_x, start_y
    start_x = event.x
    start_y = event.y

def on_move_press(event):
    global start_x, start_y, image_id
    dx = event.x - start_x
    dy = event.y - start_y
    if image_id is not None:
        coords = canvas.coords(image_id)
        if not coords:
            return
        x, y = coords
        new_x = min(max(x + dx, canvas.winfo_width() - img_width), 0)
        new_y = min(max(y + dy, canvas.winfo_height() - img_height), 0)
        canvas.coords(image_id, new_x, new_y)
        start_x = event.x
        start_y = event.y

canvas.bind("<ButtonPress-1>", on_button_press)
canvas.bind("<B1-Motion>", on_move_press)

def savefile():
    global edge
    if edge is None:
        return
    filename = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if not filename:
        return
    try:
        edge.save(filename)
    except Exception as e:
        messagebox.showwarning("Zapis nieudany", f"Zapis PNG zamiast JPG.\n\n{e}")
        if not filename.lower().endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + ".png"
        try:
            edge.save(filename, format="PNG")
        except Exception as e2:
            messagebox.showerror("Zapis nieudany", f"Błąd: {e2}")

def apply_canny():
    global edge, img, image_id
    if edge is None:
        return
    push_undo(edge)
    arr = np.array(edge)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    canny = cv2.Canny(gray, 100, 200)
    edge = Image.fromarray(canny)
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def apply_sobel():
    global edge, img, image_id
    if edge is None:
        return
    push_undo(edge)
    arr = np.array(edge)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.magnitude(sobelx, sobely)
    edge = Image.fromarray(np.uint8(edges))
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def apply_laplacian():
    global edge, img, image_id
    if edge is None:
        return
    push_undo(edge)
    arr = np.array(edge)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge = Image.fromarray(np.uint8(np.absolute(laplacian)))
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()
        
def update_displayed_image(pil_img):
    global edge, img, image_id, img_width, img_height
    edge = pil_img
    img_width, img_height = pil_img.size
    img = ImageTk.PhotoImage(pil_img)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def add_watermark():
    global edge, img, image_id
    if edge is None:
        return
    push_undo(edge)
    # Okno dialogowe do wpisania tekstu
    dialog = tk.Toplevel(root)
    dialog.title("Dodaj znak wodny")
    dialog.geometry("300x150")

    tk.Label(dialog, text="Wpisz tekst znaku wodnego:").pack(pady=10)
    text_var = tk.StringVar()
    entry = tk.Entry(dialog, textvariable=text_var, font=("Segoe UI", 12))
    entry.pack(pady=5)
    entry.focus_set()

    def apply_text_watermark():
        text = text_var.get().strip()
        if not text:
            messagebox.showwarning("Błąd", "Tekst nie może być pusty.")
            return
        dialog.destroy()
        output = Water(edge, znak_rozmiar=(40, 40), wspolczynnik=0.3, tekst=text)
        from PIL import Image
        edge_pil = Image.fromarray(output)
        update_displayed_image(edge_pil)

    tk.Button(dialog, text="Zastosuj", command=apply_text_watermark).pack(pady=10)


def watermark_multiple_images():
    from PIL import Image

    dialog = tk.Toplevel(root)
    dialog.title("Dodaj znak wodny")
    dialog.geometry("300x150")

    tk.Label(dialog, text="Wpisz tekst znaku wodnego:").pack(pady=10)
    text_var = tk.StringVar()
    entry = tk.Entry(dialog, textvariable=text_var, font=("Segoe UI", 12))
    entry.pack(pady=5)
    entry.focus_set()


    def apply_multiple_WN():
        dialog.destroy()
        file_paths = filedialog.askopenfilenames(
            title="Select Images to Watermark",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )

        if not file_paths:
            return

        success_count = 0
        error_files = []

        for path in file_paths:
            try:
                text = text_var.get().strip()
                image = Image.open(path)
                watermarked = Water(image, znak_rozmiar=(40, 40), wspolczynnik=0.3, tekst = text)
                watermarked_img = Image.fromarray(watermarked)

                base, ext = os.path.splitext(path)
                new_path = base + "_WM" + ext
                watermarked_img.save(new_path)
                success_count += 1
            except Exception as e:
                error_files.append((path, str(e)))

        summary = f"{success_count} image(s) watermarked successfully."
        if error_files:
            summary += "\n\nSome files failed:\n"
            for path, err in error_files:
                summary += f"\n{os.path.basename(path)}: {err}"

        messagebox.showinfo("Watermarking Completed", summary)
    tk.Button(dialog, text="Zastosuj", command=apply_multiple_WN).pack(pady=10)

# --- PROGOWANIE ---
def th_const(thresh_val=127, max_val=255):
    global edge, img, image_id
    if edge is None: return
    push_undo(edge)
    arr = np.array(edge)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    _, thresh = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
    edge = Image.fromarray(thresh)
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def th_adapt(max_val=255):
    global edge, img, image_id
    if edge is None: return
    push_undo(edge)
    arr = np.array(edge)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    adapt = cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    edge = Image.fromarray(adapt)
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def th_otsu():
    global edge, img, image_id
    if edge is None: return
    push_undo(edge)
    arr = np.array(edge)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = Image.fromarray(otsu)
    img = ImageTk.PhotoImage(edge)
    canvas.itemconfig(image_id, image=img)
    if show_hist.get():
        update_histogram()

def show_histogram():
    global edge
    if edge is None:
        messagebox.showinfo("Brak obrazu", "Najpierw otwórz obraz.")
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        edge.save(temp.name)
        viewer = HistogramViewer(temp.name)
        viewer.show_histograms()
    os.remove(temp.name)

def update_histogram():
    global edge, hist_label
    if edge is None or not show_hist.get():
        hist_label.config(image='')  # ukryj, jeśli wyłączony
        return

    img_arr = np.array(edge)
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)

    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        channels = ('r', 'g', 'b')
        for i, col in enumerate(channels):
            hist = cv2.calcHist([img_arr], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
    else:
        hist = cv2.calcHist([img_arr], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')

    ax.set_xlim([0, 256])
    ax.set_title("Histogram")
    ax.set_xlabel("Wartość piksela")
    ax.set_ylabel("Liczba pikseli")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img_hist = Image.open(buf)
    img_hist_tk = ImageTk.PhotoImage(img_hist)
    hist_label.config(image=img_hist_tk)
    hist_label.image = img_hist_tk  # zapobiega znikaniu

    # --- Siec neuronowa---

def detect_people():
    global edge, img, image_id  # Przenieś deklaracje globalne na początek funkcji
    
    if edge is None:
        messagebox.showinfo("Brak obrazu", "Najpierw otwórz obraz.")
        return
    push_undo(edge)

    # Okno dialogowe do wyboru klasy
    class_dialog = tk.Toplevel(root)
    class_dialog.title("Wybierz klasę do wykrycia")
    class_dialog.geometry("350x200")
    
    # Ładowanie dostępnych klas
    try:
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        messagebox.showerror("Błąd", "Nie znaleziono pliku coco.names")
        class_dialog.destroy()
        return
    
    tk.Label(class_dialog, text="Wyszukaj klasę:").pack(pady=5)
    
    search_var = tk.StringVar()
    search_entry = tk.Entry(class_dialog, textvariable=search_var)
    search_entry.pack()
    
    tk.Label(class_dialog, text="Wybierz z listy:").pack()
    
    combo = ttk.Combobox(class_dialog, values=classes, width=30)
    combo.pack(pady=5)
    combo.set("person")  # Domyślnie wybrana osoba
    
    def on_search(*args):
        search_term = search_var.get().lower()
        filtered = [c for c in classes if search_term in c.lower()]
        combo['values'] = filtered
        if filtered:
            combo.set(filtered[0])
    
    search_var.trace("w", on_search)
    
    def run_detection_with_class():
        global edge, img  # Deklaracje globalne dla funkcji zagnieżdżonej
        
        selected_class = combo.get()
        if not selected_class:
            messagebox.showwarning("Błąd", "Wybierz klasę do wykrycia")
            return
            
        # Zapisz obecny obraz do pliku tymczasowego
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_input:
            edge.save(tmp_input.name)
            tmp_path = tmp_input.name

        try:
            # Wykonaj detekcję
            cv_img = cv2.imread(tmp_path)
            if cv_img is None:
                raise ValueError("Nie udało się wczytać obrazu")
            
            detected_img = run_detection(cv_img, target_class=selected_class)
            detected_pil = Image.fromarray(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
            
            # Aktualizuj wyświetlany obraz
            edge = detected_pil
            img = ImageTk.PhotoImage(edge)
            canvas.itemconfig(image_id, image=img)
            
            if show_hist.get():
                update_histogram()
                
            class_dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    ttk.Button(class_dialog, 
              text="Wykryj", 
              command=run_detection_with_class).pack(pady=10)

# --- MENU ---
menubar = Menu(root)
root.config(menu=menubar)

file_menu = Menu(menubar, tearoff=0)
file_menu.add_command(label='Open', command=open_img)
file_menu.add_command(label='Save', command=savefile)
file_menu.add_command(label='Exit', command=root.destroy)
menubar.add_cascade(label="File", menu=file_menu)

edit_menu = Menu(menubar, tearoff=0)
edit_menu.add_command(label="Resize Image", command=lambda: ask_resize(root, edge, img_width, img_height, update_resized_image))
edit_menu.add_command(label="Batch Resize Images", command=batch_resize_images)
edit_menu.add_separator()
edit_menu.add_command(label="Undo", command=undo)
edit_menu.add_command(label="Redo", command=redo)
menubar.add_cascade(label="Edit", menu=edit_menu)

edge_menu = Menu(menubar, tearoff=0)
edge_menu.add_command(label='Canny', command=apply_canny)
edge_menu.add_command(label='Sobel', command=apply_sobel)
edge_menu.add_command(label='Laplacian', command=apply_laplacian)
menubar.add_cascade(label="Edge Detection", menu=edge_menu)

thresh_menu = Menu(menubar, tearoff=0)
thresh_menu.add_command(label='Threshold Const', command=lambda: th_const(127))
thresh_menu.add_command(label='Threshold Adapt', command=th_adapt)
thresh_menu.add_command(label='Threshold Otsu', command=th_otsu)
menubar.add_cascade(label="Thresholding", menu=thresh_menu)

watermark_menu = Menu(menubar, tearoff=0)
watermark_menu.add_command(label='Add Watermark', command=add_watermark)
watermark_menu.add_command(label='Watermark Multiple', command=watermark_multiple_images)
menubar.add_cascade(label="Watermark", menu=watermark_menu)

hist_menu = Menu(menubar, tearoff=0)
hist_menu.add_checkbutton(label="Show Histogram", variable=show_hist, command=update_histogram)
menubar.add_cascade(label="Histogram", menu=hist_menu)

detect_menu = Menu(menubar, tearoff=0)
detect_menu.add_command(label="Neural Detection", command=detect_people)
menubar.add_cascade(label="Neural Network", menu=detect_menu)
root.bind_all("<Control-z>", lambda event: undo())
root.bind_all("<Control-y>", lambda event: redo())
root.mainloop()
