# Developed by Abhisekh Padhy

# Full-featured GUI Annotator with BERT Similarity, Dark Mode, and UX Enhancements
# Features:
# - Username support
# - BERT-based semantic similarity score
# - Resume progress
# - Progress bar
# - Submit button aligned right
# - Scrollbar support for long lists
# - Ctrl+Enter shortcut for submit
# - Exit button
# - Dark mode theme
# - Color-coded answer labels
# - Tooltips and preview before submit (optional)
# - Merge multiple annotators' outputs into one CSV

import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from tkinter import ttk
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import glob

DATA_FILE = "gsm8k_wrong_answers_with_missing_prerequisites.csv"

# Ask username at the start
root = tk.Tk()
root.withdraw()
username = simpledialog.askstring("User Login", "Enter your name (no spaces):")
SAVE_FILE = f"annotations_{username}.csv"
root.deiconify()

# Apply dark mode theme
bg_color = "#2e2e2e"
fg_color = "#ffffff"
btn_color = "#444444"
entry_bg = "#3a3a3a"

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background=bg_color, foreground=fg_color)
style.configure("TButton", background=btn_color, foreground=fg_color)
style.configure("TCheckbutton", background=bg_color, foreground=fg_color)

root.configure(bg=bg_color)

# Load dataset
try:
    df = pd.read_csv(DATA_FILE).dropna(subset=["missing_prerequisite", "all_prerequisites"]).reset_index(drop=True)
except Exception as e:
    messagebox.showerror("File Error", f"Could not read CSV: {e}")
    root.quit()

# Load progress
if os.path.exists(SAVE_FILE):
    done = pd.read_csv(SAVE_FILE)
    done_ids = set(done["question_id"])
else:
    done = pd.DataFrame()
    done_ids = set()

# Load BERT model once
try:
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    messagebox.showerror("Model Error", f"Could not load BERT model: {e}")
    root.quit()

# GUI setup
root.title("Missing Prerequisite Annotator")
frame = tk.Frame(root, bg=bg_color)
frame.pack(padx=10, pady=10)

progress_var = tk.StringVar()
progress_label = tk.Label(frame, textvariable=progress_var, fg="violet", bg=bg_color)
progress_label.pack()

question_label = tk.Label(frame, text="", wraplength=500, justify="left", fg=fg_color, bg=bg_color)
question_label.pack()

correct_label = tk.Label(frame, text="", fg="lightgreen", bg=bg_color)
correct_label.pack()

wrong_label = tk.Label(frame, text="", fg="salmon", bg=bg_color)
wrong_label.pack()

gemma_label = tk.Label(frame, text="", fg="skyblue", bg=bg_color)
gemma_label.pack()

content_frame = tk.Frame(frame, bg=bg_color)
content_frame.pack(fill="both", expand=True)

# Scrollable checkbox frame
checkbox_canvas = tk.Canvas(content_frame, width=400, height=300, bg=bg_color, highlightthickness=0)
scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=checkbox_canvas.yview)
checkbox_scroll_frame = tk.Frame(checkbox_canvas, bg=bg_color)
checkbox_scroll_frame.bind(
    "<Configure>", lambda e: checkbox_canvas.configure(scrollregion=checkbox_canvas.bbox("all"))
)

checkbox_canvas.create_window((0, 0), window=checkbox_scroll_frame, anchor="nw")
checkbox_canvas.configure(yscrollcommand=scrollbar.set)

checkbox_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="left", fill="y")

button_frame = tk.Frame(content_frame, bg=bg_color)
button_frame.pack(side="right", padx=10, pady=10)

submit_button = tk.Button(button_frame, text="Submit", bg="lightblue")
submit_button.pack(pady=5)

merge_button = tk.Button(button_frame, text="Merge CSVs", bg="gold", command=lambda: merge_csvs())
merge_button.pack(pady=5)

exit_button = tk.Button(button_frame, text="Exit", bg="tomato", command=root.quit)
exit_button.pack(pady=5)

prereq_vars = []

# BERT-based cosine similarity function
def get_similarity(text1, text2):
    try:
        emb1 = bert_model.encode(text1, convert_to_tensor=True)
        emb2 = bert_model.encode(text2, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        return round(score, 3)
    except Exception as e:
        print(f"BERT similarity error: {e}")
        return 0.0

def update_progress():
    completed = len(done_ids)
    total = len(df)
    progress_var.set(f"Progress: {completed}/{total} annotated")

def next_index():
    for idx, row in df.iterrows():
        if row["question_id"] not in done_ids:
            return idx
    return None

def load_question(index):
    global prereq_vars
    for widget in checkbox_scroll_frame.winfo_children():
        widget.destroy()
    prereq_vars.clear()

    row = df.iloc[index]
    question_label.config(text="Q: " + row["question"])
    correct_label.config(text="✔️ Correct Answer: " + str(row["correct_answer"]))
    wrong_label.config(text="❌ Wrong Answer: " + str(row["wrong_answer"]))
    gemma_label.config(text="\U0001f916 Gemma3: " + str(row["missing_prerequisite"]))

    all_prereqs = [p.strip() for p in row["all_prerequisites"].split(',') if p.strip()]
    for prereq in all_prereqs:
        var = tk.BooleanVar()
        cb = tk.Checkbutton(checkbox_scroll_frame, text=prereq, variable=var, anchor="w", justify="left",
                            bg=bg_color, fg=fg_color, selectcolor=bg_color)
        cb.pack(anchor="w")
        prereq_vars.append((prereq, var))

    submit_button.config(command=lambda: save_response(index, row))
    update_progress()

def save_response(index, row):
    selected = [pr for pr, var in prereq_vars if var.get()]
    if not selected:
        if not messagebox.askyesno("Confirm", "No prerequisite selected. Submit anyway?"):
            return

    human_text = ", ".join(selected)
    model_text = row["missing_prerequisite"]
    sim_score = get_similarity(model_text, human_text)

    entry = {
        "question_id": row["question_id"],
        "human_selected_prerequisite": human_text,
        "gemma_missing_prerequisite": model_text,
        "similarity_score": f"{sim_score:.3f}",
        "annotator": username
    }

    new_df = pd.DataFrame([entry])
    new_df.to_csv(SAVE_FILE, mode='a', header=not os.path.exists(SAVE_FILE), index=False)
    done_ids.add(row["question_id"])

    next_idx = next_index()
    if next_idx is not None:
        load_question(next_idx)
    else:
        messagebox.showinfo("Done", "All questions have been annotated.")
        root.quit()

# Merge annotator CSVs
def merge_csvs():
    files = glob.glob("annotations_*.csv")
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save Merged CSV As")
    if save_path:
        all_df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Merged {len(files)} files into: {save_path}")

# Keyboard shortcut (Ctrl+Enter)
def on_ctrl_enter(event):
    submit_button.invoke()

root.bind('<Control-Return>', on_ctrl_enter)

start_idx = next_index()
if start_idx is not None:
    load_question(start_idx)
else:
    question_label.config(text="✅ All questions are already annotated!")

root.mainloop()
