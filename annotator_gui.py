#Developed by Abhisekh Padhy and Arka Mukherjee

import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import pandas as pd
import os
import glob
import json
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

DATA_FILE = "gsm8k_wrong_answers_with_missing_prerequisites.csv"
CONFIG_FILE = "user_config.json"
OUTPUT_DIR = "annotations_output"  # Shared directory for all CSV files

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class SecurityManager:
    """Handles password encryption and secure storage"""
    
    def __init__(self):
        self.salt = b'annotator_salt_2024'  # Fixed salt for consistency across machines
        
    def _get_key(self, master_password="AnnotatorSecure2024"):
        """Generate encryption key from master password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        return key
    
    def encrypt_password(self, password):
        """Encrypt a password"""
        if not password:
            return None
        f = Fernet(self._get_key())
        encrypted = f.encrypt(password.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_password(self, encrypted_password):
        """Decrypt a password"""
        if not encrypted_password:
            return None
        try:
            f = Fernet(self._get_key())
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_password.encode())
            decrypted = f.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            print(f"Password decryption failed: {e}")
            return None
    
    def hash_password(self, password):
        """Create a hash of password for verification (one-way)"""
        if not password:
            return None
        return hashlib.sha256(f"{password}{self.salt.decode('latin-1')}".encode()).hexdigest()

class AnnotatorApp:
    def __init__(self):
        self.username = None
        self.user_role = None
        self.password = None
        self.question_bank = None
        self.df = None
        self.done_ids = set()
        self.prereq_vars = []
        
        # Initialize security manager
        self.security = SecurityManager()
        
        # Question banks (50 questions each)
        self.question_banks = {
            "Bank 1 (1-50)": (0, 49),
            "Bank 2 (51-100)": (50, 99),
            "Bank 3 (101-150)": (100, 149),
            "Bank 4 (151-200)": (150, 199)
        }
        
        # User configurations
        self.user_configs = self.load_user_configs()
        
        # Clean color scheme
        self.bg_color = "#f5f5f5"
        self.text_color = "#333333"
        self.header_color = "#2c3e50"
        self.correct_bg = "#d4edda"
        self.wrong_bg = "#f8d7da"
        self.button_bg = "#007bff"
        self.button_text = "#ffffff"
        self.card_bg = "#ffffff"
        
        # Create main window
        self.root = tk.Tk()
        self.root.configure(bg=self.bg_color)
        self.root.geometry('1100x700')
        self.root.minsize(1000, 600)
        self.root.title("Missing Prerequisite Annotator")
        self.root.eval('tk::PlaceWindow . center')
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        
        self.show_login()
        
    def load_user_configs(self):
        """Load user configurations from encrypted file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    encrypted_data = json.load(f)
                
                # Decrypt passwords
                decrypted_data = {}
                for username, config in encrypted_data.items():
                    decrypted_config = config.copy()
                    if config.get("password_encrypted"):
                        decrypted_config["password"] = self.security.decrypt_password(config["password_encrypted"])
                    elif config.get("password_hash"):  # Admin password stored as hash
                        decrypted_config["password_hash"] = config["password_hash"]
                        decrypted_config["password"] = None  # Admin password verified differently
                    decrypted_data[username] = decrypted_config
                
                return decrypted_data
                
            except Exception as e:
                print(f"Could not load config: {e}")
        
        # Default configuration with encrypted admin password
        admin_password_hash = self.security.hash_password("configurator25")
        return {
            "config": {
                "role": "administrator",
                "password_hash": admin_password_hash,
                "password": None,
                "question_bank": None
            },
            "Arka": {
                "role": "annotator",
                "password": None,
                "question_bank": None
            },
            "Abhisekh": {
                "role": "annotator", 
                "password": None,
                "question_bank": None
            },
            "Nilam": {
                "role": "annotator",
                "password": None,
                "question_bank": None
            },
            "Sriram": {
                "role": "annotator",
                "password": None,
                "question_bank": None
            }
        }
    
    def save_user_configs(self):
        """Save user configurations with encrypted passwords"""
        try:
            encrypted_data = {}
            for username, config in self.user_configs.items():
                encrypted_config = config.copy()
                
                if username == "config":
                    # Admin password stored as hash only
                    if "password_hash" not in encrypted_config and config.get("password"):
                        encrypted_config["password_hash"] = self.security.hash_password(config["password"])
                    encrypted_config.pop("password", None)  # Remove plain password
                else:
                    # Annotator passwords encrypted
                    if config.get("password"):
                        encrypted_config["password_encrypted"] = self.security.encrypt_password(config["password"])
                        encrypted_config.pop("password", None)  # Remove plain password
                
                encrypted_data[username] = encrypted_config
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(encrypted_data, f, indent=2)
                
        except Exception as e:
            print(f"Could not save user config: {e}")
        
    def show_login(self):
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Create login frame
        login_frame = tk.Frame(self.root, bg=self.bg_color)
        login_frame.pack(expand=True, fill='both')
        
        # Center container
        center_frame = tk.Frame(login_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Title
        title_label = tk.Label(center_frame, text="Missing Prerequisite Annotator", 
                              font=('Arial', 18, 'bold'), bg=self.bg_color, fg=self.header_color)
        title_label.pack(pady=(0, 30))
        
        # Username label
        username_label = tk.Label(center_frame, text="Select Username:", 
                                 font=('Arial', 12), bg=self.bg_color, fg=self.header_color)
        username_label.pack(anchor='w', pady=(0, 5))
        
        # Username dropdown
        self.username_var = tk.StringVar()
        username_combo = ttk.Combobox(center_frame, textvariable=self.username_var, 
                                     font=('Arial', 12), width=22, state="readonly")
        username_combo['values'] = list(self.user_configs.keys())
        username_combo.pack(pady=(0, 15))
        
        # Password label
        password_label = tk.Label(center_frame, text="Password:", 
                                 font=('Arial', 12), bg=self.bg_color, fg=self.header_color)
        password_label.pack(anchor='w', pady=(0, 5))
        
        # Password entry
        self.password_entry = tk.Entry(center_frame, font=('Arial', 12), width=25, show="*")
        self.password_entry.pack(pady=(0, 20))
        
        # Buttons frame
        button_frame = tk.Frame(center_frame, bg=self.bg_color)
        button_frame.pack()
        
        # Login button
        login_button = tk.Button(button_frame, text="Login", 
                               font=('Arial', 11, 'bold'), bg=self.button_bg, fg=self.button_text,
                               relief='flat', padx=20, pady=8, cursor='hand2',
                               command=self.validate_login)
        login_button.pack(side='left', padx=(0, 10))
        
        # Exit button
        exit_button = tk.Button(button_frame, text="Exit", 
                              font=('Arial', 11), bg="#dc3545", fg=self.button_text,
                              relief='flat', padx=20, pady=8, cursor='hand2',
                              command=self.exit_app)
        exit_button.pack(side='left')
        
        # Bind Enter key to login
        self.root.bind('<Return>', lambda event: self.validate_login())
        
        # Focus on username dropdown
        username_combo.focus()
        
    def validate_login(self):
        username = self.username_var.get().strip()
        password = self.password_entry.get().strip()
        
        if not username:
            messagebox.showerror("Error", "Please select a username.")
            return
            
        if username not in self.user_configs:
            messagebox.showerror("Error", "Invalid username.")
            return
            
        user_config = self.user_configs[username]
        
        if username == "config":
            # Admin password verification using hash
            password_hash = self.security.hash_password(password)
            if user_config.get("password_hash") != password_hash:
                messagebox.showerror("Error", "Incorrect password.")
                return
        else:
            # Annotator password handling
            if user_config["password"] is None:
                # First time login - set password
                if not password:
                    messagebox.showerror("Error", "Please set a password for first-time login.")
                    return
                if len(password) < 4:
                    messagebox.showerror("Error", "Password must be at least 4 characters long.")
                    return
                
                # Set password for first time
                user_config["password"] = password
                self.save_user_configs()
                messagebox.showinfo("Password Set", "Password has been set successfully!")
                
            else:
                # Validate existing password
                if user_config["password"] != password:
                    messagebox.showerror("Error", "Incorrect password.")
                    return
        
        self.username = username
        self.user_role = user_config["role"]
        self.password = password
        
        if self.user_role == "administrator":
            self.load_admin_interface()
        else:
            self.setup_annotator()
            
    def setup_annotator(self):
        user_config = self.user_configs[self.username]
        
        # Check if question bank is already assigned
        if user_config["question_bank"] is not None:
            self.question_bank = user_config["question_bank"]
            self.load_data()
        else:
            self.show_bank_selection()
            
    def show_bank_selection(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Create bank selection frame
        bank_frame = tk.Frame(self.root, bg=self.bg_color)
        bank_frame.pack(expand=True, fill='both')
        
        # Center container
        center_frame = tk.Frame(bank_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Title
        title_label = tk.Label(center_frame, text=f"Welcome, {self.username}!", 
                              font=('Arial', 18, 'bold'), bg=self.bg_color, fg=self.header_color)
        title_label.pack(pady=(0, 20))
        
        subtitle_label = tk.Label(center_frame, text="Choose your question bank (50 questions each):", 
                                 font=('Arial', 12), bg=self.bg_color, fg=self.header_color)
        subtitle_label.pack(pady=(0, 20))
        
        # Bank selection buttons
        self.selected_bank = tk.StringVar()
        
        for bank_name in self.question_banks.keys():
            # Check if bank is already taken
            taken_by = self.get_bank_owner(bank_name)
            if taken_by and taken_by != self.username:
                btn_text = f"{bank_name} (Taken by {taken_by})"
                btn_state = "disabled"
                btn_color = "#6c757d"
            else:
                btn_text = bank_name
                btn_state = "normal"
                btn_color = self.button_bg
                
            btn = tk.Radiobutton(center_frame, text=btn_text, variable=self.selected_bank, 
                               value=bank_name, font=('Arial', 11), bg=self.bg_color, 
                               fg=self.text_color, state=btn_state, cursor='hand2' if btn_state == "normal" else "")
            btn.pack(anchor='w', pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(center_frame, bg=self.bg_color)
        button_frame.pack(pady=(30, 0))
        
        # Confirm button
        confirm_button = tk.Button(button_frame, text="Confirm Selection", 
                                 font=('Arial', 11, 'bold'), bg=self.button_bg, fg=self.button_text,
                                 relief='flat', padx=20, pady=8, cursor='hand2',
                                 command=self.confirm_bank_selection)
        confirm_button.pack(side='left', padx=(0, 10))
        
        # Logout button
        logout_button = tk.Button(button_frame, text="Logout", 
                                font=('Arial', 11), bg="#6c757d", fg=self.button_text,
                                relief='flat', padx=20, pady=8, cursor='hand2',
                                command=self.show_login)
        logout_button.pack(side='left')
        
    def get_bank_owner(self, bank_name):
        """Check if a bank is already assigned to someone"""
        for username, config in self.user_configs.items():
            if config.get("question_bank") == bank_name:
                return username
        return None
        
    def confirm_bank_selection(self):
        selected_bank = self.selected_bank.get()
        if not selected_bank:
            messagebox.showerror("Error", "Please select a question bank.")
            return
            
        # Check if bank is still available
        owner = self.get_bank_owner(selected_bank)
        if owner and owner != self.username:
            messagebox.showerror("Error", f"Bank {selected_bank} is already taken by {owner}.")
            return
            
        # Assign bank to user
        self.user_configs[self.username]["question_bank"] = selected_bank
        self.save_user_configs()
        self.question_bank = selected_bank
        
        messagebox.showinfo("Bank Assigned", f"You have been assigned to {selected_bank}.")
        self.load_data()
        
    def load_admin_interface(self):
        """Load administrator interface with merge functionality"""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Admin frame
        admin_frame = tk.Frame(self.root, bg=self.bg_color)
        admin_frame.pack(expand=True, fill='both', padx=40, pady=40)
        
        # Title
        title_label = tk.Label(admin_frame, text="Administrator Panel", 
                              font=('Arial', 18, 'bold'), bg=self.bg_color, fg=self.header_color)
        title_label.pack(pady=(0, 30))
        
        # Status section
        status_frame = tk.Frame(admin_frame, bg=self.card_bg, relief='solid', bd=1)
        status_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(status_frame, text="Annotation Status", font=('Arial', 14, 'bold'), 
                bg=self.card_bg, fg=self.header_color).pack(pady=(15, 10))
        
        # Show status for each annotator
        for username, config in self.user_configs.items():
            if config["role"] == "annotator":
                bank = config.get("question_bank", "Not assigned")
                csv_file = os.path.join(OUTPUT_DIR, f"annotations_{username}.csv")
                
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        count = len(df)
                        status = f"✓ {count} annotations completed"
                        color = "#28a745"
                    except:
                        status = "⚠ CSV file corrupted"
                        color = "#ffc107"
                else:
                    status = "○ Not started"
                    color = "#6c757d"
                
                status_text = f"{username}: {bank} - {status}"
                tk.Label(status_frame, text=status_text, font=('Arial', 10), 
                        bg=self.card_bg, fg=color).pack(anchor='w', padx=15, pady=2)
        
        tk.Frame(status_frame, height=10, bg=self.card_bg).pack()  # Spacer
        
        # Buttons section
        button_frame = tk.Frame(admin_frame, bg=self.bg_color)
        button_frame.pack(pady=(20, 0))
        
        # Merge CSVs button
        merge_button = tk.Button(button_frame, text="Merge All CSV Files", 
                               font=('Arial', 12, 'bold'), bg="#28a745", fg="white",
                               relief='flat', padx=30, pady=10, cursor='hand2',
                               command=self.merge_csvs)
        merge_button.pack(pady=(0, 10))
        
        # Refresh status button
        refresh_button = tk.Button(button_frame, text="Refresh Status", 
                                 font=('Arial', 11), bg=self.button_bg, fg=self.button_text,
                                 relief='flat', padx=20, pady=8, cursor='hand2',
                                 command=self.load_admin_interface)
        refresh_button.pack(pady=(0, 10))
        
        # Logout button
        logout_button = tk.Button(button_frame, text="Logout", 
                                font=('Arial', 11), bg="#dc3545", fg=self.button_text,
                                relief='flat', padx=20, pady=8, cursor='hand2',
                                command=self.show_login)
        logout_button.pack()
        
    def load_data(self):
        try:
            full_df = pd.read_csv(DATA_FILE).dropna(subset=["missing_prerequisite", "all_prerequisites"]).reset_index(drop=True)
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read CSV: {e}")
            return
            
        # Filter to assigned question bank
        if self.question_bank and self.question_bank in self.question_banks:
            start_idx, end_idx = self.question_banks[self.question_bank]
            self.df = full_df.iloc[start_idx:end_idx+1].reset_index(drop=True)
            print(f"Loaded {len(self.df)} questions from {self.question_bank}")
        else:
            messagebox.showerror("Error", "No question bank assigned!")
            return
            
        # Load progress with error handling
        save_file = os.path.join(OUTPUT_DIR, f"annotations_{self.username}.csv")
        if os.path.exists(save_file):
            try:
                done = pd.read_csv(save_file)
                self.done_ids = set(done["question_id"])
                print(f"Loaded {len(self.done_ids)} completed annotations from {save_file}")
            except Exception as e:
                print(f"Warning: Could not read existing progress file {save_file}: {e}")
                
                response = messagebox.askyesno(
                    "Progress File Error", 
                    f"The existing progress file appears to be corrupted.\n\n"
                    f"Error: {str(e)}\n\n"
                    "Click 'Yes' to start fresh (old file will be backed up)\n"
                    "Click 'No' to exit and fix the file manually"
                )
                
                if response:
                    backup_name = f"{save_file}.backup"
                    try:
                        os.rename(save_file, backup_name)
                        print(f"Backed up corrupted file to {backup_name}")
                        messagebox.showinfo("Backup Created", f"Old file backed up as:\n{backup_name}")
                    except Exception as backup_error:
                        print(f"Could not backup file: {backup_error}")
                    
                    self.done_ids = set()
                else:
                    return
        else:
            self.done_ids = set()
            
        self.show_annotator()
        
    def show_annotator(self):
        # Clear login widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        self.root.unbind('<Return>')  # Remove login Enter binding
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.bg_color)
        header_frame.pack(fill='x', padx=20, pady=(15, 10))

        title_label = tk.Label(header_frame, text=f"Annotator: {self.username} | {self.question_bank}", 
                              font=('Arial', 16, 'bold'), bg=self.bg_color, fg=self.header_color)
        title_label.pack()

        self.progress_var = tk.StringVar()
        progress_label = tk.Label(header_frame, textvariable=self.progress_var, 
                                 font=('Arial', 10), bg=self.bg_color, fg=self.header_color)
        progress_label.pack(pady=(5, 0))

        # Main content frame
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))

        # Left side - Question and Answers
        left_frame = tk.Frame(content_frame, bg=self.bg_color)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Question section
        question_frame = tk.Frame(left_frame, bg=self.card_bg, relief='solid', bd=1)
        question_frame.pack(fill='x', pady=(0, 10))

        tk.Label(question_frame, text="Question", font=('Arial', 12, 'bold'), 
                 bg=self.card_bg, fg=self.header_color).pack(anchor='w', padx=15, pady=(12, 5))

        self.question_label = tk.Label(question_frame, text="", wraplength=500, justify="left", 
                                 font=('Arial', 10), bg=self.card_bg, fg=self.text_color)
        self.question_label.pack(anchor='w', padx=15, pady=(0, 12))

        # Answers section - side by side with dynamic height
        answers_container = tk.Frame(left_frame, bg=self.bg_color)
        answers_container.pack(fill='both', expand=True, pady=(0, 10))

        # Correct answer box with scrollbar
        correct_frame = tk.Frame(answers_container, bg=self.correct_bg, relief='solid', bd=1)
        correct_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        tk.Label(correct_frame, text="✓ Correct Answer", font=('Arial', 10, 'bold'), 
                 bg=self.correct_bg, fg="#155724").pack(anchor='w', padx=12, pady=(8, 3))

        correct_text_frame = tk.Frame(correct_frame, bg=self.correct_bg)
        correct_text_frame.pack(fill='both', expand=True, padx=12, pady=(0, 8))

        self.correct_canvas = tk.Canvas(correct_text_frame, bg=self.correct_bg, highlightthickness=0)
        correct_scrollbar = ttk.Scrollbar(correct_text_frame, orient="vertical", command=self.correct_canvas.yview)
        correct_scroll_frame = tk.Frame(self.correct_canvas, bg=self.correct_bg)

        correct_scroll_frame.bind(
            "<Configure>", lambda e: self.correct_canvas.configure(scrollregion=self.correct_canvas.bbox("all"))
        )

        self.correct_canvas.create_window((0, 0), window=correct_scroll_frame, anchor="nw")
        self.correct_canvas.configure(yscrollcommand=correct_scrollbar.set)

        self.correct_label = tk.Label(correct_scroll_frame, text="", wraplength=220, justify="left",
                                font=('Arial', 9), bg=self.correct_bg, fg="#155724")
        self.correct_label.pack(anchor='w')

        self.correct_canvas.pack(side="left", fill="both", expand=True)
        correct_scrollbar.pack(side="right", fill="y")

        # Wrong answer box with scrollbar
        wrong_frame = tk.Frame(answers_container, bg=self.wrong_bg, relief='solid', bd=1)
        wrong_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        tk.Label(wrong_frame, text="✗ Wrong Answer", font=('Arial', 10, 'bold'), 
                 bg=self.wrong_bg, fg="#721c24").pack(anchor='w', padx=12, pady=(8, 3))

        wrong_text_frame = tk.Frame(wrong_frame, bg=self.wrong_bg)
        wrong_text_frame.pack(fill='both', expand=True, padx=12, pady=(0, 8))

        self.wrong_canvas = tk.Canvas(wrong_text_frame, bg=self.wrong_bg, highlightthickness=0)
        wrong_scrollbar = ttk.Scrollbar(wrong_text_frame, orient="vertical", command=self.wrong_canvas.yview)
        wrong_scroll_frame = tk.Frame(self.wrong_canvas, bg=self.wrong_bg)

        wrong_scroll_frame.bind(
            "<Configure>", lambda e: self.wrong_canvas.configure(scrollregion=self.wrong_canvas.bbox("all"))
        )

        self.wrong_canvas.create_window((0, 0), window=wrong_scroll_frame, anchor="nw")
        self.wrong_canvas.configure(yscrollcommand=wrong_scrollbar.set)

        self.wrong_label = tk.Label(wrong_scroll_frame, text="", wraplength=220, justify="left",
                              font=('Arial', 9), bg=self.wrong_bg, fg="#721c24")
        self.wrong_label.pack(anchor='w')

        self.wrong_canvas.pack(side="left", fill="both", expand=True)
        wrong_scrollbar.pack(side="right", fill="y")

        # Right side - Prerequisites
        right_frame = tk.Frame(content_frame, bg=self.bg_color, width=400)
        right_frame.pack(side='right', fill='both', expand=False)
        right_frame.pack_propagate(False)

        prereq_frame = tk.Frame(right_frame, bg=self.card_bg, relief='solid', bd=1)
        prereq_frame.pack(fill='both', expand=True)

        tk.Label(prereq_frame, text="Select Missing Prerequisites", font=('Arial', 12, 'bold'), 
                 bg=self.card_bg, fg=self.header_color).pack(anchor='w', padx=15, pady=(12, 8))

        prereq_container = tk.Frame(prereq_frame, bg=self.card_bg)
        prereq_container.pack(fill='both', expand=True, padx=15, pady=(0, 12))

        self.prereq_canvas = tk.Canvas(prereq_container, bg=self.card_bg, highlightthickness=0)
        prereq_scrollbar = ttk.Scrollbar(prereq_container, orient="vertical", command=self.prereq_canvas.yview)
        self.prereq_scroll_frame = tk.Frame(self.prereq_canvas, bg=self.card_bg)

        self.prereq_scroll_frame.bind(
            "<Configure>", lambda e: self.prereq_canvas.configure(scrollregion=self.prereq_canvas.bbox("all"))
        )

        self.prereq_canvas.create_window((0, 0), window=self.prereq_scroll_frame, anchor="nw")
        self.prereq_canvas.configure(yscrollcommand=prereq_scrollbar.set)

        self.prereq_canvas.pack(side="left", fill="both", expand=True)
        prereq_scrollbar.pack(side="right", fill="y")

        # Button frame
        button_frame = tk.Frame(self.root, bg=self.bg_color)
        button_frame.pack(fill='x', padx=20, pady=(0, 15))

        self.submit_button = tk.Button(button_frame, text="Submit Annotation", 
                                 font=('Arial', 11, 'bold'), bg=self.button_bg, fg=self.button_text,
                                 relief='flat', padx=25, pady=8, cursor='hand2')
        self.submit_button.pack(side='left')

        logout_button = tk.Button(button_frame, text="Logout", 
                                font=('Arial', 10), bg="#dc3545", fg=self.button_text,
                                relief='flat', padx=20, pady=6, cursor='hand2',
                                command=self.show_login)
        logout_button.pack(side='right')

        # Setup mouse wheel scrolling
        self.correct_canvas.bind("<MouseWheel>", lambda e: self.correct_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.wrong_canvas.bind("<MouseWheel>", lambda e: self.wrong_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.prereq_canvas.bind("<MouseWheel>", lambda e: self.prereq_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Setup keyboard shortcuts
        self.root.bind('<Control-Return>', lambda event: self.submit_button.invoke())
        self.root.bind('<Escape>', lambda event: self.show_login() if messagebox.askyesno("Logout", "Are you sure you want to logout?") else None)

        self.load_first_question()
        
    def calculate_agreement_metrics(self, human_selected, model_text, all_prereqs):
        """Calculate agreement metrics between human selections and model predictions"""
        human_set = set([p.strip() for p in human_selected.split(',') if p.strip()])
        
        model_set = set()
        model_text_lower = model_text.lower()
        
        for prereq in all_prereqs:
            prereq_lower = prereq.lower()
            if prereq_lower in model_text_lower:
                model_set.add(prereq)
        
        intersection = human_set.intersection(model_set)
        union = human_set.union(model_set)
        
        jaccard = len(intersection) / len(union) if union else 1.0
        exact_match = 1.0 if human_set == model_set else 0.0
        precision = len(intersection) / len(model_set) if model_set else 0.0
        recall = len(intersection) / len(human_set) if human_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'jaccard': round(jaccard, 3),
            'exact_match': exact_match,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3)
        }

    def update_progress(self):
        completed = len(self.done_ids)
        total = len(self.df)
        percentage = (completed / total * 100) if total > 0 else 0
        self.progress_var.set(f"Progress: {completed}/{total} annotated ({percentage:.1f}%)")

    def next_index(self):
        for idx, row in self.df.iterrows():
            if row["question_id"] not in self.done_ids:
                return idx
        return None

    def load_question(self, index):
        for widget in self.prereq_scroll_frame.winfo_children():
            widget.destroy()
        self.prereq_vars.clear()

        row = self.df.iloc[index]
        
        question_text = str(row["question"]).strip().strip('"').strip("'")
        self.question_label.config(text=question_text)
        
        correct_text = str(row["correct_answer"]).strip().strip('"').strip("'")
        wrong_text = str(row["wrong_answer"]).strip().strip('"').strip("'")
        
        self.correct_label.config(text=correct_text)
        self.wrong_label.config(text=wrong_text)

        all_prereqs = [p.strip().strip('"').strip("'") for p in row["all_prerequisites"].split(',') if p.strip()]
        
        for i, prereq in enumerate(all_prereqs):
            var = tk.BooleanVar()
            
            cb = tk.Checkbutton(self.prereq_scroll_frame, text=prereq, variable=var, 
                               anchor="w", justify="left", wraplength=350,
                               font=('Arial', 9), bg=self.card_bg, fg=self.text_color,
                               selectcolor=self.card_bg, cursor='hand2',
                               activebackground=self.card_bg, activeforeground=self.text_color)
            cb.pack(anchor='w', pady=1, padx=5)
            
            self.prereq_vars.append((prereq, var))

        self.submit_button.config(command=lambda: self.save_response(index, row))
        self.update_progress()

    def save_response(self, index, row):
        selected = [pr for pr, var in self.prereq_vars if var.get()]
        if not selected:
            if not messagebox.askyesno("Confirm", "No prerequisite selected. Submit anyway?"):
                return

        human_text = ", ".join(selected)
        model_text = str(row["missing_prerequisite"]).strip().strip('"').strip("'")
        
        all_prereqs = [p.strip().strip('"').strip("'") for p in row["all_prerequisites"].split(',') if p.strip()]
        
        agreement_metrics = self.calculate_agreement_metrics(human_text, model_text, all_prereqs)

        entry = {
            "question_id": row["question_id"],
            "human_selected_prerequisite": human_text,
            "model_missing_prerequisite": model_text,
            "jaccard_similarity": agreement_metrics['jaccard'],
            "exact_match": agreement_metrics['exact_match'],
            "precision": agreement_metrics['precision'],
            "recall": agreement_metrics['recall'],
            "f1_score": agreement_metrics['f1_score'],
            "annotator": self.username,
            "question_bank": self.question_bank
        }

        save_file = os.path.join(OUTPUT_DIR, f"annotations_{self.username}.csv")
        new_df = pd.DataFrame([entry])
        new_df.to_csv(save_file, mode='a', header=not os.path.exists(save_file), index=False)
        self.done_ids.add(row["question_id"])

        next_idx = self.next_index()
        if next_idx is not None:
            self.load_question(next_idx)
        else:
            messagebox.showinfo("Completed!", 
                              f"Congratulations! You have completed all 50 questions in {self.question_bank}.\n\n"
                              f"Your annotations have been saved to:\n{save_file}\n\n"
                              f"Please send this file to the configurator via WhatsApp for final analysis.")
            self.show_completion_screen()

    def show_completion_screen(self):
        """Show completion screen with file location"""
        for widget in self.root.winfo_children():
            widget.destroy()
            
        completion_frame = tk.Frame(self.root, bg=self.bg_color)
        completion_frame.pack(expand=True, fill='both')
        
        center_frame = tk.Frame(completion_frame, bg=self.bg_color)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Success message
        tk.Label(center_frame, text="🎉 Annotation Complete!", 
                font=('Arial', 20, 'bold'), bg=self.bg_color, fg="#28a745").pack(pady=(0, 20))
        
        tk.Label(center_frame, text=f"You have successfully annotated all 50 questions in {self.question_bank}", 
                font=('Arial', 12), bg=self.bg_color, fg=self.header_color).pack(pady=(0, 20))
        
        # File location
        save_file = os.path.join(OUTPUT_DIR, f"annotations_{self.username}.csv")
        tk.Label(center_frame, text="Your annotation file:", 
                font=('Arial', 11, 'bold'), bg=self.bg_color, fg=self.header_color).pack()
        
        file_frame = tk.Frame(center_frame, bg="#e9ecef", relief='solid', bd=1)
        file_frame.pack(pady=(5, 20), padx=20)
        
        tk.Label(file_frame, text=save_file, font=('Arial', 10), 
                bg="#e9ecef", fg="#495057").pack(padx=15, pady=8)
        
        # Instructions
        tk.Label(center_frame, text="Next Steps:", 
                font=('Arial', 12, 'bold'), bg=self.bg_color, fg=self.header_color).pack(pady=(0, 10))
        
        instructions = [
            "1. Locate the file above on your computer",
            "2. Send it via WhatsApp to the configurator",
            "3. The configurator will merge all annotation files"
        ]
        
        for instruction in instructions:
            tk.Label(center_frame, text=instruction, font=('Arial', 10), 
                    bg=self.bg_color, fg=self.text_color).pack(anchor='w', pady=2)
        
        # Buttons
        button_frame = tk.Frame(center_frame, bg=self.bg_color)
        button_frame.pack(pady=(30, 0))
        
        open_folder_button = tk.Button(button_frame, text="Open File Location", 
                                     font=('Arial', 11, 'bold'), bg=self.button_bg, fg=self.button_text,
                                     relief='flat', padx=20, pady=8, cursor='hand2',
                                     command=lambda: os.startfile(OUTPUT_DIR))
        open_folder_button.pack(side='left', padx=(0, 10))
        
        logout_button = tk.Button(button_frame, text="Logout", 
                                font=('Arial', 11), bg="#6c757d", fg=self.button_text,
                                relief='flat', padx=20, pady=8, cursor='hand2',
                                command=self.show_login)
        logout_button.pack(side='left')

    def load_first_question(self):
        start_idx = self.next_index()
        if start_idx is not None:
            self.load_question(start_idx)
        else:
            self.question_label.config(text="All questions in your bank are already annotated!")
            self.submit_button.config(state='disabled')

    def merge_csvs(self):
        """Administrator function to merge CSV files from WhatsApp transfers"""
        # Look for CSV files in the OUTPUT_DIR
        csv_files = glob.glob(os.path.join(OUTPUT_DIR, "annotations_*.csv"))
        
        if not csv_files:
            messagebox.showwarning("No Files", 
                                 f"No annotation files found in:\n{OUTPUT_DIR}\n\n"
                                 "Please ensure all annotator CSV files have been transferred "
                                 "to this directory via WhatsApp.")
            return
            
        try:
            all_dataframes = []
            file_info = []
            
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    all_dataframes.append(df)
                    
                    # Extract info about the file
                    annotator = df['annotator'].iloc[0] if 'annotator' in df.columns and len(df) > 0 else "Unknown"
                    question_bank = df['question_bank'].iloc[0] if 'question_bank' in df.columns and len(df) > 0 else "Unknown"
                    count = len(df)
                    
                    file_info.append(f"• {annotator}: {question_bank} ({count} annotations)")
                    
                except Exception as e:
                    messagebox.showerror("File Error", f"Could not read {file_path}:\n{e}")
                    return
            
            # Combine all dataframes
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Show merge preview
            preview_text = f"Found {len(csv_files)} annotation files:\n\n" + "\n".join(file_info)
            preview_text += f"\n\nTotal annotations: {len(merged_df)}"
            
            if not messagebox.askyesno("Confirm Merge", 
                                     f"{preview_text}\n\nProceed with merging these files?"):
                return
            
            # Ask where to save merged file
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                title="Save Merged Annotations As",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname="merged_annotations_all_annotators.csv"
            )
            
            if save_path:
                merged_df.to_csv(save_path, index=False)
                
                # Generate summary report
                summary = self.generate_merge_summary(merged_df, file_info)
                
                messagebox.showinfo("Merge Complete!", 
                                  f"Successfully merged {len(csv_files)} files!\n\n"
                                  f"Saved to: {save_path}\n\n"
                                  f"Summary:\n{summary}")
                
        except Exception as e:
            messagebox.showerror("Merge Error", f"Failed to merge files:\n{e}")

    def generate_merge_summary(self, merged_df, file_info):
        """Generate a summary report of the merged data"""
        try:
            total_annotations = len(merged_df)
            unique_annotators = merged_df['annotator'].nunique() if 'annotator' in merged_df.columns else 0
            
            # Calculate average metrics
            avg_jaccard = merged_df['jaccard_similarity'].mean() if 'jaccard_similarity' in merged_df.columns else 0
            avg_f1 = merged_df['f1_score'].mean() if 'f1_score' in merged_df.columns else 0
            exact_matches = merged_df['exact_match'].sum() if 'exact_match' in merged_df.columns else 0
            
            summary = f"""Total Annotations: {total_annotations}
Annotators: {unique_annotators}
Average Jaccard Similarity: {avg_jaccard:.3f}
Average F1 Score: {avg_f1:.3f}
Exact Matches: {exact_matches}/{total_annotations} ({exact_matches/total_annotations*100:.1f}%)"""
            
            return summary
            
        except Exception as e:
            return f"Summary generation failed: {e}"

    def exit_app(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# Create and run the application
app = AnnotatorApp()
app.run()
