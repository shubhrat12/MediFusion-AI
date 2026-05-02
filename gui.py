import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
from threading import Thread
import io
import contextlib
from datetime import datetime

# Import the medical pipeline
try:
    from run_all_models import MedicalAIPipeline
except ImportError:
    messagebox.showerror("Import Error", "Could not import MedicalAIPipeline from run_all_models.py\nMake sure the file is in the same directory.")
    sys.exit(1)

class MedicalAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical AI Pipeline - Advanced Diagnostic Assistant")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)
        
        # Configure modern theme
        self.setup_modern_theme()
        
        # Configure root grid weights for proper resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize pipeline as None
        self.pipeline = None
        
        # Variables for inputs
        self.image_path = tk.StringVar()
        
        # Initialize feature variables
        self.heart_vars = []
        self.kidney_vars = []
        self.diabetes_vars = []
        
        self.setup_ui()
        
    def setup_modern_theme(self):
        # Configure modern styling
        style = ttk.Style()
        
        # Configure colors for modern look
        bg_color = "#f8f9fa"
        primary_color = "#007bff"
        accent_color = "#17a2b8"
        success_color = "#28a745"
        danger_color = "#dc3545"
        
        self.root.configure(bg=bg_color)
        
        # Configure modern fonts with better sizes
        self.title_font = ("Segoe UI", 14, "bold")
        self.subtitle_font = ("Segoe UI", 12)
        self.body_font = ("Segoe UI", 10)
        self.small_font = ("Segoe UI", 9)
        
        # Configure ttk styles with enhanced appearance
        style.configure('Title.TLabelframe', font=self.title_font)
        style.configure('Title.TLabelframe.Label', font=self.title_font, foreground=primary_color)
        style.configure('Subtitle.TLabel', font=self.subtitle_font)
        style.configure('Body.TLabel', font=self.body_font)
        style.configure('Primary.TButton', font=self.body_font, padding=10)
        
        # Configure custom styles for widgets
        style.configure('Output.TFrame', background='white')
        style.configure('Status.TLabel', font=self.body_font, foreground=accent_color)
        style.configure('Error.TLabel', font=self.body_font, foreground=danger_color)
        style.configure('Success.TLabel', font=self.body_font, foreground=success_color)
        
    def setup_ui(self):
        # Create main container with proper grid configuration
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Create paned window for better layout management
        paned_window = ttk.PanedWindow(main_container, orient='horizontal')
        paned_window.grid(row=0, column=0, sticky="nsew")
        
        # Left panel for inputs
        left_panel = self.create_left_panel(paned_window)
        paned_window.add(left_panel, weight=3)
        
        # Right panel for output
        right_panel = self.create_right_panel(paned_window)
        paned_window.add(right_panel, weight=2)
        
    def create_left_panel(self, parent):
        # Create scrollable left panel
        left_frame = ttk.Frame(parent)
        
        # Create canvas and scrollbar for left panel
        canvas = tk.Canvas(left_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar_left = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrollable region
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        def configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_left.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_left.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            
        canvas.bind('<Enter>', bind_mousewheel)
        canvas.bind('<Leave>', unbind_mousewheel)
        
        # Setup input sections
        self.setup_model_paths(scrollable_frame)
        self.setup_image_input(scrollable_frame)
        self.setup_text_input(scrollable_frame)
        self.setup_tabular_inputs(scrollable_frame)
        self.setup_run_section(scrollable_frame)
        
        return left_frame
    
    def create_right_panel(self, parent):
        # Create right panel for output
        right_frame = ttk.Frame(parent)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Output section
        self.setup_output_section(right_frame)
        
        return right_frame
        
    def setup_model_paths(self, parent):
        frame = ttk.LabelFrame(parent, text="🔧 Model Configuration", style='Title.TLabelframe', padding=20)
        frame.pack(fill="x", padx=10, pady=15)
        frame.grid_columnconfigure(1, weight=1)
        
        # Model path variables with default paths
        self.model_paths = {
            'image_model': tk.StringVar(value=r"Image\tranfer_learning.pt"),
            'text_model': tk.StringVar(value=r"Text\final_finetuned.pt"),
            'heart_model': tk.StringVar(value=r"Table\Heart_Disease\tabular_model.pt"),
            'kidney_model': tk.StringVar(value=r"Table\Kidney\ckd_model.joblib"),
            'diabetes_model': tk.StringVar(value=r"Table\Diabetes\diabetes_model.joblib"),
            'diabetes_scaler': tk.StringVar(value=r"Table\Diabetes\diabetes_scaler.joblib")
        }
        
        model_labels = {
            'image_model': '🖼️ Image Model',
            'text_model': '📝 Text Model',
            'heart_model': '❤️ Heart Model',
            'kidney_model': '🫘 Kidney Model',
            'diabetes_model': '🩺 Diabetes Model',
            'diabetes_scaler': '⚖️ Diabetes Scaler'
        }
        
        for i, (key, label) in enumerate(model_labels.items()):
            # Create label
            label_widget = ttk.Label(frame, text=label, style='Body.TLabel')
            label_widget.grid(row=i, column=0, sticky="w", padx=10, pady=8)
            
            # Create entry
            entry = ttk.Entry(frame, textvariable=self.model_paths[key], font=self.body_font)
            entry.grid(row=i, column=1, sticky="ew", padx=10, pady=8)
            
            # Create browse button
            browse_btn = ttk.Button(frame, text="📁 Browse", 
                                  command=lambda k=key: self.browse_model_file(k))
            browse_btn.grid(row=i, column=2, padx=10, pady=8)
        
        # Initialize models button
        init_frame = ttk.Frame(frame)
        init_frame.grid(row=len(model_labels), column=0, columnspan=3, pady=20)
        
        self.init_button = ttk.Button(init_frame, text="🚀 Initialize Models", 
                                     style='Primary.TButton', command=self.initialize_models)
        self.init_button.pack()
        
        # Status indicator
        self.init_status = ttk.Label(init_frame, text="⚠️ Models not initialized", 
                                   foreground="#dc3545", font=self.small_font)
        self.init_status.pack(pady=5)
        
    def browse_model_file(self, model_key):
        filename = filedialog.askopenfilename(
            title=f"Select {model_key} file",
            filetypes=[("Model files", "*.pt *.joblib"), ("All files", "*.*")]
        )
        if filename:
            self.model_paths[model_key].set(filename)
            
    def initialize_models(self):
        try:
            self.init_button.config(state="disabled", text="🔄 Initializing...")
            self.init_status.config(text="🔄 Initializing models...", foreground="#007bff")
            self.root.update()
            
            # Convert StringVar to regular dict
            paths = {key: var.get() for key, var in self.model_paths.items()}
            
            # Check if all paths exist
            for key, path in paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")
            
            # Initialize pipeline
            self.pipeline = MedicalAIPipeline(paths)
            
            # Update status
            self.init_status.config(text="✅ Models initialized successfully!", foreground="#28a745")
            messagebox.showinfo("Success", "Models initialized successfully!")
            
        except Exception as e:
            self.init_status.config(text="❌ Initialization failed!", foreground="#dc3545")
            messagebox.showerror("Error", f"Failed to initialize models: {str(e)}")
        finally:
            self.init_button.config(state="normal", text="🚀 Initialize Models")
    
    def setup_image_input(self, parent):
        frame = ttk.LabelFrame(parent, text="🖼️ Chest X-Ray Image", style='Title.TLabelframe', padding=20)
        frame.pack(fill="x", padx=10, pady=15)
        frame.grid_columnconfigure(1, weight=1)
        
        # Create inner frame for better alignment
        input_frame = ttk.Frame(frame)
        input_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Image label with improved styling
        ttk.Label(input_frame, 
                 text="📁 Select Image File:", 
                 style='Body.TLabel').grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        # Entry widget with improved appearance
        entry = ttk.Entry(input_frame, 
                         textvariable=self.image_path, 
                         font=self.body_font)
        entry.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Browse button with better styling
        browse_btn = ttk.Button(input_frame, 
                              text="🔍 Browse", 
                              command=self.browse_image,
                              style='Primary.TButton')
        browse_btn.grid(row=0, column=2, padx=(5, 0), pady=5)
        
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select chest X-ray image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.image_path.set(filename)
    
    def setup_text_input(self, parent):
        frame = ttk.LabelFrame(parent, text="📝 Radiology Report", style='Title.TLabelframe', padding=20)
        frame.pack(fill="x", padx=10, pady=15)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        ttk.Label(frame, text="Enter clinical text or radiology report:", style='Body.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.grid(row=1, column=0, sticky="ew", pady=5)
        text_frame.grid_columnconfigure(0, weight=1)
        
        self.text_widget = tk.Text(text_frame, height=4, wrap="word", font=self.body_font,
                                 relief="solid", borderwidth=1)
        self.text_widget.grid(row=0, column=0, sticky="ew")
        
    def setup_tabular_inputs(self, parent):
        # Create notebook for tabular inputs with dynamic sizing
        notebook_frame = ttk.LabelFrame(parent, text="📊 Clinical Parameters", style='Title.TLabelframe', padding=20)
        notebook_frame.pack(fill="both", expand=True, padx=10, pady=15)
        notebook_frame.grid_columnconfigure(0, weight=1)
        notebook_frame.grid_rowconfigure(0, weight=1)
        
        notebook = ttk.Notebook(notebook_frame)
        notebook.grid(row=0, column=0, sticky="nsew", pady=10)
        
        # Heart disease tab
        heart_frame = ttk.Frame(notebook)
        notebook.add(heart_frame, text="❤️ Heart Disease")
        heart_frame.grid_columnconfigure(0, weight=1)
        heart_frame.grid_rowconfigure(0, weight=1)
        self.setup_heart_inputs(heart_frame)
        
        # Kidney disease tab
        kidney_frame = ttk.Frame(notebook)
        notebook.add(kidney_frame, text="🫘 Kidney Disease")
        kidney_frame.grid_columnconfigure(0, weight=1)
        kidney_frame.grid_rowconfigure(0, weight=1)
        self.setup_kidney_inputs(kidney_frame)
        
        # Diabetes tab
        diabetes_frame = ttk.Frame(notebook)
        notebook.add(diabetes_frame, text="🩺 Diabetes")
        diabetes_frame.grid_columnconfigure(0, weight=1)
        diabetes_frame.grid_rowconfigure(0, weight=1)
        self.setup_diabetes_inputs(diabetes_frame)
        
    def setup_heart_inputs(self, parent):
        # Create a frame that will expand with the window
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create canvas with scrollbar for dynamic content
        canvas = tk.Canvas(main_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure canvas scroll region
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Set the canvas width to match the parent width
            canvas_width = event.width if event else parent.winfo_width()
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout for canvas and scrollbar
        canvas.grid(row=0, column=0, sticky="nsew", padx=(5, 0))
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid for content
        for i in range(6):
            scrollable_frame.grid_columnconfigure(i, weight=1)
        
        # Bind canvas resizing
        canvas.bind("<Configure>", lambda e: configure_scroll_region(e))
        
        # Heart disease features with improved layout
        heart_labels = [
            "Age", "Sex (1=Male, 0=Female)", 
            "Chest Pain: Typical Angina", "Chest Pain: Atypical Angina", 
            "Chest Pain: Non-anginal", "Chest Pain: Asymptomatic",
            "Resting Blood Pressure", "Cholesterol", "Fasting Blood Sugar > 120",
            "ECG: Normal", "ECG: ST-T Abnormality", "ECG: LV Hypertrophy", 
            "Max Heart Rate", "Exercise Angina (1=Yes)", "ST Depression",
            "ST Slope: Upsloping", "ST Slope: Flat", "ST Slope: Downsloping",
            "Vessels: 0", "Vessels: 1", "Vessels: 2", "Vessels: 3",
            "Thal: Normal", "Thal: Fixed", "Thal: Reversible", "Thal: Unknown",
            "Extra 1", "Extra 2"
        ]
        
        # Initialize variables and create UI elements
        self.heart_vars = []
        for i, label in enumerate(heart_labels):
            var = tk.StringVar()
            self.heart_vars.append(var)
            
            row = i // 3
            col = i % 3
            
            # Create container with proper spacing
            container = ttk.Frame(scrollable_frame)
            container.grid(row=row, column=col, sticky="ew", padx=10, pady=8)
            container.grid_columnconfigure(1, weight=1)
            
            # Label with improved styling
            ttk.Label(container, text=f"{label}:", 
                     font=self.small_font,
                     style='Body.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 2))
            
            # Input widget with consistent sizing
            if any(keyword in label.lower() for keyword in ["chest pain", "ecg:", "st slope:", "vessels:", "thal:"]):
                widget = ttk.Combobox(container, textvariable=var, values=["0", "1"],
                                    font=self.small_font)
                widget.set("0")
            elif any(keyword in label.lower() for keyword in ["sex", "fasting", "exercise"]):
                widget = ttk.Combobox(container, textvariable=var, values=["0", "1"],
                                    font=self.small_font)
                widget.set("0")
            else:
                widget = ttk.Entry(container, textvariable=var, font=self.small_font)
                # Set default values
                if "age" in label.lower():
                    var.set("50")
                elif "pressure" in label.lower():
                    var.set("120")
                elif "cholesterol" in label.lower():
                    var.set("200")
                elif "heart rate" in label.lower():
                    var.set("150")
                else:
                    var.set("0")
            
            widget.grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def setup_kidney_inputs(self, parent):
        # Create main frame with dynamic sizing
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(main_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure canvas scroll region
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = event.width if event else parent.winfo_width()
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Layout for canvas and scrollbar
        canvas.grid(row=0, column=0, sticky="nsew", padx=(5, 0))
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid columns for content
        for i in range(3):
            scrollable_frame.grid_columnconfigure(i, weight=1)
        
        # Bind canvas resizing
        canvas.bind("<Configure>", lambda e: configure_scroll_region(e))
        
        kidney_labels = [
            "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
            "RBC", "Pus Cell", "Pus Cell Clumps", "Bacteria", "Blood Glucose Random",
            "Blood Urea", "Serum Creatinine", "Sodium", "Potassium", "Hemoglobin",
            "Packed Cell Volume", "WBC Count", "RBC Count", "Hypertension",
            "Diabetes Mellitus", "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia"
        ]
        
        categorical_fields = {
            5: ["normal", "abnormal"], 6: ["normal", "abnormal"], 7: ["present", "notpresent"], 
            8: ["present", "notpresent"], 18: ["yes", "no"], 19: ["yes", "no"], 20: ["yes", "no"],
            21: ["good", "poor"], 22: ["yes", "no"], 23: ["yes", "no"]
        }
        
        self.kidney_vars = []
        for i, label in enumerate(kidney_labels):
            var = tk.StringVar()
            self.kidney_vars.append(var)
            
            row = i // 3
            col = i % 3
            
            # Create container with improved spacing
            container = ttk.Frame(scrollable_frame)
            container.grid(row=row, column=col, sticky="ew", padx=10, pady=8)
            container.grid_columnconfigure(1, weight=1)
            
            # Label with consistent styling
            ttk.Label(container, text=f"{label}:", 
                     font=self.small_font,
                     style='Body.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 2))
            
            # Input widget with appropriate sizing
            if i in categorical_fields:
                widget = ttk.Combobox(container, textvariable=var,
                                    values=categorical_fields[i],
                                    font=self.small_font)
                widget.set(categorical_fields[i][0])
            else:
                widget = ttk.Entry(container, textvariable=var, font=self.small_font)
                defaults = {0: "50", 1: "80", 2: "1.020", 9: "121", 10: "36", 11: "1.2", 
                          12: "15", 13: "4.6", 14: "15", 15: "44", 16: "7800", 17: "5.2"}
                var.set(defaults.get(i, "0"))
            
            widget.grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def setup_diabetes_inputs(self, parent):
        # Create main frame with dynamic sizing
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(main_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure canvas scroll region
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = event.width if event else parent.winfo_width()
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Layout for canvas and scrollbar
        canvas.grid(row=0, column=0, sticky="nsew", padx=(5, 0))
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid columns for content
        for i in range(3):
            scrollable_frame.grid_columnconfigure(i, weight=1)
        
        # Bind canvas resizing
        canvas.bind("<Configure>", lambda e: configure_scroll_region(e))
        
        diabetes_labels = [
            "High BP", "High Cholesterol", "Cholesterol Check", "BMI", "Smoker",
            "Stroke", "Heart Disease/Attack", "Physical Activity", "Fruits", "Vegetables",
            "Heavy Alcohol Consumption", "Any Healthcare", "No Doctor Cost", "General Health (1-5)",
            "Mental Health Days", "Physical Health Days", "Difficulty Walking", "Sex (1=Male)",
            "Age", "Education (1-6)", "Income (1-8)"
        ]
        
        self.diabetes_vars = []
        for i, label in enumerate(diabetes_labels):
            var = tk.StringVar()
            self.diabetes_vars.append(var)
            
            row = i // 3
            col = i % 3
            
            # Create container with improved spacing
            container = ttk.Frame(scrollable_frame)
            container.grid(row=row, column=col, sticky="ew", padx=10, pady=8)
            container.grid_columnconfigure(1, weight=1)
            
            # Label with consistent styling
            ttk.Label(container, text=f"{label}:", 
                     font=self.small_font,
                     style='Body.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 2))
            
            # Input widget with appropriate sizing and style
            if i == 3:  # BMI
                widget = ttk.Entry(container, textvariable=var, font=self.small_font)
                var.set("25.0")
            elif i == 13:  # General Health
                widget = ttk.Combobox(container, textvariable=var, 
                                    values=["1", "2", "3", "4", "5"],
                                    font=self.small_font)
                var.set("3")
            elif i == 18:  # Age
                widget = ttk.Entry(container, textvariable=var, font=self.small_font)
                var.set("45")
            elif i == 19:  # Education
                widget = ttk.Combobox(container, textvariable=var,
                                    values=["1", "2", "3", "4", "5", "6"],
                                    font=self.small_font)
                var.set("4")
            elif i == 20:  # Income
                widget = ttk.Combobox(container, textvariable=var,
                                    values=["1", "2", "3", "4", "5", "6", "7", "8"],
                                    font=self.small_font)
                var.set("5")
            elif i in [14, 15]:  # Mental/Physical health days
                widget = ttk.Entry(container, textvariable=var, font=self.small_font)
                var.set("0")
            else:  # Binary features
                widget = ttk.Combobox(container, textvariable=var,
                                    values=["0", "1"],
                                    font=self.small_font)
                var.set("0")
            
            widget.grid(row=0, column=1, sticky="ew", padx=(5, 0))
    
    def setup_run_section(self, parent):
        frame = ttk.LabelFrame(parent, text="🚀 Execute Analysis", style='Title.TLabelframe', padding=20)
        frame.pack(fill="x", padx=10, pady=15)
        frame.grid_columnconfigure(0, weight=1)
        
        # Create inner frame for better alignment
        inner_frame = ttk.Frame(frame)
        inner_frame.grid(row=0, column=0, sticky="ew", pady=10)
        inner_frame.grid_columnconfigure(0, weight=1)
        
        # Enhanced run button with modern styling
        self.run_button = ttk.Button(inner_frame, 
                                   text="🔬 Run Medical AI Pipeline",
                                   style='Primary.TButton',
                                   command=self.run_pipeline)
        self.run_button.grid(row=0, column=0, pady=(5, 15))
        
        # Progress bar with better visibility
        self.progress = ttk.Progressbar(inner_frame, 
                                      mode='indeterminate',
                                      length=400)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Status label with dynamic styling
        self.status_label = ttk.Label(inner_frame,
                                    text="✨ Ready to analyze",
                                    font=self.body_font,
                                    style='Status.TLabel')
        self.status_label.grid(row=2, column=0, pady=5)
        
        # Add separator for visual appeal
        ttk.Separator(inner_frame, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=15)
        
    def setup_output_section(self, parent):
        frame = ttk.LabelFrame(parent, text="🤖 AI Medical Recommendation", style='Title.TLabelframe', padding=20)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=15)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # Create text frame with improved styling
        text_frame = ttk.Frame(frame, style='Output.TFrame')
        text_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        # Enhanced text widget with custom styling
        self.output_text = tk.Text(text_frame, 
                                 wrap="word", 
                                 font=("Segoe UI", 11),
                                 relief="solid", 
                                 borderwidth=1, 
                                 state="disabled",
                                 bg="white", 
                                 padx=20, 
                                 pady=20,
                                 spacing1=5,  # Space above each line
                                 spacing2=2,  # Space between lines
                                 spacing3=5)  # Space below each line
        self.output_text.grid(row=0, column=0, sticky="nsew")
        
        # Modern scrollbar
        output_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.output_text.yview)
        output_scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        
        # Bind mousewheel to output text with smooth scrolling
        def on_mousewheel(event):
            delta = -1 * (event.delta / 120)
            self.output_text.yview_scroll(int(delta), "units")
        self.output_text.bind("<MouseWheel>", on_mousewheel)
        
        # Initial welcome message with improved formatting
        self.output_text.config(state="normal")
        welcome_msg = """🏥 Welcome to Medical AI Pipeline

This advanced system integrates multiple AI models to provide comprehensive medical analysis:

📊 Analysis Components:
• Chest X-ray Analysis
  - Deep learning model for detecting respiratory conditions
  - State-of-the-art image processing

• Clinical Text Understanding
  - Natural language processing
  - Context-aware medical text analysis

• Health Risk Assessment
  - Heart Disease Risk Analysis
  - Kidney Disease Prediction
  - Diabetes Risk Evaluation

⚕️ How to Use:
1. Initialize the AI models
2. Upload a chest X-ray image
3. Enter clinical notes/observations
4. Fill in the required health parameters
5. Click "Run Medical AI Pipeline"

⚠️ Important Note:
All results are AI-generated recommendations and should be 
validated by qualified medical professionals."""

        self.output_text.insert("1.0", welcome_msg)
        
        # Add tags for styling specific parts
        self.output_text.tag_configure("heading", font=("Segoe UI", 14, "bold"), foreground="#007bff")
        self.output_text.tag_configure("subheading", font=("Segoe UI", 12, "bold"), foreground="#17a2b8")
        self.output_text.tag_configure("important", foreground="#dc3545")
        
        # Apply tags
        self.output_text.tag_add("heading", "1.0", "1.end")
        self.output_text.tag_add("subheading", "5.0", "5.end")
        self.output_text.tag_add("important", "24.0", "26.end")
        
        self.output_text.config(state="disabled")
    
    def validate_inputs(self):
        if self.pipeline is None:
            messagebox.showerror("Error", "Please initialize models first!")
            return False
        
        if not self.image_path.get() or not os.path.exists(self.image_path.get()):
            messagebox.showerror("Error", "Please select a valid chest X-ray image!")
            return False
        
        clinical_text = self.text_widget.get("1.0", tk.END).strip()
        if not clinical_text:
            messagebox.showerror("Error", "Please enter clinical text!")
            return False
        
        try:
            # Validate all tabular inputs
            for var_list, name in [(self.heart_vars, "heart"), (self.kidney_vars, "kidney"), (self.diabetes_vars, "diabetes")]:
                for var in var_list:
                    val = var.get().strip()
                    if not val:
                        messagebox.showerror("Error", f"Please fill all {name} disease fields!")
                        return False
        except Exception as e:
            messagebox.showerror("Error", f"Input validation error: {str(e)}")
            return False
        
        return True
    
    def run_pipeline(self):
        if not self.validate_inputs():
            return
        
        # Update UI state
        self.run_button.config(state="disabled", text="🔄 Processing...")
        self.progress.start()
        self.status_label.config(text="🔬 Analyzing medical data...")
        
        # Clear output
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", "🔄 Running pipeline... Please wait while we analyze your medical data...")
        self.output_text.config(state="disabled")
        
        # Run in separate thread
        thread = Thread(target=self._run_pipeline_thread)
        thread.daemon = True
        thread.start()
    
    def _run_pipeline_thread(self):
        try:
            # Collect inputs
            image_path = self.image_path.get()
            clinical_text = self.text_widget.get("1.0", tk.END).strip()
            
            # Process tabular data
            heart_features = [float(var.get()) for var in self.heart_vars]
            
            kidney_features = []
            for i, var in enumerate(self.kidney_vars):
                val = var.get().strip()
                if i in [5, 6, 7, 8, 18, 19, 20, 21, 22, 23]:
                    kidney_features.append(val)
                else:
                    kidney_features.append(float(val))
            
            diabetes_features = [float(var.get()) for var in self.diabetes_vars]
            
            # Update status
            self.root.after(0, lambda: self.status_label.config(text="🧠 AI models processing..."))
            
            # Run pipeline
            print("Starting Medical AI Pipeline...")
            print(f"Image path: {image_path}")
            print(f"Clinical text length: {len(clinical_text)} characters")
            print(f"Heart features: {len(heart_features)} values")
            print(f"Kidney features: {len(kidney_features)} values") 
            print(f"Diabetes features: {len(diabetes_features)} values")
            
            result = self.pipeline.run_complete_pipeline(
                image_path=image_path,
                clinical_text=clinical_text,
                heart_features=heart_features,
                kidney_features=kidney_features,
                diabetes_features=diabetes_features
            )
            
            print("Pipeline completed successfully!")
            
            # Update GUI in main thread
            self.root.after(0, self._update_output, result)
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            self.root.after(0, self._show_error, str(e))
    
    def _update_output(self, result):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        
        # Format the output nicely
        formatted_result = f"🏥 MEDICAL AI ANALYSIS REPORT\n"
        formatted_result += "=" * 60 + "\n\n"
        formatted_result += "📊 ANALYSIS COMPLETE\n"
        formatted_result += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        formatted_result += "🤖 AI RECOMMENDATIONS:\n"
        formatted_result += "-" * 40 + "\n\n"
        formatted_result += result
        formatted_result += "\n\n" + "=" * 60
        formatted_result += "\n⚠️  IMPORTANT DISCLAIMER:"
        formatted_result += "\nThese are AI-generated recommendations for informational purposes only."
        formatted_result += "\nAlways consult with qualified healthcare professionals for medical decisions."
        
        self.output_text.insert("1.0", formatted_result)
        self.output_text.config(state="disabled")
        
        # Reset UI state
        self.progress.stop()
        self.run_button.config(state="normal", text="🔬 Run Medical AI Pipeline")
        self.status_label.config(text="✅ Analysis completed successfully!")
        
        # Scroll to top of output
        self.output_text.see("1.0")
    
    def _show_error(self, error_msg):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        
        error_text = f"❌ ANALYSIS ERROR\n"
        error_text += "=" * 60 + "\n\n"
        error_text += f"An error occurred during analysis:\n\n"
        error_text += f"Error Details: {error_msg}\n\n"
        error_text += "Please check:\n"
        error_text += "• All model paths are correct\n"
        error_text += "• Models are properly initialized\n"
        error_text += "• All input fields are filled correctly\n"
        error_text += "• Image file exists and is accessible\n\n"
        error_text += "If the problem persists, check the console for detailed error logs."
        
        self.output_text.insert("1.0", error_text)
        self.output_text.config(state="disabled")
        
        # Reset UI state
        self.progress.stop()
        self.run_button.config(state="normal", text="🔬 Run Medical AI Pipeline")
        self.status_label.config(text="❌ Analysis failed - check inputs")
        
        messagebox.showerror("Pipeline Error", f"Analysis failed: {error_msg}")

def main():
    root = tk.Tk()
    app = MedicalAIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()