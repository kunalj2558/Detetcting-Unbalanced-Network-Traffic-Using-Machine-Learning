import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import winsound  # Windows beep sound


class EnsembleGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("⚡ Real-Time Network Anomaly Detection By Kunal, Ayush & Tanesh")
        self.master.configure(bg="#2c3e50")

        self.df = None
        self.stop_detection_flag = False

        style_font = ("Arial", 10, "bold")
        btn_bg = "#3498db"
        btn_fg = "#ffffff"

        self.top_frame = tk.Frame(master, bg="#2c3e50")
        self.top_frame.pack(pady=5)

        self.mid_frame = tk.Frame(master, bg="#2c3e50")
        self.mid_frame.pack(pady=5)

        self.bottom_frame = tk.Frame(master, bg="#2c3e50")
        self.bottom_frame.pack(pady=5)

        self.load_btn = tk.Button(self.top_frame, text="📂 Load Dataset", command=self.load_dataset,
                                  bg=btn_bg, fg=btn_fg, font=style_font, relief="raised", bd=3, padx=10)
        self.load_btn.grid(row=0, column=0, padx=5)

        self.xgb_n_estimators = self.add_entry(self.mid_frame, "XGBoost n_estimators", "200", 0)
        self.xgb_lr = self.add_entry(self.mid_frame, "XGBoost learning_rate", "0.05", 1)
        self.xgb_max_depth = self.add_entry(self.mid_frame, "XGBoost max_depth", "7", 2)
        self.rf_n_estimators = self.add_entry(self.mid_frame, "RF n_estimators", "200", 3)
        self.rf_max_depth = self.add_entry(self.mid_frame, "RF max_depth", "10", 4)

        self.train_btn = tk.Button(self.top_frame, text="🏋 Train Model", command=self.train_model,
                                   bg="#27ae60", fg=btn_fg, font=style_font, relief="raised", bd=3, padx=10)
        self.train_btn.grid(row=0, column=1, padx=5)

        self.detect_btn = tk.Button(self.top_frame, text="🚨 Start Detection", command=self.start_detection,
                                    bg="#e67e22", fg=btn_fg, font=style_font, relief="raised", bd=3, padx=10)
        self.detect_btn.grid(row=0, column=2, padx=5)

        self.stop_btn = tk.Button(self.top_frame, text="⛔ Stop", command=self.stop_detection,
                                  bg="#c0392b", fg=btn_fg, font=style_font, relief="raised", bd=3, padx=10)
        self.stop_btn.grid(row=0, column=3, padx=5)

        self.result_text = scrolledtext.ScrolledText(self.bottom_frame, height=18, width=110,
                                                     font=("Courier New", 10, "bold"),
                                                     bg="#1c2833", fg="#ecf0f1", relief="sunken", bd=5)
        self.result_text.pack(pady=5)

        self.canvas_frame = tk.Frame(master, bg="#2c3e50")
        self.canvas_frame.pack()

        # Models and data placeholders
        self.xgb_model = None
        self.rf_model = None
        self.meta_model = None
        self.X_test = None
        self.y_test = None

    def add_entry(self, parent, label_text, default_val, row):
        label = tk.Label(parent, text=label_text, bg="#2c3e50", fg="white", font=("Arial", 10, "bold"))
        label.grid(row=row, column=0, sticky='e', padx=5, pady=3)
        entry = tk.Entry(parent, bg="white", font=("Arial", 10), width=12)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, padx=5, pady=3)
        return entry

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            self.df = pd.read_csv(file_path)
            self.df.columns = self.df.columns.str.strip()
            self.df.dropna(inplace=True)
            if 'Label' not in self.df.columns:
                raise ValueError("No 'Label' column found in dataset.")
            self.result_text.insert(tk.END, "✅ Dataset loaded successfully.\n")
            self.result_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_smote(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts > 1].index

        mask = y.isin(valid_classes)

        X_filtered = X.loc[mask]
        y_filtered = y.loc[mask]

        if y_filtered.empty:
            raise ValueError("No class has enough samples for SMOTE.")

        min_class_count = class_counts[valid_classes].min()
        k_neighbors = min(5, max(1, min_class_count - 1))

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

        return X_resampled, y_resampled

    def train_model(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        try:
            xgb_n_estimators = int(self.xgb_n_estimators.get())
            xgb_lr = float(self.xgb_lr.get())
            xgb_max_depth = int(self.xgb_max_depth.get())
            rf_n_estimators = int(self.rf_n_estimators.get())
            rf_max_depth = int(self.rf_max_depth.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid hyperparameter values.")
            return

        df = self.df.copy()

        # Encode categorical features except 'Label'
        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            if col != 'Label':
                df[col] = label_encoder.fit_transform(df[col])

        # Encode labels initially
        df['Label'] = label_encoder.fit_transform(df['Label'])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Remove classes with fewer than 2 samples
        class_counts = df['Label'].value_counts()
        df = df[df['Label'].isin(class_counts[class_counts > 1].index)]

        # Sample 20% stratified for training/testing
        _, df_sampled = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

        selector = VarianceThreshold(threshold=0.01)
        X = pd.DataFrame(selector.fit_transform(df_sampled.drop(columns='Label')))
        y = df_sampled['Label'].reset_index(drop=True)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X))

        try:
            X_res, y_res = self.apply_smote(X, y)
        except Exception as e:
            messagebox.showerror("SMOTE Error", f"SMOTE failed: {e}")
            return

        # Re-encode labels to be contiguous starting at 0
        le_final = LabelEncoder()
        y_res = le_final.fit_transform(y_res)

        self.plot_class_distribution(y, y_res)

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        xgb = XGBClassifier(
            n_estimators=xgb_n_estimators,
            learning_rate=xgb_lr,
            max_depth=xgb_max_depth,
            random_state=42,
            eval_metric='logloss'
        )
        rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)

        xgb.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        xgb_preds_train = xgb.predict_proba(X_train)[:, 1]
        rf_preds_train = rf.predict_proba(X_train)[:, 1]
        meta_train_input = np.column_stack((xgb_preds_train, rf_preds_train))

        meta_model = LogisticRegression(class_weight='balanced', random_state=42)
        meta_model.fit(meta_train_input, y_train)

        xgb_preds_test = xgb.predict_proba(X_test)[:, 1]
        rf_preds_test = rf.predict_proba(X_test)[:, 1]
        stacked_test = np.column_stack((xgb_preds_test, rf_preds_test))

        final_preds = meta_model.predict(stacked_test)

        acc = accuracy_score(y_test, final_preds)
        report = classification_report(y_test, final_preds, zero_division=1)
        cm = confusion_matrix(y_test, final_preds)

        self.result_text.insert(tk.END, f"\n✅ Accuracy: {acc:.4f}\n")
        self.result_text.insert(tk.END, f"\n📊 Classification Report:\n{report}\n")
        self.result_text.insert(tk.END, f"🧮 Confusion Matrix:\n{cm}\n")
        self.result_text.see(tk.END)

        self.plot_confusion_matrix(cm)

        self.xgb_model = xgb
        self.rf_model = rf
        self.meta_model = meta_model
        self.X_test = pd.DataFrame(X_test).reset_index(drop=True)
        self.y_test = pd.Series(y_test).reset_index(drop=True)

        winsound.Beep(750, 500)

    def start_detection(self):
        if not self.meta_model:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        self.result_text.insert(tk.END, "\n🚨 Starting Real-Time Detection...\n")
        self.result_text.see(tk.END)

        self.stop_detection_flag = False
        self.current_index = 0
        self.run_detection_step()

    def run_detection_step(self):
        if self.stop_detection_flag or self.current_index >= len(self.X_test):
            if self.stop_detection_flag:
                self.result_text.insert(tk.END, "\n🛑 Detection Stopped.\n")
            else:
                self.result_text.insert(tk.END, "✅ Detection Completed.\n")
            self.result_text.see(tk.END)
            return

        i = self.current_index
        x_row = self.X_test.iloc[i:i + 1]
        true_label = self.y_test.iloc[i]

        xgb_prob = self.xgb_model.predict_proba(x_row)[:, 1]
        rf_prob = self.rf_model.predict_proba(x_row)[:, 1]
        stacked = np.column_stack((xgb_prob, rf_prob))
        pred = self.meta_model.predict(stacked)[0]

        alert = f"Packet {i + 1}: Predicted={pred}, Actual={true_label}"
        if pred != true_label:
            alert += " ⚠ ALERT!"
            winsound.Beep(1000, 300)
            messagebox.showwarning("⚠ Network Anomaly", f"Packet {i + 1} anomaly detected!")

        self.result_text.insert(tk.END, alert + "\n")
        self.result_text.see(tk.END)

        self.current_index += 1
        self.master.after(300, self.run_detection_step)

    def stop_detection(self):
        self.stop_detection_flag = True

    def plot_class_distribution(self, y_before, y_after):
        plt.close('all')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(x=y_before, ax=axes[0])
        axes[0].set_title("Before SMOTE")
        sns.countplot(x=y_after, ax=axes[1])
        axes[1].set_title("After SMOTE")
        plt.tight_layout()
        self.show_plot_on_canvas(fig)

    def plot_confusion_matrix(self, cm):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        self.show_plot_on_canvas(fig)

    def show_plot_on_canvas(self, fig):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = EnsembleGUI(root)
    root.mainloop()
