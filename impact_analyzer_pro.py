import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ImpactKinematicsEngine:
    @staticmethod
    def process_telemetry(t_raw, E_raw, F_raw, mass, height):
        g = 9.80665
        E0_nominal = mass * g * height
        v0 = np.sqrt(2.0 * g * height)

        t = np.array(t_raw, dtype=float)
        E = np.array(E_raw, dtype=float)

        # Ensure sorted and zero-based time
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        E = E[sort_idx]
        t = t - t[0]

        # Fix scale if time was given in milliseconds
        if np.max(t) > 2.0:
            t = t / 1000.0

        # Fix scale if energy was in mJ
        if np.max(E) > 1000.0 and E0_nominal < 500.0:
            E = E / 1000.0

        # Velocity profile: v(t) = sqrt(2 * max(0, E0 - E(t)) / m)
        v = np.zeros_like(t)
        for i in range(len(t)):
            rem_ke = E0_nominal - E[i]
            v[i] = np.sqrt(2.0 * rem_ke / mass) if rem_ke > 0 else 0.0

        # Numerical integration for penetration deflection: d(t) = integral(v dt)
        d = np.zeros_like(t)
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            d[i] = d[i-1] + 0.5 * (v[i] + v[i-1]) * dt

        d_mm = d * 1000.0

        # Force calculation: if F_raw was zero or missing, derive F = dE / dd
        if F_raw is None or len(F_raw) == 0 or np.max(F_raw) == 0.0:
            d_meters = d.copy()
            # Avoid division by zero on identical displacement points
            d_meters[1:] = np.where(d_meters[1:] == d_meters[:-1], d_meters[1:] + 1e-6, d_meters[1:])
            F_N = np.gradient(E, d_meters + 1e-9)
            F_N = np.clip(F_N, 0, None)
        else:
            F_N = np.array(F_raw, dtype=float)[sort_idx]

        F_kN = F_N / 1000.0

        metrics = {
            "E0": E0_nominal,
            "v0": v0,
            "F_max": float(np.max(F_kN)) if len(F_kN) > 0 else 0.0,
            "d_max": float(np.max(d_mm)) if len(d_mm) > 0 else 0.0,
            "E_absorbed": float(np.max(E)) if len(E) > 0 else 0.0
        }
        return t * 1000.0, E, F_kN, v, d_mm, metrics


class GlobeTexApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("RVCE Mech -Impact Tester Interface")
        self.geometry("1400x880")

        self.is_monitoring = False
        self.monitor_thread = None

        self.time_data = []
        self.energy_data = []
        self.force_data = []

        self._build_sidebar()
        self._build_plot_area()

        # Load realistic working dataset on startup
        self.generate_and_render_sample()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        ctk.CTkLabel(
            self.sidebar, 
            text="IMPACT CONTROLLER", 
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(padx=20, pady=(20, 5), anchor="w")

        ctk.CTkLabel(
            self.sidebar, 
            text="Modernized Telemetry Suite",
            text_color="gray70", 
            font=ctk.CTkFont(size=12), 
            justify="left"
        ).pack(padx=20, pady=(0, 15), anchor="w")

        # Dynamic Height Sensor Box
        sensor_frame = ctk.CTkFrame(self.sidebar)
        sensor_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(
            sensor_frame, 
            text="DYNAMIC HEIGHT SENSOR", 
            font=ctk.CTkFont(size=12, weight="bold"), 
            text_color="#3498db"
        ).pack(padx=10, pady=(8, 2), anchor="w")

        self.lbl_sensor_status = ctk.CTkLabel(
            sensor_frame, 
            text="Sensor: ACTIVE (0 - 200 cm)", 
            font=ctk.CTkFont(size=11), 
            text_color="#2ecc71"
        )
        self.lbl_sensor_status.pack(padx=10, pady=0, anchor="w")

        h_box_frame = ctk.CTkFrame(sensor_frame, fg_color="transparent")
        h_box_frame.pack(fill="x", padx=10, pady=(6, 8))

        ctk.CTkLabel(h_box_frame, text="Height (m):", font=ctk.CTkFont(size=13)).pack(side="left")
        self.entry_height = ctk.CTkEntry(h_box_frame, width=90, font=("Consolas", 14))
        self.entry_height.insert(0, "1.00")
        self.entry_height.pack(side="right")

        self.lbl_cm = ctk.CTkLabel(
            sensor_frame, 
            text="Current Reading: 100 cm", 
            font=ctk.CTkFont(size=12, weight="bold"), 
            text_color="#f39c12"
        )
        self.lbl_cm.pack(padx=10, pady=(0, 8), anchor="w")

        # Preset Buttons
        btn_preset_frame = ctk.CTkFrame(sensor_frame, fg_color="transparent")
        btn_preset_frame.pack(fill="x", padx=10, pady=(0, 8))
        for val in ["0.5", "1.0", "1.5", "2.0"]:
            ctk.CTkButton(
                btn_preset_frame, 
                text=f"{val}m", 
                width=55, 
                height=24,
                command=lambda v=val: self._set_preset_height(v)
            ).pack(side="left", padx=2)

        # Carriage Mass
        mass_frame = ctk.CTkFrame(self.sidebar)
        mass_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(mass_frame, text="CARRIAGE MASS", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(8, 2), anchor="w")
        m_box_frame = ctk.CTkFrame(mass_frame, fg_color="transparent")
        m_box_frame.pack(fill="x", padx=10, pady=(6, 8))
        ctk.CTkLabel(m_box_frame, text="Mass (kg):", font=ctk.CTkFont(size=13)).pack(side="left")
        self.entry_mass = ctk.CTkEntry(m_box_frame, width=90, font=("Consolas", 14))
        self.entry_mass.insert(0, "3.0")
        self.entry_mass.pack(side="right")

        # Control Buttons
        btn_frame = ctk.CTkFrame(self.sidebar)
        btn_frame.pack(fill="x", padx=15, pady=8)

        self.btn_live = ctk.CTkButton(
            btn_frame, 
            text="START LIVE SENSOR STREAM", 
            fg_color="#27ae60", 
            hover_color="#1e8449",
            command=self.toggle_live_monitoring
        )
        self.btn_live.pack(fill="x", padx=10, pady=6)

        self.btn_file = ctk.CTkButton(
            btn_frame, 
            text="Load TESTLOG1.DAT / CSV", 
            fg_color="#2980b9", 
            hover_color="#1f618d",
            command=self.load_file
        )
        self.btn_file.pack(fill="x", padx=10, pady=(0, 6))

        self.btn_calc = ctk.CTkButton(
            btn_frame, 
            text="Calculate ASTM Parameters", 
            fg_color="#d35400", 
            hover_color="#a04000",
            command=self.calculate_parameters
        )
        self.btn_calc.pack(fill="x", padx=10, pady=(0, 6))

        # Metrics Panel
        self.metric_frame = ctk.CTkFrame(self.sidebar)
        self.metric_frame.pack(fill="x", padx=15, pady=(10, 15))

        self.lbl_e0 = ctk.CTkLabel(self.metric_frame, text="Nominal E0: -- J", anchor="w", font=ctk.CTkFont(size=13))
        self.lbl_e0.pack(fill="x", padx=12, pady=4)

        self.lbl_v0 = ctk.CTkLabel(self.metric_frame, text="Impact Velocity V0: -- m/s", anchor="w", font=ctk.CTkFont(size=13))
        self.lbl_v0.pack(fill="x", padx=12, pady=4)

        self.lbl_fmax = ctk.CTkLabel(self.metric_frame, text="Peak Force F_max: -- kN", anchor="w", font=ctk.CTkFont(size=13))
        self.lbl_fmax.pack(fill="x", padx=12, pady=4)

        self.lbl_dmax = ctk.CTkLabel(self.metric_frame, text="Max Deflection: -- mm", anchor="w", font=ctk.CTkFont(size=13, weight="bold"), text_color="#f1c40f")
        self.lbl_dmax.pack(fill="x", padx=12, pady=4)

    def _build_plot_area(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.tabs = {
            "ASTM D7136 Loop (E-d)": self.tabview.add("ASTM D7136 Loop (E-d)"),
            "Force vs Displacement": self.tabview.add("Force vs Displacement"),
            "Velocity vs Time": self.tabview.add("Velocity vs Time"),
            "Raw Energy vs Time": self.tabview.add("Raw Energy vs Time")
        }

        self.canvases = {}
        self.figures = {}
        self.axes = {}

        for name, tab in self.tabs.items():
            fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1a1a1a")
            ax.set_facecolor("#121212")
            ax.grid(True, linestyle="--", alpha=0.3, color="gray")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("gray")

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, tab)
            toolbar.update()

            self.figures[name] = fig
            self.axes[name] = ax
            self.canvases[name] = canvas

    def _set_preset_height(self, h_str):
        self.entry_height.delete(0, tk.END)
        self.entry_height.insert(0, f"{float(h_str):.2f}")
        self.lbl_cm.configure(text=f"Current Reading: {int(float(h_str)*100)} cm")
        self.calculate_parameters()

    def toggle_live_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.btn_live.configure(text="STOP LIVE SENSOR STREAM", fg_color="#c0392b")
            self.lbl_sensor_status.configure(text="Sensor: STREAMING LIVE", text_color="#f1c40f")
            self.monitor_thread = threading.Thread(target=self._hardware_stream_worker, daemon=True)
            self.monitor_thread.start()
        else:
            self.is_monitoring = False
            self.btn_live.configure(text="START LIVE SENSOR STREAM", fg_color="#27ae60")
            self.lbl_sensor_status.configure(text="Sensor: IDLE / PAUSED", text_color="#95a5a6")

    def _hardware_stream_worker(self):
        sim_height_cm = 120
        direction = 1
        while self.is_monitoring:
            sim_height_cm += direction
            if sim_height_cm >= 185:
                direction = -1
            elif sim_height_cm <= 45:
                direction = 1

            h_m = sim_height_cm / 100.0
            self.after(0, lambda val=h_m, cm=sim_height_cm: self._update_live_height_display(val, cm))
            time.sleep(0.08)

    def _update_live_height_display(self, height_m, height_cm):
        self.entry_height.delete(0, tk.END)
        self.entry_height.insert(0, f"{height_m:.2f}")
        self.lbl_cm.configure(text=f"Current Reading: {height_cm} cm")

    def generate_and_render_sample(self):
        try:
            mass = float(self.entry_mass.get())
            height = float(self.entry_height.get())
        except ValueError:
            mass, height = 3.0, 1.0

        # Create realistic 15 ms impact profile
        t = np.linspace(0.0, 0.015, 300)
        E_nom = mass * 9.80665 * height

        # S-curve absorbed energy profile
        tau = t / 0.015
        E_raw = E_nom * 0.85 * (1.0 / (1.0 + np.exp(-10.0 * (tau - 0.45))))
        E_raw = E_raw - E_raw[0]

        # Contact Force profile
        F_raw = (np.sin(np.pi * tau) * 6500.0) + (np.random.normal(0, 0.02, len(t)) * 100.0)
        F_raw = np.clip(F_raw, 0, None)

        self.time_data = t
        self.energy_data = E_raw
        self.force_data = F_raw

        self.calculate_parameters()

    
    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Data Files", "*.DAT *.CSV *.dat *.csv *.txt"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            # Read all lines to inspect delimiter and strip header text
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_lines = f.readlines()

            parsed_rows = []
            for line in raw_lines:
                clean = line.strip()
                if not clean:
                    continue
                
                # Replace commas and tabs with single spaces
                clean = clean.replace(",", " ").replace("\t", " ")
                parts = clean.split()

                # Extract only lines where the first token is a valid number
                try:
                    num_row = [float(x) for x in parts]
                    if len(num_row) >= 2:
                        parsed_rows.append(num_row)
                except ValueError:
                    # Skip text header lines (e.g., "Report No", "Date", etc.)
                    continue

            if len(parsed_rows) == 0:
                raise ValueError("No numeric data rows found in the selected file.")

            data = np.array(parsed_rows, dtype=float)
            
            # Column 0: Time
            self.time_data = data[:, 0]

            # If 3 or more columns: Time, Force, Energy
            if data.shape[1] >= 3:
                self.force_data = data[:, 1]
                self.energy_data = data[:, 2]
            else:
                # If 2 columns: Time, Energy
                self.energy_data = data[:, 1]
                self.force_data = np.zeros_like(self.time_data)

            # Check if Time column contains timestamps or values in ms
            if np.max(self.time_data) > 2.0:
                self.time_data = self.time_data / 1000.0

            messagebox.showinfo("Telemetry Loaded", f"Successfully extracted {len(self.time_data)} data rows from {os.path.basename(path)}.")
            self.calculate_parameters()

        except Exception as e:
            messagebox.showerror("Parse Error", f"Could not parse file:\n{e}")

    def calculate_parameters(self):
        if len(self.time_data) == 0:
            return

        try:
            mass = float(self.entry_mass.get())
            height = float(self.entry_height.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please verify Mass and Height inputs.")
            return

        t_ms, E, F_kN, v, d_mm, metrics = ImpactKinematicsEngine.process_telemetry(
            self.time_data, self.energy_data, self.force_data, mass, height
        )

        self.lbl_e0.configure(text=f"Nominal E0: {metrics['E0']:.2f} J")
        self.lbl_v0.configure(text=f"Impact Velocity V0: {metrics['v0']:.2f} m/s")
        self.lbl_fmax.configure(text=f"Peak Force F_max: {metrics['F_max']:.2f} kN")
        self.lbl_dmax.configure(text=f"Max Deflection: {metrics['d_max']:.2f} mm")

        # 1. ASTM D7136 Loop
        ax1 = self.axes["ASTM D7136 Loop (E-d)"]
        ax1.clear()
        ax1.plot(d_mm, E, color="#2ecc71", linewidth=2.2, label="ASTM D7136 E-d Loop")
        ax1.set_xlabel("Penetration Deflection d (mm)", color="white", fontsize=11)
        ax1.set_ylabel("Absorbed Energy (J)", color="white", fontsize=11)
        ax1.set_title("ASTM D7136 Structural Loop (E-d)", color="white", fontsize=13, weight="bold")
        ax1.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax1.legend(loc="lower right")
        self.canvases["ASTM D7136 Loop (E-d)"].draw()

        # 2. Force vs Displacement
        ax2 = self.axes["Force vs Displacement"]
        ax2.clear()
        ax2.plot(d_mm, F_kN, color="#e74c3c", linewidth=2.0, label="Contact Force Profile")
        ax2.set_xlabel("Penetration Deflection d (mm)", color="white", fontsize=11)
        ax2.set_ylabel("Impact Force (kN)", color="white", fontsize=11)
        ax2.set_title("Specimen Contact Stiffness (F-d)", color="white", fontsize=13, weight="bold")
        ax2.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax2.legend(loc="upper right")
        self.canvases["Force vs Displacement"].draw()

        # 3. Velocity vs Time
        ax3 = self.axes["Velocity vs Time"]
        ax3.clear()
        ax3.plot(t_ms, v, color="#3498db", linewidth=2.0, label="Striker Deceleration v(t)")
        ax3.set_xlabel("Time (ms)", color="white", fontsize=11)
        ax3.set_ylabel("Velocity (m/s)", color="white", fontsize=11)
        ax3.set_title("Striker Deceleration Velocity Profile", color="white", fontsize=13, weight="bold")
        ax3.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax3.legend(loc="upper right")
        self.canvases["Velocity vs Time"].draw()

        # 4. Raw Energy vs Time
        ax4 = self.axes["Raw Energy vs Time"]
        ax4.clear()
        ax4.plot(t_ms, E, color="#f39c12", linewidth=2.0, label="Absorbed Energy Accumulator")
        ax4.axhline(metrics["E0"], color="red", linestyle=":", linewidth=1.8, label=f"Nominal E0 ({metrics['E0']:.1f} J)")
        ax4.set_xlabel("Time (ms)", color="white", fontsize=11)
        ax4.set_ylabel("Energy (J)", color="white", fontsize=11)
        ax4.set_title("Absorbed Energy vs Time", color="white", fontsize=13, weight="bold")
        ax4.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax4.legend(loc="lower right")
        self.canvases["Raw Energy vs Time"].draw()


if __name__ == "__main__":
    app = GlobeTexApp()
    app.mainloop()