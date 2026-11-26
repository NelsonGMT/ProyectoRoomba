#!/usr/bin/env python3
"""Interfaz Tkinter para detectar papel/plástico y registrar tracks en CSV."""
from __future__ import annotations
import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent

def abrir_camara(indice, csi, ancho=640, alto=480):
    if csi:
        from picamera2 import Picamera2
        cam = Picamera2(camera_num=indice)
        cam.configure(cam.create_video_configuration(
            main={"size": (ancho, alto), "format": "RGB888"}, buffer_count=2))
        cam.start()
        time.sleep(1)
        return ("csi", cam)
    cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo abrir la cámara USB índice {indice}.")
    return ("usb", cap)

def leer_camara(cam):
    tipo, obj = cam
    if tipo == "csi":
        return obj.capture_array()
    ok, frame = obj.read()
    if not ok or frame is None:
        raise RuntimeError("La cámara no entregó un fotograma.")
    return frame

def cerrar_camara(cam):
    if cam:
        tipo, obj = cam
        if tipo == "csi":
            obj.stop()
            obj.close()
        else:
            obj.release()

class YoloCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO | Detección de papel y plástico")
        self.root.geometry("1100x700")
        self.messages = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.worker = None
        self.closing = False
        self.photo = None
        self.events = {}
        self.seen_ids = set()
        self.current_counts = {}
        self.camera_var = tk.StringVar(value="0")
        self.csi_var = tk.BooleanVar(value=True)
        self.model_var = tk.StringVar(value=str(APP_DIR / "modelo_basura_ncnn"))
        self.conf_var = tk.DoubleVar(value=0.35)
        self.status_var = tk.StringVar(value="Detenido")
        self.fps_var = tk.StringVar(value="FPS: —")
        self.unique_var = tk.StringVar(value="IDs seguidos: 0")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.root.after(50, self.poll)

    def _build(self):
        settings = ttk.LabelFrame(self.root, text="Configuración", padding=8)
        settings.pack(fill="x", padx=10, pady=8)
        ttk.Label(settings, text="Índice cámara").grid(row=0, column=0, padx=4)
        ttk.Entry(settings, textvariable=self.camera_var, width=5).grid(row=0, column=1, padx=4)
        ttk.Checkbutton(settings, text="Usar CSI (desmarcar para USB)",
                        variable=self.csi_var).grid(row=0, column=2, padx=8)
        ttk.Label(settings, text="Modelo").grid(row=0, column=3, padx=4)
        ttk.Entry(settings, textvariable=self.model_var, width=35).grid(row=0, column=4, padx=4)
        ttk.Button(settings, text="Examinar", command=self.choose_model).grid(row=0, column=5, padx=4)
        ttk.Label(settings, text="Confianza").grid(row=1, column=0, sticky="w", padx=4, pady=8)
        ttk.Scale(settings, from_=0.15, to=0.85, variable=self.conf_var,
                  orient="horizontal", length=220).grid(row=1, column=1, columnspan=2, sticky="w")
        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        video = ttk.LabelFrame(body, text="Cámara en vivo", padding=6)
        video.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        video.rowconfigure(0, weight=1)
        video.columnconfigure(0, weight=1)
        self.video_label = ttk.Label(video, text="Pulsa Iniciar cámara", anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        side = ttk.LabelFrame(body, text="Estado y detecciones", padding=8)
        side.grid(row=0, column=1, sticky="nsew")
        ttk.Label(side, textvariable=self.status_var).pack(anchor="w", pady=3)
        ttk.Label(side, textvariable=self.fps_var).pack(anchor="w", pady=3)
        ttk.Label(side, textvariable=self.unique_var).pack(anchor="w", pady=3)
        self.text = tk.Text(side, width=25, height=20, state="disabled")
        self.text.pack(fill="both", expand=True, pady=8)
        buttons = ttk.Frame(self.root, padding=8)
        buttons.pack(fill="x")
        self.start_btn = ttk.Button(buttons, text="Iniciar cámara", command=self.start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=4)
        self.stop_btn = ttk.Button(buttons, text="Detener", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(buttons, text="Exportar CSV", command=self.export_csv).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(buttons, text="Salir", command=self.on_exit).pack(side="left", expand=True, fill="x", padx=4)

    def choose_model(self):
        path = filedialog.askopenfilename(title="Seleccionar modelo .pt",
             filetypes=[("Modelo YOLO", "*.pt"), ("Todos", "*.*")])
        if path:
            self.model_var.set(path)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        path = Path(self.model_var.get().strip()).expanduser()
        if not path.exists():
            messagebox.showerror("Modelo no encontrado", f"No existe:\n{path}")
            return
        try:
            index = int(self.camera_var.get())
            if index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Cámara", "El índice debe ser un entero no negativo.")
            return
        self.events.clear()
        self.seen_ids.clear()
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Iniciando...")
        self.worker = threading.Thread(
            target=self.worker_loop,
            args=(str(path), index, self.csi_var.get(), float(self.conf_var.get())),
            daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Deteniendo...")

    def worker_loop(self, model_path, index, csi, conf):
        cam = None
        try:
            model = YOLO(model_path, task="detect")
            cam = abrir_camara(index, csi)
            self.post("status", f"Cámara activa | clases: {model.names}")
            last = time.perf_counter()
            while not self.stop_event.is_set():
                frame = leer_camara(cam)
                result = model.track(source=frame, persist=True, tracker="bytetrack.yaml",
                                     imgsz=416, conf=conf, device="cpu",
                                     verbose=False, max_det=30)[0]
                annotated = result.plot()
                counts = {}
                observed = []
                if result.boxes is not None and len(result.boxes):
                    classes = result.boxes.cls.int().cpu().tolist()
                    ids = (result.boxes.id.int().cpu().tolist()
                           if result.boxes.id is not None else [None] * len(classes))
                    for cls, track_id in zip(classes, ids):
                        label = result.names[int(cls)]
                        counts[label] = counts.get(label, 0) + 1
                        if track_id is not None:
                            observed.append((int(track_id), label))
                now = time.perf_counter()
                fps = 1 / max(now-last, 1e-6)
                last = now
                self.post("frame", (annotated, counts, observed, fps))
        except Exception as e:
            self.post("error", str(e))
        finally:
            cerrar_camara(cam)
            self.post("finished", None)

    def post(self, kind, payload):
        # Cola acotada: descarta fotogramas antiguos, conserva la interfaz receptiva.
        try:
            self.messages.put_nowait((kind, payload))
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait((kind, payload))
            except queue.Full:
                pass

    def poll(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Error de cámara / YOLO", payload)
                elif kind == "frame":
                    frame, counts, observed, fps = payload
                    h, w = frame.shape[:2]
                    maxw = max(320, self.video_label.winfo_width()-10)
                    maxh = max(240, self.video_label.winfo_height()-10)
                    scale = min(maxw/w, maxh/h, 1.0)
                    if scale < 1:
                        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                    self.video_label.configure(image=self.photo, text="")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for track_id, label in observed:
                        self.seen_ids.add(track_id)
                        if track_id not in self.events:
                            self.events[track_id] = {"id": track_id, "clase": label,
                                "primera_vez": timestamp, "ultima_vez": timestamp}
                        else:
                            self.events[track_id]["ultima_vez"] = timestamp
                    self.fps_var.set(f"FPS: {fps:.1f}")
                    self.unique_var.set(f"IDs seguidos: {len(self.seen_ids)} (estimación)")
                    body = "OBJETOS EN ESTE CUADRO\n"
                    body += "\n".join(f"• {k}: {v}" for k,v in sorted(counts.items())) or "Sin detecciones"
                    body += "\n\nNota: un ID puede cambiar si se pierde el objeto."
                    self.text.configure(state="normal")
                    self.text.delete("1.0", "end")
                    self.text.insert("1.0", body)
                    self.text.configure(state="disabled")
                elif kind == "finished":
                    pass
        except queue.Empty:
            pass
        if self.worker and not self.worker.is_alive():
            self.worker.join(timeout=0)
            self.worker = None
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            if self.status_var.get() != "Error":
                self.status_var.set("Detenido")
        if self.closing and self.worker is None:
            self.root.destroy()
            return
        self.root.after(50, self.poll)

    def export_csv(self):
        if not self.events:
            messagebox.showinfo("Sin datos", "Todavía no hay objetos seguidos con ID.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
            initialfile="detecciones_basura.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["id","clase","primera_vez","ultima_vez"])
            writer.writeheader()
            writer.writerows(sorted(self.events.values(), key=lambda x: x["id"]))
        messagebox.showinfo("CSV guardado", path)

    def on_exit(self):
        self.closing = True
        self.stop_event.set()
        if self.worker is None:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    YoloCamApp(root)
    root.mainloop()
