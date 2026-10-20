from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


MODEL_PATH = Path(__file__).with_name("my_cnn_model.keras")
CANVAS_SIZE = 320
INK_WIDTH = 20


class DigitClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CNN Digit Classifier")
        self.root.resizable(False, False)

        self.model: Any | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.last_point: tuple[int, int] | None = None
        self.drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawing_pen = ImageDraw.Draw(self.drawing)

        self._build_ui()
        self.root.after(100, self._process_events)
        self._set_status("Loading my_cnn_model.keras...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=20)
        frame.grid()

        ttk.Label(
            frame, text="Draw a digit", font=("Segoe UI", 20, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(0, 4))
        ttk.Label(
            frame, text="Draw one number, then select Predict."
        ).grid(row=1, column=0, columnspan=2, pady=(0, 14))

        self.canvas = tk.Canvas(
            frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="black",
            highlightthickness=1,
            highlightbackground="#777777",
            cursor="crosshair",
        )
        self.canvas.grid(row=2, column=0, columnspan=2)
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        self.predict_button = ttk.Button(
            frame, text="Predict", command=self._predict, state=tk.DISABLED
        )
        self.predict_button.grid(row=3, column=0, sticky="ew", pady=(14, 0), padx=(0, 6))
        ttk.Button(frame, text="Clear", command=self._clear).grid(
            row=3, column=1, sticky="ew", pady=(14, 0), padx=(6, 0)
        )

        self.result_label = ttk.Label(
            frame, text="Prediction: —", font=("Segoe UI", 18, "bold")
        )
        self.result_label.grid(row=4, column=0, columnspan=2, pady=(16, 2))
        self.alternatives_label = ttk.Label(frame, text="")
        self.alternatives_label.grid(row=5, column=0, columnspan=2)
        self.status_label = ttk.Label(frame, text="", wraplength=320)
        self.status_label.grid(row=6, column=0, columnspan=2, pady=(8, 0))

    def _start_stroke(self, event: tk.Event) -> None:
        point = (event.x, event.y)
        self.last_point = point
        radius = INK_WIDTH // 2
        self.drawing_pen.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=255,
        )
        self.canvas.create_oval(
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
            fill="white",
            outline="white",
        )

    def _continue_stroke(self, event: tk.Event) -> None:
        if self.last_point is None:
            self._start_stroke(event)
            return

        point = (event.x, event.y)
        self.drawing_pen.line(
            (self.last_point, point), fill=255, width=INK_WIDTH
        )
        radius = INK_WIDTH // 2
        self.drawing_pen.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=255,
        )
        self.canvas.create_line(
            *self.last_point,
            *point,
            fill="white",
            width=INK_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.canvas.create_oval(
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
            fill="white",
            outline="white",
        )
        self.last_point = point

    def _end_stroke(self, _event: tk.Event) -> None:
        self.last_point = None

    def _clear(self) -> None:
        self.canvas.delete("all")
        self.drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawing_pen = ImageDraw.Draw(self.drawing)
        self.result_label.configure(text="Prediction: —")
        self.alternatives_label.configure(text="")
        self._set_status("")

    def _prepare_input(self) -> np.ndarray | None:
        bounds = self.drawing.getbbox()
        if bounds is None:
            return None

        digit = self.drawing.crop(bounds)
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

        image = Image.new("L", (28, 28), color=0)
        position = ((28 - digit.width) // 2, (28 - digit.height) // 2)
        image.paste(digit, position)
        return np.asarray(image, dtype=np.float32)[np.newaxis, :, :, np.newaxis] / 255.0

    def _load_model(self) -> None:
        try:
            from tensorflow.keras.models import load_model

            model = load_model(MODEL_PATH, compile=False)
            if tuple(model.input_shape[1:]) != (28, 28, 1):
                raise ValueError(
                    f"Expected model input shape (28, 28, 1), got {model.input_shape[1:]}."
                )
            self.events.put(("loaded", model))
        except Exception as error:
            self.events.put(("load_error", error))

    def _predict(self) -> None:
        if self.model is None:
            return
        model_input = self._prepare_input()
        if model_input is None:
            self._set_status("Draw a digit before predicting.")
            return

        self.predict_button.configure(state=tk.DISABLED)
        self._set_status("Classifying...")
        threading.Thread(
            target=self._run_prediction, args=(model_input,), daemon=True
        ).start()

    def _run_prediction(self, model_input: np.ndarray) -> None:
        try:
            probabilities = np.asarray(self.model.predict(model_input, verbose=0))
            if probabilities.shape != (1, 10):
                raise ValueError(
                    f"Expected 10 class scores, got output shape {probabilities.shape}."
                )
            self.events.put(("predicted", probabilities[0]))
        except Exception as error:
            self.events.put(("prediction_error", error))

    def _process_events(self) -> None:
        try:
            while True:
                event, data = self.events.get_nowait()
                if event == "loaded":
                    self.model = data
                    self.predict_button.configure(state=tk.NORMAL)
                    self._set_status("Ready. Draw a digit from 0 to 9.")
                elif event == "load_error":
                    self._set_status(f"Could not load the model: {data}")
                    messagebox.showerror("Model loading failed", str(data), parent=self.root)
                elif event == "predicted":
                    scores = data
                    ranked = np.argsort(scores)[::-1]
                    best = int(ranked[0])
                    self.result_label.configure(
                        text=f"Prediction: {best}  ({scores[best]:.1%})"
                    )
                    alternatives = ", ".join(
                        f"{int(index)}: {scores[index]:.1%}" for index in ranked[:3]
                    )
                    self.alternatives_label.configure(
                        text=f"Top results: {alternatives}"
                    )
                    self._set_status("")
                    self.predict_button.configure(state=tk.NORMAL)
                elif event == "prediction_error":
                    self._set_status(f"Prediction failed: {data}")
                    messagebox.showerror("Prediction failed", str(data), parent=self.root)
                    self.predict_button.configure(state=tk.NORMAL)
        except queue.Empty:
            pass

        self.root.after(100, self._process_events)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)


def main() -> None:
    root = tk.Tk()
    DigitClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
