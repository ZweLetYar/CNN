from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


MODEL_PATH = Path(__file__).with_name("cnn_model.pth")
CANVAS_SIZE = 320
INK_WIDTH = 10
MODEL_IMAGE_SIZE = 128
DIGIT_MAX_SIZE = 92
DRAWING_SCALE = 3


def _build_model() -> Any:
    """Create the PyTorch CNN architecture used by cnn_model.pth."""
    from torch import nn

    class CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x: Any) -> Any:
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.flatten(start_dim=1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    return CNN()


class DigitClassifierApp:
    """Window for drawing a digit and asking the CNN to classify it."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CNN Digit Classifier")
        self.root.resizable(False, False)

        self.model: Any | None = None
        # Worker threads send results here so only Tkinter's main thread
        # updates the window.
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.last_point: tuple[int, int] | None = None
        # Keep a grayscale copy of the canvas for the model input.
        self.drawing = Image.new(
            "L",
            (CANVAS_SIZE * DRAWING_SCALE, CANVAS_SIZE * DRAWING_SCALE),
            color=255,
        )
        self.drawing_pen = ImageDraw.Draw(self.drawing)

        self._build_ui()
        self.root.after(100, self._process_events)
        self._set_status("Loading cnn_model.pth...")
        # Loading a model can take time, so do it without freezing the window.
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self) -> None:
        """Create the drawing canvas, buttons, and result labels."""
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
            background="white",
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
        self._draw_dot(point)

    def _continue_stroke(self, event: tk.Event) -> None:
        if self.last_point is None:
            self._start_stroke(event)
            return

        point = (event.x, event.y)
        self._draw_line(self.last_point, point)
        self._draw_dot(point)
        self.last_point = point

    def _draw_dot(self, point: tuple[int, int]) -> None:
        """Draw one round stroke end on both the visible and saved canvases."""
        radius = INK_WIDTH // 2
        left = point[0] - radius
        top = point[1] - radius
        right = point[0] + radius
        bottom = point[1] + radius
        scale = DRAWING_SCALE
        self.drawing_pen.ellipse(
            (left * scale, top * scale, right * scale, bottom * scale),
            fill=0,
        )
        self.canvas.create_oval(left, top, right, bottom, fill="black", outline="black")

    def _draw_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Connect two points on both the visible and saved canvases."""
        scale = DRAWING_SCALE
        self.drawing_pen.line(
            (
                start[0] * scale,
                start[1] * scale,
                end[0] * scale,
                end[1] * scale,
            ),
            fill=0,
            width=INK_WIDTH * scale,
        )
        self.canvas.create_line(
            start[0],
            start[1],
            end[0],
            end[1],
            fill="black",
            width=INK_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
        )

    def _end_stroke(self, _event: tk.Event) -> None:
        self.last_point = None

    def _clear(self) -> None:
        """Clear the visible drawing and its saved image."""
        self.canvas.delete("all")
        self.drawing = Image.new(
            "L",
            (CANVAS_SIZE * DRAWING_SCALE, CANVAS_SIZE * DRAWING_SCALE),
            color=255,
        )
        self.drawing_pen = ImageDraw.Draw(self.drawing)
        self.result_label.configure(text="Prediction: —")
        self.alternatives_label.configure(text="")
        self._set_status("")

    def _prepare_input(self) -> np.ndarray | None:
        """Use OpenCV to resize and center the drawing for the CNN."""
        source = np.asarray(self.drawing, dtype=np.uint8)
        ink = 255 - source
        points = cv2.findNonZero(ink)
        if points is None:
            return None

        x, y, width, height = cv2.boundingRect(points)
        digit = source[y : y + height, x : x + width]

        # Preserve the digit's aspect ratio while making small and large drawings
        # occupy a consistent area in the model's 128x128 input.
        scale = DIGIT_MAX_SIZE / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        digit = cv2.resize(
            digit,
            (resized_width, resized_height),
            interpolation=interpolation,
        )

        image = np.full(
            (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), 255, dtype=np.uint8
        )
        left = (MODEL_IMAGE_SIZE - resized_width) // 2
        top = (MODEL_IMAGE_SIZE - resized_height) // 2
        image[top : top + resized_height, left : left + resized_width] = digit

        moments = cv2.moments(255 - image, binaryImage=False)
        if moments["m00"] > 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            target_center = (MODEL_IMAGE_SIZE - 1) / 2
            offset_x = round(target_center - center_x)
            offset_y = round(target_center - center_y)
            offset_x = min(max(offset_x, -left), MODEL_IMAGE_SIZE - left - resized_width)
            offset_y = min(max(offset_y, -top), MODEL_IMAGE_SIZE - top - resized_height)
            transform = np.float32(
                [[1, 0, offset_x], [0, 1, offset_y]]
            )
            image = cv2.warpAffine(
                image,
                transform,
                (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
        else:
            return None

        # PyTorch expects normalized grayscale data in NCHW order.
        return image.astype(np.float32)[np.newaxis, np.newaxis, :, :] / 255.0

    def _load_model(self) -> None:
        """Load and validate the model in a worker thread."""
        try:
            import torch

            model = _build_model()
            state_dict = torch.load(
                MODEL_PATH, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
            model.eval()
            self.events.put(("loaded", model))
        except Exception as error:
            self.events.put(("load_error", error))

    def _predict(self) -> None:
        """Prepare the drawing and start classification when Predict is clicked."""
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
        """Run the model in a worker thread and send its scores to the window."""
        try:
            import torch

            with torch.inference_mode():
                logits = self.model(torch.from_numpy(model_input))
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            if probabilities.shape != (1, 10):
                raise ValueError(
                    f"Expected 10 class scores, got output shape {probabilities.shape}."
                )
            self.events.put(("predicted", probabilities[0]))
        except Exception as error:
            self.events.put(("prediction_error", error))

    def _process_events(self) -> None:
        """Apply worker results on Tkinter's main thread and check again shortly."""
        try:
            while True:
                event_name, event_data = self.events.get_nowait()
                if event_name == "loaded":
                    self.model = event_data
                    self.predict_button.configure(state=tk.NORMAL)
                    self._set_status("Ready. Draw a digit from 0 to 9.")
                elif event_name == "load_error":
                    self._set_status(f"Could not load the model: {event_data}")
                    messagebox.showerror(
                        "Model loading failed", str(event_data), parent=self.root
                    )
                elif event_name == "predicted":
                    scores = event_data
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
                elif event_name == "prediction_error":
                    self._set_status(f"Prediction failed: {event_data}")
                    messagebox.showerror(
                        "Prediction failed", str(event_data), parent=self.root
                    )
                    self.predict_button.configure(state=tk.NORMAL)
        except queue.Empty:
            # There are no more worker results waiting right now.
            pass

        self.root.after(100, self._process_events)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)


def main() -> None:
    """Create the Tkinter window and start its event loop."""
    root = tk.Tk()
    DigitClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
