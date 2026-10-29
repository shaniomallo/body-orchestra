"""
learner.py  —  Python port of Learner.js for Dorothy
=====================================================

Quick-start::

    from learner import Learner, LearnerGUI, CLASSIFICATION, REGRESSION
    from dorothy import Dorothy

    dot     = Dorothy()
    learner = Learner(num_inputs=2, num_outputs=1, model_type=CLASSIFICATION)
    gui     = LearnerGUI(learner, dot)

    class MySketch:
        def setup(self):
            gui.setup()

        def draw(self):
            dot.background((30, 30, 30))

            # 1. Feed the current input every frame
            learner.input = [dot.mouse_x, dot.mouse_y]

            # 2. Record labelled examples while recording is on
            learner.new_example([dot.mouse_x, dot.mouse_y], [0])

            # 3. React to the latest prediction — available after toggle_run()
            if learner.output is not None:
                dot.fill((learner.output[0] * 80, 100, 200))
                dot.circle((200, 200), 50)

            gui.draw()   # buttons + status line

    if __name__ == '__main__':
        import __main__
        dot.start_livecode_loop(__main__)

Dataset is automatically saved to <model_name>.csv in the working directory
and reloaded on startup.

Dependencies:  scikit-learn  (pip install scikit-learn)
"""

import csv
import os
import threading
import time

import numpy as np

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputClassifier
except ImportError as _e:
    raise ImportError("scikit-learn is required: pip install scikit-learn") from _e

# ── public constants ──────────────────────────────────────────────────────────
CLASSIFICATION = "classification"
REGRESSION     = "regression"


# ─────────────────────────────────────────────────────────────────────────────
class Learner:
    """
    Interactive ML helper for Dorothy sketches.

    Supports:
      • KNN classification  (discrete outputs, n_neighbours configurable)
      • MLP regression      (continuous outputs, hidden layers configurable)

    Dataset persists as CSV and is reloaded automatically on construction.

    Typical draw-loop pattern::

        # Set the current input each frame so the inference thread can use it
        learner.input = [dot.mouse_x, dot.mouse_y]

        # Read the latest prediction whenever you need it
        if learner.output is not None:
            do_something(learner.output)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_inputs:  int,
        num_outputs: int = 1,
        num_classes: int = None,
        model_type:  str = CLASSIFICATION,
        model_name:  str = "learner",
        data_dir:    str = os.path.dirname(os.path.abspath(__file__)),
    ):
        """
        Args:
            num_inputs:  Number of input features per example.
            num_outputs: Number of outputs per example.
                         Classification → discrete int labels.
                         Regression     → continuous float values.
            model_type:  CLASSIFICATION or REGRESSION.
            model_name:  Used as the CSV filename stem.
            data_dir:    Directory in which the CSV is stored (default: cwd).
        """
        if model_type not in (CLASSIFICATION, REGRESSION):
            raise ValueError(
                f"model_type must be CLASSIFICATION or REGRESSION, got {model_type!r}"
            )

        self.num_inputs  = num_inputs
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        self.model_type  = model_type
        self.model_name  = model_name
        self.data_path   = os.path.join(data_dir, f"{model_name}.csv")

        self.recording: bool = False
        self.running:   bool = False
        self.trained:   bool = False

        self.rec_ctr = 0

        # Set this every frame with the current sensor/mouse/audio values.
        # The inference thread reads it automatically.
        self.input = None

        # The inference thread writes predictions here.
        # Read this in your draw loop whenever you need the latest output.
        self.output = None

        self._model      = None
        self._run_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()   # protects self.dataset

        self.dataset: list[tuple[list[float], list[float], int]] = []

        self._load_csv()
        self._build_model()

    # ------------------------------------------------------------------
    # Internal – model construction
    # ------------------------------------------------------------------

    def _build_model(self):
        if self.model_type == CLASSIFICATION:
            base = KNeighborsClassifier(n_neighbors=3)
            self._model = MultiOutputClassifier(base) if self.num_outputs > 1 else base
        else:
            self._model = MLPRegressor(
                hidden_layer_sizes=(16, 16),
                max_iter=2000,
                learning_rate_init=0.001,
                random_state=42,
            )

    # ------------------------------------------------------------------
    # Internal – CSV persistence
    # ------------------------------------------------------------------

    def _load_csv(self):
        print(f"[Learner] looking for CSV at: {os.path.abspath(self.data_path)}")
        if not os.path.exists(self.data_path):
            return
        loaded = 0
        with open(self.data_path, newline="") as fh:
            for row in csv.reader(fh):
                n = self.num_inputs + self.num_outputs
                if len(row) == n:
                    tag = 0   # legacy CSV — no session column
                elif len(row) == n + 1:
                    tag = int(float(row[-1]))
                else:
                    continue
                inp = [float(v) for v in row[: self.num_inputs]]
                out = [float(v) for v in row[self.num_inputs : n]]
                self.dataset.append((inp, out, tag))
                loaded += 1
        if loaded:
            print(
                f"[Learner:{self.model_name}] Loaded {loaded} examples"
                f" from {self.data_path}"
            )

    def _append_csv(self, inp: list, out: list, tag: int):
        with open(self.data_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [f"{v:.6g}" for v in inp] + [f"{v:.6g}" for v in out] + [tag]
            )

    # ------------------------------------------------------------------
    # Public – data collection
    # ------------------------------------------------------------------

    def add_example(self, input, output):
        """
        Unconditionally add an input/output example and persist it to CSV.

        Args:
            input:  Sequence of ``num_inputs`` numbers.
            output: Sequence of ``num_outputs`` numbers.
        """
        inp = [float(v) for v in input]
        out = [float(v) for v in output]
        tag = self.rec_ctr
        with self._lock:
            self.dataset.append((inp, out, tag))
        self._append_csv(inp, out, tag)

    def new_example(self, input, output):
        """
        Call every frame with the current input/output pair.
        Always updates ``self.input`` (used by the inference thread).
        Only saves to the dataset when ``self.recording`` is True.
        """
        self.input = input
        if self.recording:
            self.add_example(input, output)

    def toggle_record(self) -> bool:
        """Flip the recording flag.  Returns the new state."""
        
        self.recording = not self.recording
        state = "ON" if self.recording else "OFF"
        print(
            f"[Learner:{self.model_name}] Recording {state}"
            f"  ({len(self.dataset)} examples so far)"
        )
        if self.recording:
            self.rec_ctr += 1
        return self.recording

    def num_examples(self) -> int:
        return len(self.dataset)

    def clear(self):
        """Delete all training data (memory + CSV) and reset the model."""
        with self._lock:
            self.dataset.clear()
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        self._build_model()
        self.trained = False
        self.output  = None
        print(f"[Learner:{self.model_name}] Cleared all data")

    # ------------------------------------------------------------------
    # Public – hyperparameter configuration
    # ------------------------------------------------------------------

    def set_model_options(
        self,
        k:             int   = 3,
        hidden_layers: tuple = (16, 16),
        epochs:        int   = 2000,
    ):
        """
        Reconfigure model hyperparameters.  Call *before* train().

        Args:
            k:             (classification) number of nearest neighbours.
            hidden_layers: (regression) hidden layer sizes, e.g. ``(32, 32)``.
            epochs:        (regression) max training iterations.
        """
        if self.model_type == CLASSIFICATION:
            base = KNeighborsClassifier(n_neighbors=k)
            self._model = MultiOutputClassifier(base) if self.num_outputs > 1 else base
        else:
            self._model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layers),
                max_iter=epochs,
                random_state=42,
            )
        self.trained = False

    # ------------------------------------------------------------------
    # Public – training
    # ------------------------------------------------------------------

    def train(self) -> bool:
        """
        Train the model on all collected examples.

        Returns True on success, False if there is no data.
        """
        with self._lock:
            data = list(self.dataset)

        if not data:
            print(f"[Learner:{self.model_name}] No training data — add examples first")
            return False

        X = [d[0] for d in data]

        if self.model_type == CLASSIFICATION:
            if self.num_outputs == 1:
                y = [int(d[1][0]) for d in data]
            else:
                y = [[int(v) for v in d[1]] for d in data]
        else:
            if self.num_outputs == 1:
                y = [d[1][0] for d in data]
            else:
                y = [d[1] for d in data]
        # d[2] is the recording-session tag — used for bookkeeping, not training

        self._model.fit(X, y)
        self.trained = True
        print(f"[Learner:{self.model_name}] Trained on {len(X)} examples")
        return True

    # ------------------------------------------------------------------
    # Public – inference
    # ------------------------------------------------------------------

    def predict(self, input) -> list | None:
        """
        Run a single synchronous prediction.

        Returns a list of ``num_outputs`` values, or ``None`` if untrained.
        """
        if not self.trained:
            return None
        inp = np.array(input, dtype=float).reshape(1, -1)
        result = self._model.predict(inp)

        if self.model_type == CLASSIFICATION:
            if self.num_outputs == 1:
                return [int(result[0])]
            return [int(v) for v in result[0]]
        else:
            if self.num_outputs == 1:
                return [float(result[0])]
            return [float(v) for v in result[0]]

    def toggle_run(self, interval: float = 0.05):
        """
        Start or stop the background inference loop.

        While running the loop reads ``self.input``, calls ``predict()``,
        and stores the result in ``self.output`` — all on a daemon thread
        so Dorothy's draw loop is never blocked.

        Read ``learner.output`` in your draw function to use predictions.

        Args:
            interval: Seconds between predictions (default 0.05 = 20 Hz).
        """
        if self.running:
            self._stop_event.set()
            print(f"[Learner:{self.model_name}] Stopping inference…")
        else:
            if not self.trained:
                print(f"[Learner:{self.model_name}] Train the model before running")
                return
            self.running = True
            self._stop_event.clear()
            self._run_thread = threading.Thread(
                target=self._run_loop,
                args=(interval,),
                daemon=True,
                name=f"LearnerInference-{self.model_name}",
            )
            self._run_thread.start()
            print(
                f"[Learner:{self.model_name}] Inference started"
                f" ({1 / interval:.0f} Hz)"
            )

    def _run_loop(self, interval):
        while not self._stop_event.is_set():
            try:
                inp = self.input
                if inp is not None:
                    out = self.predict(inp)
                    if out is not None:
                        self.output = out
            except Exception as exc:
                print(f"[Learner:{self.model_name}] Inference error: {exc}")
            time.sleep(interval)
        self.running = False
        print(f"[Learner:{self.model_name}] Inference stopped")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"<Learner name={self.model_name!r} type={self.model_type} "
            f"inputs={self.num_inputs} outputs={self.num_outputs} "
            f"examples={len(self.dataset)} trained={self.trained}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
class LearnerGUI:
    """
    Minimal Dorothy button GUI for a :class:`Learner`.

    Creates three toggle buttons — **Record**, **Train**, **Run** — plus a
    status line.  Buttons light up green when active.

    Usage::

        gui = LearnerGUI(learner, dot)

        class MySketch:
            def setup(self):
                gui.setup()

            def draw(self):
                learner.input = [dot.mouse_x, dot.mouse_y]
                if learner.output is not None:
                    do_something(learner.output)
                gui.draw()

    If you already call ``dot.update_buttons()`` / ``dot.draw_buttons()``
    elsewhere, pass ``update=False`` to ``gui.draw()``.
    """

    # Button colours — integer RGBA 4-tuples (0–255)
    _COL_IDLE   = (56,  56,  56,  255)
    _COL_HOVER  = (87,  87,  87,  255)
    _COL_ACTIVE = (38,  158, 84,  255)   # green when toggled on
    _COL_TEXT   = (255, 255, 255, 255)

    _COLS_PER_ROW = 1   # sliders per row for regression

    def __init__(
        self,
        learner:     "Learner",
        x:     int = 10,
        y:     int = 10,
        btn_w: int = 130,
        btn_h: int = 38,
        gap:   int = 8,
    ):
        self.learner       = learner
        self.x, self.y     = x, y
        self.btn_w         = btn_w
        self.btn_h         = btn_h
        self.gap           = gap
        self._btn_record   = None
        self._btn_train    = None
        self._btn_run      = None
        self._class_btns   = []
        self._sliders      = []   # regression only
        # Currently selected class label (classification) — use in new_example()
        self.label         = [0]

    # ------------------------------------------------------------------

    def setup(self, dot):
        """Create Dorothy buttons (and sliders for regression).
        Call once inside your sketch's setup()."""
        x, y = self.x, self.y
        w, h = self.btn_w, self.btn_h
        g    = self.gap

        self._btn_record = dot.create_button(
            x, y,
            w, h,
            text="Record",
            id="learner_record",
            on_release=self._on_record,
        )
        self._btn_train = dot.create_button(
            x, y + (h + g),
            w, h,
            text="Train",
            id="learner_train",
            on_release=self._on_train,
        )
        self._btn_run = dot.create_button(
            x, y + 2 * (h + g),
            w, h,
            text="Run",
            id="learner_run",
            on_release=self._on_run,
        )

        row4_y = 10 + y + 3 * (h + g)

        if self.learner.model_type == CLASSIFICATION:
            # One button per class in a row
            n      = self.learner.num_classes
            cell_w = max(1, (w - g * (n - 1)) // n)
            for i in range(n):
                btn = dot.create_button(
                    x + i * (cell_w + g), row4_y,
                    cell_w, h,
                    text=str(i),
                    id=f"learner_class_{i}",
                    on_release=self._make_class_handler(i),
                )
                self._class_btns.append(btn)
        else:
            # One slider per output, laid out in rows of _COLS_PER_ROW
            n_out  = self.learner.num_outputs
            cols   = self._COLS_PER_ROW
            cell_w = max(1, (w - g * (cols - 1)) // cols)
            sldr_h = h
            for i in range(n_out):
                col   = i % cols
                row   = i // cols
                sx    = x + col * (cell_w + g)
                sy    = row4_y + row * (sldr_h + g + 28)  # +14 for label above
                sldr  = dot.create_slider(
                    sx, sy,
                    cell_w, sldr_h,
                    min_val=0.0, max_val=1.0, value=0.5,
                    label=f"out {i}",
                    id=f"learner_out_{i}",
                )
                sldr.set_style(label_color = dot.white)
                self._sliders.append(sldr)

        self._sync_styles()

    def draw(self, dot, update: bool = True):
        """
        Update button/slider interactions, render them, and show status text.
        Call every frame inside your sketch's draw().

        Args:
            update: If True (default) calls dot.update_buttons/sliders and
                    dot.draw_buttons/sliders. Pass False if you manage those yourself.
        """
        if update:
            dot.update_buttons()
            dot.draw_buttons()
            if self._sliders:
                dot.update_sliders()
                dot.draw_sliders()
        self._sync_styles()

        # For regression, keep label in sync with slider values
        if self.learner.model_type == REGRESSION and not self.learner.running:
            self.label = [s.value for s in self._sliders]

        # When running, push output back to sliders so they reflect predictions
        if self.learner.running and self.learner.output is not None:
            for i, s in enumerate(self._sliders):
                if i < len(self.learner.output):
                    s.set_value(self.learner.output[i])

        # Status text — sits below buttons+sliders
        n     = self.learner.num_examples()
        state = []
        if self.learner.recording: state.append("recording")
        if self.learner.trained:   state.append("trained")
        if self.learner.running:   state.append("running")
        suffix = "  [" + ", ".join(state) + "]" if state else ""

        n_out    = self.learner.num_outputs
        n_rows   = 1 if self.learner.model_type == CLASSIFICATION else \
                   ((n_out - 1) // self._COLS_PER_ROW + 1)
        sldr_row_h = self.btn_h + self.gap + 14
        text_y = 15 + self.y + 3 * (self.btn_h + self.gap) + n_rows * sldr_row_h + 6
        dot.fill((220, 220, 220))
        dot.text(f"Examples: {n}{suffix}", self.x, text_y, 13)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _make_class_handler(self, class_index):
        def handler(_btn):
            self.label = [class_index]
            self._sync_styles()
        return handler

    def _on_record(self, _btn):
        self.learner.toggle_record()
        self._sync_styles()

    def _on_train(self, _btn):
        self.learner.train()
        self._sync_styles()

    def _on_run(self, _btn):
        self.learner.toggle_run()
        # Brief delay so the thread has time to confirm start/stop
        threading.Timer(0.06, self._sync_styles).start()

    # ------------------------------------------------------------------
    # Style sync
    # ------------------------------------------------------------------

    def _sync_styles(self):
        l = self.learner
        self._apply_style(self._btn_record, l.recording)
        self._apply_style(self._btn_train,  l.trained)
        self._apply_style(self._btn_run,    l.running)
        # When running, highlight the predicted class; otherwise the selected label
        if l.running and l.output is not None:
            active_class = l.output[0]
        else:
            active_class = self.label[0]
        for i, btn in enumerate(self._class_btns):
            self._apply_style(btn, i == active_class)

    def _apply_style(self, btn, active: bool):
        if btn is None:
            return
        idle  = self._COL_ACTIVE if active else self._COL_IDLE
        hover = self._COL_ACTIVE if active else self._COL_HOVER
        btn.set_style(idle_color=idle, hover_color=hover, text_color=self._COL_TEXT)
