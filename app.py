#!/usr/bin/env python3
"""
Pro Vector DB: Cascading B-Spline Transforms, GPU Acceleration & Projection Geometry
PySide6 application for high-dimensional vector database operations with
B-spline scoring, dimensionality reduction visualization, and KNN/range queries.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import sys
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline, make_lsq_spline, SmoothBivariateSpline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QSplitter,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea, QFormLayout,
    QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QStatusBar, QAbstractItemView, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont, QColor, QClipboard
import json
import os
import csv
import configparser
import warnings
import datetime
import logging

# ── Terminal Logging Setup ───────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(message)s"
DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, stream=sys.stdout)
logger = logging.getLogger("VectorDB")

try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("CuPy imported successfully — GPU acceleration available")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy not found — GPU acceleration disabled (install: conda install -c conda-forge cupy=12.6)")

try:
    import pandas as pd
    HAS_PANDAS = True
    logger.debug("Pandas imported successfully")
except ImportError:
    HAS_PANDAS = False
    logger.debug("Pandas not found — parallel coordinates tab disabled")

try:
    import umap
    HAS_UMAP = True
    logger.debug("UMAP imported successfully")
except ImportError:
    HAS_UMAP = False
    logger.debug("UMAP not found — UMAP reduction disabled")

warnings.filterwarnings('ignore')

# ── Matplotlib Global Style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#181825",
    "axes.edgecolor": "#45475a",
    "axes.labelcolor": "#cdd6f4",
    "text.color": "#cdd6f4",
    "xtick.color": "#a6adc8",
    "ytick.color": "#a6adc8",
    "grid.color": "#313244",
    "font.family": "monospace"
})


# ── Log Signal Bridge (routes logger → Qt UI) ───────────────────────────────
class LogSignalBridge(QObject):
    """QObject bridge so a logging.Handler can emit Qt signals."""
    message = Signal(str, str)  # level_name, formatted_message


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal."""
    def __init__(self, bridge: LogSignalBridge):
        super().__init__()
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.bridge.message.emit(record.levelname, msg)
        except Exception:
            self.handleError(record)


# ── Vector Database Class ────────────────────────────────────────────────────
class VectorDatabase:
    """
    Core vector database with B-spline scoring.
    Supports two architectures:
      • Global 1D Additive  — independent 1D spline per dimension, averaged
      • Cascading Local 2D  — global PCA→cluster→local PCA→2D bivariate spline
    """

    def __init__(self, X, y=None, use_gpu=False):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64) if y is not None else np.zeros(len(X))
        self.n_samples, self.n_dims = self.X.shape
        logger.debug(f"VectorDatabase init: {self.n_samples} samples, {self.n_dims} dimensions")

        self.use_gpu = use_gpu and HAS_CUPY and cp.is_available()
        if self.use_gpu:
            self.X_cupy = cp.array(self.X, dtype=cp.float32)
            logger.info(f"GPU mode ON — transferred {self.n_samples}x{self.n_dims} array to CuPy device")
        else:
            self.X_cupy = None
            if use_gpu:
                logger.warning("GPU requested but CuPy not available — falling back to CPU")
            else:
                logger.debug("GPU mode OFF — using NumPy CPU backend")

        # Spline state
        self.splines_1d = None
        self.degree = 3
        self.n_knots = 8

        # Cascading state
        self.mode = "Global 1D Additive"
        self.global_pca = None
        self.clusterer = None
        self.cluster_predictor = None
        self.local_pcas = []
        self.local_splines_2d = []
        self.n_clusters = 0
        self.cluster_labels = None
        self.cluster_centers = None

    # ── Fitting ──────────────────────────────────────────────────────────────
    def fit_splines(self, degree=3, n_knots=8, mode="Global 1D Additive",
                    n_clusters=5, eps=1.0, min_samples=5, progress_callback=None):
        logger.info(f"Fitting splines — mode={mode}, degree={degree}, "
                     f"n_knots={n_knots}, n_clusters={n_clusters}, eps={eps}, min_samples={min_samples}")
        self.mode = mode
        self.degree = degree
        self.n_knots = n_knots
        self.n_clusters = n_clusters

        if progress_callback:
            progress_callback.emit(10)

        if mode == "Global 1D Additive":
            self._fit_global_1d(progress_callback)
        elif "Cascading" in mode:
            self._fit_cascading_local_2d(eps, min_samples, progress_callback)

        if progress_callback:
            progress_callback.emit(100)
        logger.info("Spline fitting complete")

    def _fit_global_1d(self, progress_callback=None):
        logger.debug("Fitting Global 1D Additive splines...")
        self.splines_1d = []
        total = self.n_dims
        for d in range(total):
            if progress_callback and d % max(1, total // 10) == 0:
                progress_callback.emit(10 + int(80 * d / total))
            idx = np.argsort(self.X[:, d])
            x_d, y_d = self.X[idx, d], self.y[idx]
            if len(np.unique(x_d)) < self.degree + 2:
                t = np.array([x_d.min(), x_d.max()])
                spline = BSpline(t, [0.0, 1.0], k=1)
                logger.debug(f"  Dim {d}: not enough unique values, using linear fallback")
            else:
                t_interior = np.linspace(
                    x_d.min() + 1e-6, x_d.max() - 1e-6, max(1, self.n_knots - 2))
                t = np.concatenate([
                    np.full(self.degree + 1, x_d.min()),
                    t_interior,
                    np.full(self.degree + 1, x_d.max())
                ])
                spline = make_lsq_spline(x_d, y_d, t=t, k=self.degree)
                logger.debug(f"  Dim {d}: fit degree-{self.degree} spline with {len(t)} knots")
            self.splines_1d.append(spline)

    def _fit_cascading_local_2d(self, eps, min_samples, progress_callback=None):
        logger.debug("Fitting Cascading Local 2D transforms...")
        if progress_callback:
            progress_callback.emit(15)

        # Step 1: Global PCA to 2D world space
        self.global_pca = PCA(n_components=2)
        X_world_2d = self.global_pca.fit_transform(self.X)
        var_captured = self.global_pca.explained_variance_ratio_.sum()
        logger.info(f"  Global PCA (2D) captures {var_captured * 100:.2f}% variance")

        if progress_callback:
            progress_callback.emit(30)

        # Step 2: Clustering in world space
        if "DBSCAN" in self.mode:
            logger.debug(f"  Running DBSCAN: eps={eps}, min_samples={min_samples}")
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.cluster_labels = self.clusterer.fit_predict(X_world_2d)
            valid_mask = self.cluster_labels != -1
            n_noise = int(np.sum(self.cluster_labels == -1))

            if not np.any(valid_mask):
                self.cluster_labels = np.zeros(self.n_samples, dtype=int)
                valid_mask = np.ones(self.n_samples, dtype=bool)
                self.cluster_centers = X_world_2d.mean(axis=0).reshape(1, 2)
                self.n_clusters = 1
                self.cluster_predictor = KNeighborsClassifier(n_neighbors=1)
                self.cluster_predictor.fit(X_world_2d, self.cluster_labels)
                logger.warning("  DBSCAN marked all points as noise — single cluster fallback")
            else:
                unique_labels = np.unique(self.cluster_labels[valid_mask])
                self.n_clusters = len(unique_labels)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                self.cluster_labels = np.array(
                    [label_map.get(l, -1) for l in self.cluster_labels])
                self.cluster_predictor = KNeighborsClassifier(n_neighbors=1)
                self.cluster_predictor.fit(X_world_2d[valid_mask], self.cluster_labels[valid_mask])
                centers = []
                for i in range(self.n_clusters):
                    centers.append(X_world_2d[self.cluster_labels == i].mean(axis=0))
                self.cluster_centers = np.array(centers) if centers else np.empty((0, 2))
                logger.info(f"  DBSCAN found {self.n_clusters} clusters, {n_noise} noise points")
        else:
            logger.debug(f"  Running K-Means: n_clusters={self.n_clusters}")
            self.clusterer = KMeans(
                n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.cluster_labels = self.clusterer.fit_predict(X_world_2d)
            self.cluster_centers = self.clusterer.cluster_centers_.copy()
            self.cluster_predictor = self.clusterer
            logger.info(f"  K-Means assigned {self.n_clusters} clusters")

        if progress_callback:
            progress_callback.emit(50)

        # Step 3: Local PCA + 2D spline per cluster
        self.local_pcas, self.local_splines_2d = [], []
        skipped = 0
        for i in range(self.n_clusters):
            if progress_callback and i % max(1, self.n_clusters // 5) == 0:
                progress_callback.emit(50 + int(45 * i / max(1, self.n_clusters)))
            mask = (self.cluster_labels == i)
            count = int(np.sum(mask))
            if count < self.degree + 2:
                self.local_pcas.append(None)
                self.local_splines_2d.append(None)
                skipped += 1
                logger.debug(f"  Local cluster {i}: only {count} pts, "
                             f"skipping (need {self.degree + 2})")
                continue

            X_local_highd, y_local = self.X[mask], self.y[mask]
            local_pca = PCA(n_components=2)
            X_local_2d = local_pca.fit_transform(X_local_highd)
            self.local_pcas.append(local_pca)
            try:
                spline_2d = SmoothBivariateSpline(
                    X_local_2d[:, 0], X_local_2d[:, 1], y_local,
                    kx=min(self.degree, 3), ky=min(self.degree, 3),
                    s=len(y_local) * 0.01)
                logger.debug(f"  Local cluster {i}: fit 2D spline ({count} pts)")
            except Exception as e:
                spline_2d = None
                logger.warning(f"  Local cluster {i}: spline fit failed — {e}")
            self.local_splines_2d.append(spline_2d)

        if skipped:
            logger.warning(f"  Skipped {skipped}/{self.n_clusters} local clusters "
                           f"due to insufficient points")

    # ── Scoring ──────────────────────────────────────────────────────────────
    def get_percentage_score(self, x):
        if self.mode == "Global 1D Additive":
            return self._score_global(x)
        return self._score_cascading(x)

    def _score_global(self, x):
        if self.splines_1d is None:
            return 0.5
        scores = []
        for d in range(self.n_dims):
            try:
                scores.append(np.clip(float(self.splines_1d[d](x[d])), 0, 1))
            except Exception:
                scores.append(0.5)
        return float(np.clip(np.mean(scores), 0, 1))

    def _score_cascading(self, x):
        if self.global_pca is None or self.cluster_centers is None or len(self.cluster_centers) == 0:
            return 0.5
        x_reshaped = np.asarray(x).reshape(1, -1)
        x_world_2d = self.global_pca.transform(x_reshaped)[0]
        nearest_idx = int(np.argmin(
            np.linalg.norm(self.cluster_centers - x_world_2d, axis=1)))
        if nearest_idx >= len(self.local_pcas) or self.local_pcas[nearest_idx] is None:
            return 0.5
        x_local_2d = self.local_pcas[nearest_idx].transform(x_reshaped)[0]
        spline = self.local_splines_2d[nearest_idx]
        if spline is not None:
            try:
                val = spline(x_local_2d[0], x_local_2d[1], grid=False)
                return float(np.clip(val[0] if np.ndim(val) > 0 else val, 0, 1))
            except Exception:
                return 0.5
        return 0.5

    def score_grid_cascading(self, xx, yy):
        """Evaluate cascading scores on a 2D meshgrid in world space."""
        grid_world = np.column_stack([xx.ravel(), yy.ravel()])
        cluster_ids = self.cluster_predictor.predict(grid_world)
        zz = np.full(len(grid_world), 0.5)
        for i in range(self.n_clusters):
            mask = (cluster_ids == i)
            if not np.any(mask):
                continue
            if i >= len(self.local_pcas) or self.local_pcas[i] is None:
                continue
            pts_highd = self.global_pca.inverse_transform(grid_world[mask])
            pts_local_2d = self.local_pcas[i].transform(pts_highd)
            spline = self.local_splines_2d[i]
            if spline is not None:
                try:
                    zz[mask] = np.clip(
                        spline(pts_local_2d[:, 0], pts_local_2d[:, 1], grid=False),
                        0, 1)
                except Exception:
                    pass
        return zz.reshape(xx.shape)

    def get_all_scores(self, progress_callback=None):
        """Compute scores for all stored vectors."""
        n = len(self.X)
        scores = np.empty(n)
        for i in range(n):
            if progress_callback and i % max(1, n // 20) == 0:
                progress_callback.emit(int(100 * i / n))
            scores[i] = self.get_percentage_score(self.X[i])
        return scores

    # ── Queries ──────────────────────────────────────────────────────────────
    def query(self, x_query, k=5):
        x_query = np.asarray(x_query, dtype=np.float64)
        if self.use_gpu and self.X_cupy is not None:
            x_c = cp.array(x_query, dtype=cp.float32)
            distances = cp.linalg.norm(self.X_cupy - x_c, axis=1).get()
        else:
            distances = np.linalg.norm(self.X - x_query, axis=1)

        k = min(k, len(distances))
        top_k_idx = np.argsort(distances)[:k]
        results = [
            {
                'index': int(idx),
                'distance': float(distances[idx]),
                'score': float(self.get_percentage_score(self.X[idx]))
            }
            for idx in top_k_idx
        ]
        logger.debug(f"KNN query: k={k}, top_dist={results[0]['distance']:.4f}, "
                     f"top_score={results[0]['score']:.4f}")
        return results

    def range_query(self, x_query, radius=1.0):
        x_query = np.asarray(x_query, dtype=np.float64)
        if self.use_gpu and self.X_cupy is not None:
            x_c = cp.array(x_query, dtype=cp.float32)
            distances = cp.linalg.norm(self.X_cupy - x_c, axis=1).get()
        else:
            distances = np.linalg.norm(self.X - x_query, axis=1)

        mask = distances <= radius
        results = [
            {
                'index': int(i),
                'distance': float(distances[i]),
                'score': float(self.get_percentage_score(self.X[i]))
            }
            for i in np.where(mask)[0]
        ]
        logger.debug(f"Range query: radius={radius}, found {len(results)} points")
        return results

    # ── Evaluation ───────────────────────────────────────────────────────────
    def evaluate_knn_accuracy(self, k=5, progress_callback=None):
        """Leave-one-out KNN purity: fraction of samples where majority of
        k neighbours share the same class label."""
        if len(np.unique(self.y)) < 2:
            return None
        logger.info(f"Evaluating KNN purity accuracy (k={k}) over {len(self.X)} samples...")
        correct = 0
        n = len(self.X)
        for i in range(n):
            if progress_callback and i % max(1, n // 20) == 0:
                progress_callback.emit(int(100 * i / n))
            results = self.query(self.X[i], k=k + 1)[1:]  # skip self
            same_class = sum(
                1 for r in results if abs(self.y[r['index']] - self.y[i]) < 1e-9)
            if same_class >= k / 2:
                correct += 1
        acc = correct / n
        logger.info(f"KNN purity accuracy: {acc * 100:.2f}% ({correct}/{n} correct)")
        return acc

    # ── Serialization ────────────────────────────────────────────────────────
    def to_dict(self):
        return {
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'n_dims': int(self.n_dims),
            'n_samples': int(self.n_samples)
        }


# ── Worker Thread ────────────────────────────────────────────────────────────
class ComputationThread(QThread):
    """Runs a callable in a background thread, forwarding progress signals."""
    finished = Signal(object)
    progress = Signal(int)

    def __init__(self, operation, op_name="operation", *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.op_name = op_name
        self.args = args
        self.kwargs = kwargs
        logger.debug(f"Worker thread spawned: {op_name}")

    def run(self):
        try:
            result = self.operation(
                *self.args, **self.kwargs, progress_callback=self.progress)
            logger.debug(f"Worker thread finished: {self.op_name}")
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Worker thread FAILED ({self.op_name}): {e}", exc_info=True)
            self.finished.emit({'error': str(e)})


# ── Matplotlib Canvas ────────────────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e2e')
        super().__init__(self.fig)
        self.setParent(parent)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        policy.setVerticalPolicy(QSizePolicy.Policy.Expanding)
        self.setSizePolicy(policy)
        self.updateGeometry()


# ── Main Application ─────────────────────────────────────────────────────────
class BplineVectorDBApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Pro Vector DB — Cascading B-Spline Transforms, GPU & Projection Geometry")
        self.setMinimumSize(1600, 1000)

        # State
        self.vdb = None
        self.current_dataset = None
        self.current_scores = None
        self.pca_2d = PCA(n_components=2)
        self.pca_3d = PCA(n_components=3)
        self.reducer_2d = self.pca_2d
        self.var_2d = 0.0
        self.var_3d = 0.0
        self._active_worker = None
        self._log_lines: list[str] = []
        self._MAX_LOG_LINES = 500
        self.last_query_points_2d = None
        self.last_query_results = None

        # UI
        self.init_ui()
        self.apply_dark_theme()

        # Log bridge → routes terminal logs into the UI log tab
        self._log_bridge = LogSignalBridge()
        self._log_bridge.message.connect(self._append_log)
        _qt_handler = QtLogHandler(self._log_bridge)
        _qt_handler.setLevel(logging.INFO)
        logger.addHandler(_qt_handler)

        # Status bar
        self.statusBar().showMessage("Ready — select a dataset and click Load & Z-Scale")
        self.statusBar().setStyleSheet(
            "color: #a6adc8; background-color: #181825; font-family: Consolas;")

        self.load_settings()
        logger.info("Application initialized — window size %dx%d",
                     self.width(), self.height())

    # ── UI Construction ──────────────────────────────────────────────────────
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ── Left Panel (scrollable controls) ─────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(330)
        left_scroll.setMinimumWidth(290)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        left_scroll.setWidget(left_panel)

        # Dataset group
        dg = QGroupBox("📊 Dataset & Whitening")
        dl = QFormLayout(dg)
        self.dataset_combo = QComboBox()
        datasets = [
            "Moons (2D)", "Circles (2D)", "Iris (4D)",
            "Digits (64D)", "Random Gaussian (10D)", "Load from File..."
        ]
        self.dataset_combo.addItems(datasets)
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_change)
        self.dataset_combo.setToolTip("Choose a built-in dataset or load from file")
        dl.addRow("Type:", self.dataset_combo)

        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(50, 50000)
        self.n_samples_spin.setValue(1000)
        self.n_samples_spin.setSingleStep(100)
        self.n_samples_spin.setToolTip("Number of samples to generate (ignored for Iris/Digits)")
        dl.addRow("Samples:", self.n_samples_spin)

        self.load_dataset_btn = QPushButton("🔄 Load & Z-Scale")
        self.load_dataset_btn.setToolTip("Load dataset and apply StandardScaler (zero mean, unit variance)")
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        dl.addRow(self.load_dataset_btn)
        left_layout.addWidget(dg)

        # Dimensionality reduction group
        rg = QGroupBox("📉 Dim Reduction (Vis)")
        rl = QFormLayout(rg)
        self.reduction_combo = QComboBox()
        red_methods = ["PCA", "t-SNE", "Isomap", "MDS"]
        if HAS_UMAP:
            red_methods.append("UMAP")
        self.reduction_combo.addItems(red_methods)
        self.reduction_combo.setToolTip(
            "Method for projecting high-D data to 2D/3D for visualization.\n"
            "PCA is used internally by Cascading mode regardless of this setting.")
        rl.addRow("Method:", self.reduction_combo)
        left_layout.addWidget(rg)

        # Math architecture group
        bg = QGroupBox("📐 Math Architecture")
        bl = QFormLayout(bg)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Global 1D Additive",
            "Cascading Local 2D (K-Means)",
            "Cascading Local 2D (DBSCAN)"
        ])
        self.mode_combo.currentTextChanged.connect(self.toggle_local_params)
        self.mode_combo.setToolTip(
            "Global 1D: fit independent spline per dimension, average scores.\n"
            "Cascading: PCA→cluster→local PCA→2D bivariate spline per cluster.")
        bl.addRow("Mode:", self.mode_combo)

        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 5)
        self.degree_spin.setValue(3)
        self.degree_spin.setToolTip("B-spline polynomial degree (p). Higher = smoother but may overfit.")
        bl.addRow("Degree (p):", self.degree_spin)

        self.n_knots_spin = QSpinBox()
        self.n_knots_spin.setRange(3, 30)
        self.n_knots_spin.setValue(8)
        self.n_knots_spin.setToolTip("Number of internal knots for Global 1D mode")
        bl.addRow("Knots (Global):", self.n_knots_spin)

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 30)
        self.n_clusters_spin.setValue(5)
        self.n_clusters_spin.setEnabled(False)
        self.n_clusters_spin.setToolTip("Number of clusters for K-Means cascading")
        bl.addRow("Local Clusters:", self.n_clusters_spin)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.01, 50.0)
        self.eps_spin.setValue(1.0)
        self.eps_spin.setSingleStep(0.1)
        self.eps_spin.setDecimals(2)
        self.eps_spin.setEnabled(False)
        self.eps_spin.setToolTip("DBSCAN neighbourhood radius in 2D world space")
        bl.addRow("DBSCAN Eps:", self.eps_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(2, 100)
        self.min_samples_spin.setValue(5)
        self.min_samples_spin.setEnabled(False)
        self.min_samples_spin.setToolTip("DBSCAN minimum points to form a cluster")
        bl.addRow("DBSCAN MinPts:", self.min_samples_spin)

        self.fit_splines_btn = QPushButton("🔧 Fit All Transforms")
        self.fit_splines_btn.setToolTip("Fit splines, compute all scores, run dim reduction")
        self.fit_splines_btn.clicked.connect(self.fit_splines)
        bl.addRow(self.fit_splines_btn)
        left_layout.addWidget(bg)

        # Query group
        qg = QGroupBox("🔍 KNN / Range Query")
        ql = QFormLayout(qg)
        self.k_neighbors_spin = QSpinBox()
        self.k_neighbors_spin.setRange(1, 50)
        self.k_neighbors_spin.setValue(5)
        ql.addRow("K Neighbors:", self.k_neighbors_spin)

        self.random_query_btn = QPushButton("🎲 Random Query")
        self.random_query_btn.setToolTip("Generate a random query vector in R^n and find KNN")
        self.random_query_btn.clicked.connect(self.random_query)
        ql.addRow(self.random_query_btn)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 100.0)
        self.radius_spin.setValue(2.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setDecimals(2)
        ql.addRow("Radius:", self.radius_spin)

        self.range_query_btn = QPushButton("🎯 Range Query")
        self.range_query_btn.setToolTip("Find all vectors within radius of a random query point")
        self.range_query_btn.clicked.connect(self.random_range_query)
        ql.addRow(self.range_query_btn)

        self.query_results_text = QTextEdit()
        self.query_results_text.setReadOnly(True)
        self.query_results_text.setMaximumHeight(110)
        self.query_results_text.setFont(QFont("Consolas", 9))
        ql.addRow("Results:", self.query_results_text)

        self.copy_results_btn = QPushButton("📋 Copy Results")
        self.copy_results_btn.setToolTip("Copy query results to clipboard")
        self.copy_results_btn.clicked.connect(self._copy_results_to_clipboard)
        ql.addRow(self.copy_results_btn)
        left_layout.addWidget(qg)

        # Evaluation group
        eg = QGroupBox("🧪 Evaluation")
        el = QVBoxLayout(eg)
        self.eval_btn = QPushButton("📊 KNN Purity Accuracy")
        self.eval_btn.setToolTip(
            "Leave-one-out: for each sample, check if majority of K neighbours "
            "share the same class label.\nOnly meaningful for classification datasets.")
        self.eval_btn.clicked.connect(self.evaluate_model)
        el.addWidget(self.eval_btn)
        self.eval_text = QTextEdit()
        self.eval_text.setReadOnly(True)
        self.eval_text.setMaximumHeight(50)
        self.eval_text.setFont(QFont("Consolas", 9))
        el.addWidget(self.eval_text)
        left_layout.addWidget(eg)

        # History group
        hg = QGroupBox("📜 Query History")
        hl = QVBoxLayout(hg)
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(
            ["Time", "Type", "Avg Dist", "Avg Score"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setMaximumHeight(120)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        hl.addWidget(self.history_table)

        self.clear_history_btn = QPushButton("🗑️ Clear History")
        self.clear_history_btn.clicked.connect(self.clear_history)
        hl.addWidget(self.clear_history_btn)
        left_layout.addWidget(hg)

        # I/O group
        ig = QGroupBox("💾 I/O & System")
        il = QVBoxLayout(ig)
        io_row1 = QHBoxLayout()
        self.export_btn = QPushButton("💾 Exp Data")
        self.export_btn.setToolTip("Export X, y to JSON")
        self.export_btn.clicked.connect(self.export_database)
        io_row1.addWidget(self.export_btn)
        self.import_btn = QPushButton("📂 Imp Data")
        self.import_btn.setToolTip("Import X, y from JSON")
        self.import_btn.clicked.connect(self.import_database)
        io_row1.addWidget(self.import_btn)
        self.export_plot_btn = QPushButton("🖼️ Exp Plot")
        self.export_plot_btn.setToolTip("Save current tab's plot as PNG/SVG/PDF")
        self.export_plot_btn.clicked.connect(self.export_visualization)
        io_row1.addWidget(self.export_plot_btn)
        il.addLayout(io_row1)

        self.gpu_checkbox = QCheckBox("GPU Acceleration (CuPy)")
        self.gpu_checkbox.setChecked(False)
        self.gpu_checkbox.setToolTip(
            "Use CuPy for GPU-accelerated distance computations.\n"
            "Requires CuPy installed and a compatible NVIDIA GPU.")
        if not HAS_CUPY:
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setToolTip(
                "CuPy not installed.\nInstall: conda install -c conda-forge cupy=12.6")
        il.addWidget(self.gpu_checkbox)
        left_layout.addWidget(ig)
        left_layout.addStretch()

        # ── Right Panel (tabs) ───────────────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        self.tab_2d = QWidget()
        self.canvas_2d = MplCanvas(self, width=6, height=5)
        QVBoxLayout(self.tab_2d).addWidget(self.canvas_2d)
        self.tabs.addTab(self.tab_2d, "🌐 World / Vis (2D)")

        self.tab_geom = QWidget()
        self.canvas_geom = MplCanvas(self, width=6, height=5)
        QVBoxLayout(self.tab_geom).addWidget(self.canvas_geom)
        self.tabs.addTab(self.tab_geom, "🧊 Projection Geometry")

        self.tab_3d = QWidget()
        self.canvas_3d = MplCanvas(self, width=6, height=5)
        QVBoxLayout(self.tab_3d).addWidget(self.canvas_3d)
        self.tabs.addTab(self.tab_3d, "🎯 3D Projection")

        self.tab_splines = QWidget()
        self.canvas_splines = MplCanvas(self, width=8, height=5)
        QVBoxLayout(self.tab_splines).addWidget(self.canvas_splines)
        self.tabs.addTab(self.tab_splines, "📈 Spline Geometry")

        # ── NEW: B-Spline Detail Tab ─────────────────────────────────────────
        self.tab_bspline_detail = QWidget()
        bspline_layout = QVBoxLayout(self.tab_bspline_detail)
        bspline_layout.setContentsMargins(4, 4, 4, 4)

        bspline_ctrl_row = QHBoxLayout()
        self.bspline_view_combo = QComboBox()
        self.bspline_view_combo.addItems([
            "All 1D Curves (Grid)",
            "All 1D Curves (Overlay)",
            "Basis Functions",
            "Knot Diagram",
            "Cascading 2D Surfaces"
        ])
        self.bspline_view_combo.setToolTip(
            "All 1D Curves: individual spline per dimension with data.\n"
            "Basis Functions: show the B-spline basis pieces.\n"
            "Knot Diagram: knot positions on a number line.\n"
            "Cascading 2D Surfaces: bivariate spline heatmaps per cluster.")
        self.bspline_view_combo.currentTextChanged.connect(self.update_bspline_detail_tab)
        bspline_ctrl_row.addWidget(QLabel("View:"))
        bspline_ctrl_row.addWidget(self.bspline_view_combo, stretch=1)

        self.bspline_dim_spin = QSpinBox()
        self.bspline_dim_spin.setRange(0, 999)
        self.bspline_dim_spin.setValue(0)
        self.bspline_dim_spin.setToolTip("Highlight a specific dimension (0-based)")
        self.bspline_dim_spin.valueChanged.connect(self.update_bspline_detail_tab)
        bspline_ctrl_row.addWidget(QLabel("Highlight Dim:"))
        bspline_ctrl_row.addWidget(self.bspline_dim_spin)

        self.bspline_refresh_btn = QPushButton("🔄 Refresh")
        self.bspline_refresh_btn.clicked.connect(self.update_bspline_detail_tab)
        bspline_ctrl_row.addWidget(self.bspline_refresh_btn)
        bspline_layout.addLayout(bspline_ctrl_row)

        self.canvas_bspline_detail = MplCanvas(self, width=10, height=7)
        bspline_layout.addWidget(self.canvas_bspline_detail)

        self.bspline_info_text = QTextEdit()
        self.bspline_info_text.setReadOnly(True)
        self.bspline_info_text.setMaximumHeight(80)
        self.bspline_info_text.setFont(QFont("Consolas", 9))
        bspline_layout.addWidget(self.bspline_info_text)

        self.tabs.addTab(self.tab_bspline_detail, "🌀 B-Spline Detail")
        # ── END NEW TAB ─────────────────────────────────────────────────────

        self.tab_hist = QWidget()
        self.canvas_hist = MplCanvas(self, width=6, height=5)
        QVBoxLayout(self.tab_hist).addWidget(self.canvas_hist)
        self.tabs.addTab(self.tab_hist, "📊 Score Histogram")

        self.tab_heatmap = QWidget()
        self.canvas_heatmap = MplCanvas(self, width=6, height=5)
        QVBoxLayout(self.tab_heatmap).addWidget(self.canvas_heatmap)
        self.tabs.addTab(self.tab_heatmap, "🔥 Distance Heatmap")

        self.tab_elbow = QWidget()
        self.canvas_elbow = MplCanvas(self, width=6, height=5)
        QVBoxLayout(self.tab_elbow).addWidget(self.canvas_elbow)
        self.tabs.addTab(self.tab_elbow, "📐 Elbow Method")

        if HAS_PANDAS:
            self.tab_parallel = QWidget()
            self.canvas_parallel = MplCanvas(self, width=6, height=5)
            QVBoxLayout(self.tab_parallel).addWidget(self.canvas_parallel)
            self.tabs.addTab(self.tab_parallel, "〰️ Parallel Coords")

        self.tab_stats = QWidget()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        QVBoxLayout(self.tab_stats).addWidget(self.stats_text)
        self.tabs.addTab(self.tab_stats, "📋 High-D Math Stats")

        self.tab_log = QWidget()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        QVBoxLayout(self.tab_log).addWidget(self.log_text)
        self.tabs.addTab(self.tab_log, "📝 Log")

        right_layout.addWidget(self.tabs)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        right_layout.addWidget(self.progress_bar)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        # Canvas click for interactive query
        self.canvas_2d.mpl_connect('button_press_event', self.on_canvas_click)

    # ── Theme ────────────────────────────────────────────────────────────────
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QScrollArea { background-color: #1e1e2e; }
            QWidget { background-color: #1e1e2e; color: #cdd6f4;
                      font-family: 'Consolas', 'Courier New', monospace;
                      font-size: 11px; }
            QGroupBox { background-color: #313244; border: 1px solid #45475a;
                        border-radius: 5px; margin-top: 10px; padding-top: 14px;
                        font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px;
                               padding: 0 5px; color: #89b4fa; }
            QPushButton { background-color: #585b70; color: #cdd6f4; border: none;
                          padding: 7px 14px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #7f849c; }
            QPushButton:pressed { background-color: #45475a; }
            QPushButton:disabled { background-color: #313244; color: #585b70; }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #313244; border: 1px solid #45475a;
                padding: 4px 6px; border-radius: 3px; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #313244;
                                          color: #cdd6f4; selection-background-color: #585b70; }
            QTextEdit, QTableWidget { background-color: #181825; border: 1px solid #45475a;
                                      border-radius: 3px; padding: 4px; color: #a6e3a1; }
            QTableWidget::item { padding: 2px; }
            QTableWidget::item:selected { background-color: #45475a; }
            QHeaderView::section { background-color: #313244; color: #89b4fa;
                                   border: 1px solid #45475a; padding: 4px; font-weight: bold; }
            QProgressBar { border: 1px solid #45475a; border-radius: 3px;
                           text-align: center; background-color: #313244;
                           color: #cdd6f4; min-height: 18px; }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 2px; }
            QTabWidget::pane { border: 1px solid #45475a; border-radius: 5px;
                               background-color: #313244; }
            QTabBar::tab { background-color: #45475a; color: #cdd6f4;
                           padding: 7px 10px; margin-right: 2px;
                           border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #89b4fa; color: #1e1e2e;
                                    font-weight: bold; }
            QTabBar::tab:hover:!selected { background-color: #585b70; }
            QCheckBox::indicator { width: 15px; height: 15px; }
            QCheckBox::indicator:unchecked { background-color: #45475a;
                                             border: 1px solid #585b70; border-radius: 3px; }
            QCheckBox::indicator:checked { background-color: #89b4fa;
                                           border: 1px solid #89b4fa; border-radius: 3px; }
            QScrollBar:vertical { background: #1e1e2e; width: 10px; border: none; }
            QScrollBar::handle:vertical { background: #45475a; border-radius: 4px;
                                          min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QSplitter::handle { background-color: #45475a; width: 3px; }
        """)

    # ── Param Toggles ────────────────────────────────────────────────────────
    def toggle_local_params(self, text):
        is_cascading = "Cascading" in text
        is_kmeans = "K-Means" in text
        self.n_clusters_spin.setEnabled(is_kmeans)
        self.n_knots_spin.setEnabled(text == "Global 1D Additive")
        self.eps_spin.setEnabled(is_cascading and not is_kmeans)
        self.min_samples_spin.setEnabled(is_cascading and not is_kmeans)
        logger.debug(f"Mode → '{text}'  cascading={is_cascading}, kmeans={is_kmeans}")

    # ── Utility ──────────────────────────────────────────────────────────────
    def _is_busy(self):
        return self._active_worker is not None and self._active_worker.isRunning()

    def _set_ui_busy(self, busy: bool):
        for w in [self.load_dataset_btn, self.fit_splines_btn,
                  self.random_query_btn, self.range_query_btn, self.eval_btn]:
            w.setEnabled(not busy)

    def _append_log(self, level: str, msg: str):
        color = "#a6e3a1" if level == "INFO" else "#f9e2af" if level == "WARNING" else "#f38ba8"
        self._log_lines.append(f'<span style="color:{color}">[{level}] {msg}</span>')
        if len(self._log_lines) > self._MAX_LOG_LINES:
            self._log_lines = self._log_lines[-self._MAX_LOG_LINES:]
        self.log_text.setHtml("<br>".join(self._log_lines))
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Dataset Loading ──────────────────────────────────────────────────────
    def on_dataset_change(self, text):
        if text == "Load from File...":
            self.load_dataset_from_file()

    def load_dataset(self):
        if self._is_busy():
            return
        self._set_ui_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        dataset_type = self.dataset_combo.currentText()
        n_samples = self.n_samples_spin.value()
        logger.info(f"Loading dataset: {dataset_type} (n={n_samples})")
        self.statusBar().showMessage(f"Loading {dataset_type}...")

        def load_op(progress_callback=None):
            if progress_callback:
                progress_callback.emit(10)

            if dataset_type == "Moons (2D)":
                X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
                y = y.astype(float)
            elif dataset_type == "Circles (2D)":
                X, y = make_circles(
                    n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
                y = y.astype(float)
            elif dataset_type == "Iris (4D)":
                iris = load_iris()
                X, y = iris.data.copy(), iris.target.astype(float) / 2.0
                if len(X) > n_samples:
                    idx = np.random.choice(len(X), n_samples, replace=False)
                    X, y = X[idx], y[idx]
            elif dataset_type == "Digits (64D)":
                digits = load_digits()
                X, y = digits.data.copy(), digits.target.astype(float) / 9.0
                if len(X) > n_samples:
                    idx = np.random.choice(len(X), n_samples, replace=False)
                    X, y = X[idx], y[idx]
            elif dataset_type == "Random Gaussian (10D)":
                X = np.random.randn(n_samples, 10) * 2
                y = np.exp(-np.sum(X ** 2, axis=1) / 8)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            if progress_callback:
                progress_callback.emit(60)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            if progress_callback:
                progress_callback.emit(100)

            logger.info(f"Dataset loaded: {X.shape[0]}x{X.shape[1]}, "
                         f"y range=[{y.min():.3f}, {y.max():.3f}]")
            return {'X': X, 'y': y}

        self._active_worker = ComputationThread(
            load_op, op_name=f"load_{dataset_type}")
        self._active_worker.finished.connect(self._on_dataset_loaded)
        self._active_worker.progress.connect(self.progress_bar.setValue)
        self._active_worker.start()

    def _on_dataset_loaded(self, result):
        self.progress_bar.setVisible(False)
        self._set_ui_busy(False)
        if 'error' in result:
            logger.error(f"Dataset load error: {result['error']}")
            self.statusBar().showMessage("Dataset load FAILED")
            QMessageBox.critical(self, "Error", result['error'])
            return
        self.current_dataset = result
        self.vdb = VectorDatabase(
            result['X'], result['y'], use_gpu=self.gpu_checkbox.isChecked())
        
        # Clamp dimension spinner
        self.bspline_dim_spin.setMaximum(max(0, self.vdb.n_dims - 1))
        self.bspline_dim_spin.setValue(0)
        
        self.statusBar().showMessage(
            f"Loaded {self.vdb.n_samples}×{self.vdb.n_dims} — fitting splines...")
        self.fit_splines()

    def load_dataset_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Dataset", "",
            "NumPy (*.npy);;CSV (*.csv);;JSON (*.json);;All Files (*)")
        if not file_path:
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.setCurrentIndex(0)
            self.dataset_combo.blockSignals(False)
            return

        logger.info(f"Loading dataset from file: {file_path}")
        self.statusBar().showMessage(f"Loading {os.path.basename(file_path)}...")

        try:
            if file_path.endswith('.npy'):
                X = np.load(file_path).astype(np.float64)
                y = np.zeros(len(X))
            elif file_path.endswith('.csv'):
                if HAS_PANDAS:
                    df = pd.read_csv(file_path)
                    if 'label' in df.columns:
                        y = df['label'].values.astype(float)
                        X = df.drop('label', axis=1).values.astype(float)
                    else:
                        X = df.values.astype(float)
                        y = np.zeros(len(df))
                else:
                    with open(file_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        rows = [list(map(float, row)) for row in reader if row]
                    data = np.array(rows, dtype=np.float64)
                    if header and header[-1].strip().lower() in (
                            'label', 'class', 'target', 'y'):
                        X, y = data[:, :-1], data[:, -1]
                    else:
                        X, y = data, np.zeros(len(data))
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                X = np.array(data['X'], dtype=np.float64)
                y = np.array(
                    data.get('y', np.zeros(len(data['X']))), dtype=np.float64)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            if X.ndim != 2 or X.shape[0] < 2:
                raise ValueError(
                    f"Invalid data shape: {X.shape}. Need at least 2 samples in 2D.")

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            self.current_dataset = {'X': X, 'y': y}
            self.vdb = VectorDatabase(X, y, use_gpu=self.gpu_checkbox.isChecked())

            self.bspline_dim_spin.setMaximum(max(0, self.vdb.n_dims - 1))
            self.bspline_dim_spin.setValue(0)

            self.dataset_combo.blockSignals(True)
            self.dataset_combo.setCurrentText("Load from File...")
            self.dataset_combo.blockSignals(False)

            logger.info(f"File dataset loaded: {X.shape[0]}x{X.shape[1]} "
                         f"from {os.path.basename(file_path)}")
            self.statusBar().showMessage(
                f"Loaded {X.shape[0]}×{X.shape[1]} from file — fitting splines...")
            self.fit_splines()

        except Exception as e:
            logger.error(f"Failed to load file dataset: {e}", exc_info=True)
            self.statusBar().showMessage("File load FAILED")
            QMessageBox.critical(self, "Error", str(e))
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.setCurrentIndex(0)
            self.dataset_combo.blockSignals(False)

    # ── Dimensionality Reducer Factory ───────────────────────────────────────
    def get_reducer(self, n_components=2):
        method = self.reduction_combo.currentText()
        if method == "PCA":
            return PCA(n_components=n_components)
        elif method == "t-SNE":
            return TSNE(n_components=n_components, random_state=42, init='pca',
                        learning_rate='auto')
        elif method == "Isomap":
            return Isomap(n_components=n_components)
        elif method == "MDS":
            return MDS(n_components=n_components, random_state=42,
                       normalized_stress='auto')
        elif method == "UMAP" and HAS_UMAP:
            return umap.UMAP(n_components=n_components, random_state=42)
        return PCA(n_components=n_components)

    # ── Spline Fitting ───────────────────────────────────────────────────────
    def fit_splines(self):
        if self.vdb is None:
            QMessageBox.information(self, "Info", "Load a dataset first.")
            return
        if self._is_busy():
            return

        self._set_ui_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        mode = self.mode_combo.currentText()
        self.statusBar().showMessage(f"Fitting {mode}...")

        def fit_op(progress_callback=None):
            if progress_callback:
                progress_callback.emit(5)

            self.vdb.fit_splines(
                degree=self.degree_spin.value(),
                n_knots=self.n_knots_spin.value(),
                mode=mode,
                n_clusters=self.n_clusters_spin.value(),
                eps=self.eps_spin.value(),
                min_samples=self.min_samples_spin.value(),
                progress_callback=progress_callback
            )

            if progress_callback:
                progress_callback.emit(70)

            scores = self.vdb.get_all_scores(progress_callback=progress_callback)
            logger.info(f"All scores computed: mean={scores.mean():.4f}, "
                         f"std={scores.std():.4f}, min={scores.min():.4f}, "
                         f"max={scores.max():.4f}")

            if progress_callback:
                progress_callback.emit(90)

            self.reducer_2d = self.get_reducer(2)
            self.reducer_2d.fit(self.vdb.X)
            n_3d = min(3, self.vdb.n_dims)
            self.pca_3d = PCA(n_components=n_3d)
            self.pca_3d.fit(self.vdb.X)

            if progress_callback:
                progress_callback.emit(100)
            return {'scores': scores}

        self._active_worker = ComputationThread(
            fit_op, op_name=f"fit_splines_{mode}")
        self._active_worker.finished.connect(self._on_splines_fitted)
        self._active_worker.progress.connect(self.progress_bar.setValue)
        self._active_worker.start()

    def _on_splines_fitted(self, result):
        self.progress_bar.setVisible(False)
        self._set_ui_busy(False)

        if 'error' in result:
            logger.error(f"Spline fit error: {result['error']}")
            self.statusBar().showMessage("Spline fit FAILED")
            QMessageBox.critical(self, "Error", result['error'])
            return

        self.current_scores = result['scores']
        self.var_2d = (np.sum(self.reducer_2d.explained_variance_ratio_)
                       if hasattr(self.reducer_2d, 'explained_variance_ratio_')
                       else 0.0)
        self.var_3d = np.sum(self.pca_3d.explained_variance_ratio_)
        logger.info(f"Variance captured — 2D: {self.var_2d * 100:.2f}%, "
                     f"3D: {self.var_3d * 100:.2f}%")

        self.update_all_visualizations()
        self.update_statistics()
        self.statusBar().showMessage(
            f"Ready — {self.vdb.n_samples} samples, "
            f"{self.vdb.n_dims}D, mode={self.vdb.mode}")
        logger.info("All visualizations updated")

    # ── Visualizations ───────────────────────────────────────────────────────
    def update_all_visualizations(self):
        self.update_2d_visualization()
        self.update_geometry_visualization()
        self.update_3d_visualization()
        self.update_spline_geometry()
        self.update_bspline_detail_tab()
        self.update_histogram()
        self.update_heatmap()
        self.update_elbow()
        if HAS_PANDAS:
            self.update_parallel_coordinates()

    def update_2d_visualization(self):
        if self.vdb is None: return
        fig = self.canvas_2d.fig
        fig.clear()
        ax = fig.add_subplot(111)
        
        X_2d = self.reducer_2d.transform(self.vdb.X)
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=self.current_scores, 
                        cmap='viridis', s=10, alpha=0.7, edgecolors='none')
        
        if self.last_query_points_2d is not None:
            ax.scatter(self.last_query_points_2d[0], self.last_query_points_2d[1], 
                       color='#f38ba8', s=100, marker='x', zorder=5)
                       
        fig.colorbar(sc, ax=ax, shrink=0.8, label='Score')
        var_str = f"{self.var_2d * 100:.1f}%" if self.var_2d > 0 else "N/A"
        ax.set_title(f"2D Projection ({self.reduction_combo.currentText()}) — Var: {var_str}")
        ax.set_facecolor('#181825')
        fig.tight_layout()
        self.canvas_2d.draw()

    def update_geometry_visualization(self):
        if self.vdb is None: return
        fig = self.canvas_geom.fig
        fig.clear()
        ax = fig.add_subplot(111)
        
        X_2d = self.pca_2d.fit_transform(self.vdb.X)
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=self.vdb.y, cmap='coolwarm', s=10, alpha=0.6)
        
        # Draw covariance ellipses for clusters if in cascading mode
        if self.vdb.mode == "Global 1D Additive":
            # Draw global covariance
            cov = np.cov(X_2d, rowvar=False)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.real(lambda_)
            order = lambda_.argsort()[::-1]
            v = v[:, order]
            angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
            width, height = 2 * np.sqrt(lambda_)
            ellipse = Ellipse(xy=np.mean(X_2d, axis=0), width=width, height=height, 
                              angle=angle, edgecolor='#89b4fa', fc='None', lw=2, linestyle='--')
            ax.add_patch(ellipse)
            ax.set_title("Global Projection Geometry (95% Covariance)")
        else:
            if self.vdb.cluster_labels is not None:
                for i in range(self.vdb.n_clusters):
                    mask = self.vdb.cluster_labels == i
                    if np.sum(mask) < 3: continue
                    cov = np.cov(X_2d[mask], rowvar=False)
                    lambda_, v = np.linalg.eig(cov)
                    lambda_ = np.real(lambda_)
                    if np.any(lambda_ <= 0): continue
                    order = lambda_.argsort()[::-1]
                    v = v[:, order]
                    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
                    width, height = 2 * 2 * np.sqrt(lambda_) # 2 sigma
                    ellipse = Ellipse(xy=np.mean(X_2d[mask], axis=0), width=width, height=height, 
                                      angle=angle, edgecolor='#f9e2af', fc='None', lw=1.5)
                    ax.add_patch(ellipse)
            ax.set_title(f"Cluster Projection Geometry ({self.vdb.n_clusters} clusters)")
            
        ax.set_facecolor('#181825')
        fig.tight_layout()
        self.canvas_geom.draw()

    def update_3d_visualization(self):
        if self.vdb is None or self.vdb.n_dims < 1: return
        fig = self.canvas_3d.fig
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        
        X_3d = self.pca_3d.transform(self.vdb.X)
        
        if X_3d.shape[1] < 3:
            padding = np.zeros((X_3d.shape[0], 3 - X_3d.shape[1]))
            X_3d = np.hstack([X_3d, padding])
        
        sc = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                        c=self.current_scores, cmap='plasma', s=8, alpha=0.6)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='Score')
        ax.set_title(f"3D PCA — Var: {self.var_3d * 100:.1f}%")
        ax.set_facecolor('#181825')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        fig.tight_layout()
        self.canvas_3d.draw()

    def update_spline_geometry(self):
        if self.vdb is None or self.vdb.splines_1d is None: return
        fig = self.canvas_splines.fig
        fig.clear()
        
        # Plot aggregate spline influence
        X_test = np.linspace(self.vdb.X.min(), self.vdb.X.max(), 100)
        mean_score = np.mean([np.clip(self.vdb.splines_1d[d](X_test), 0, 1) 
                              for d in range(self.vdb.n_dims)], axis=0)
        
        ax = fig.add_subplot(111)
        ax.plot(X_test, mean_score, color='#89b4fa', lw=2, label='Mean Spline Output')
        ax.fill_between(X_test, np.clip(mean_score - 0.1, 0, 1), np.clip(mean_score + 0.1, 0, 1), 
                        color='#89b4fa', alpha=0.2)
        ax.set_title("Aggregate 1D Spline Response Curve")
        ax.set_ylim(-0.05, 1.05)
        ax.set_facecolor('#181825')
        ax.legend()
        fig.tight_layout()
        self.canvas_splines.draw()

    def update_bspline_detail_tab(self, *_args):
        if self.vdb is None or (self.vdb.splines_1d is None and self.vdb.local_splines_2d is None):
            return
        view = self.bspline_view_combo.currentText()
        fig = self.canvas_bspline_detail.fig
        fig.clear()
        if view.startswith("All 1D"): self._draw_1d_spline_curves(view.endswith("Overlay"))
        elif view == "Basis Functions": self._draw_basis_functions()
        elif view == "Knot Diagram": self._draw_knot_diagram()
        elif view == "Cascading 2D Surfaces": self._draw_cascading_2d_surfaces()
        fig.tight_layout(pad=1.5)
        self.canvas_bspline_detail.draw()

    def _draw_1d_spline_curves(self, overlay=False):
        if self.vdb is None or self.vdb.splines_1d is None: return
        splines = self.vdb.splines_1d
        n_dims = len(splines)
        highlight_dim = self.bspline_dim_spin.value()
        X, y = self.vdb.X, self.vdb.y

        cmap = plt.colormaps["tab20"].resampled(n_dims)

        if overlay:
            ax = self.canvas_bspline_detail.fig.add_subplot(111)
            for d in range(n_dims):
                col = cmap(d / max(n_dims - 1, 1))
                lw = 2.5 if d == highlight_dim else 0.8
                alpha = 1.0 if d == highlight_dim else 0.4
                x_d = X[:, d]
                x_min, x_max = x_d.min(), x_d.max()
                x_fine = np.linspace(x_min, x_max, 300)
                try:
                    y_fine = np.clip(splines[d](x_fine), 0, 1)
                except Exception: continue
                label = f"Dim {d}" if d == highlight_dim or n_dims <= 12 else None
                ax.plot(x_fine, y_fine, color=col, lw=lw, alpha=alpha, label=label)
            ax.set_xlabel("Feature Value (z-scored)")
            ax.set_ylabel("Spline Output (score)")
            ax.set_title("All 1D B-Spline Curves (Overlay)")
            ax.legend(fontsize=8, loc="best", ncol=2)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="#585b70", ls="--", lw=0.7)
        else:
            ncols = min(4, n_dims)
            nrows = int(np.ceil(n_dims / ncols))
            for d in range(n_dims):
                ax = self.canvas_bspline_detail.fig.add_subplot(nrows, ncols, d + 1)
                col = cmap(d / max(n_dims - 1, 1))
                x_d = X[:, d]
                x_fine = np.linspace(x_d.min(), x_d.max(), 300)
                try:
                    y_fine = np.clip(splines[d](x_fine), 0, 1)
                except Exception:
                    ax.set_title(f"Dim {d} (err)", fontsize=9, color="#f38ba8")
                    continue
                idx_sorted = np.argsort(x_d)
                ax.scatter(x_d[idx_sorted], y[idx_sorted], s=3, alpha=0.25, color="#a6adc8", zorder=1)
                ax.plot(x_fine, y_fine, color=col, lw=2, zorder=2)
                try:
                    knots = splines[d].t
                    interior = knots[splines[d].k:-splines[d].k]
                    for kt in interior: ax.axvline(kt, color="#f38ba8", lw=0.5, alpha=0.5)
                except Exception: pass
                
                edge_color = "#f9e2af" if d == highlight_dim else "#45475a"
                edge_width = 2.5 if d == highlight_dim else 0.5
                for spine in ax.spines.values():
                    spine.set_color(edge_color)
                    spine.set_linewidth(edge_width)
                ax.set_title(f"D{d}", fontsize=9, color="#f9e2af" if d == highlight_dim else "#cdd6f4")
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, color="#585b70", ls="--", lw=0.5)
                ax.tick_params(labelsize=7)
            self.canvas_bspline_detail.fig.suptitle("Per-Dimension 1D B-Spline Curves", fontsize=13, fontweight="bold", y=1.01)

        info = f"Mode: {self.vdb.mode} | Dims: {n_dims} | Highlight: {highlight_dim}"
        self.bspline_info_text.setPlainText(info)

    def _draw_basis_functions(self):
        if self.vdb is None or self.vdb.splines_1d is None: return
        splines = self.vdb.splines_1d
        d = min(self.bspline_dim_spin.value(), len(splines) - 1)
        spl = splines[d]
        k, t = spl.k, spl.t
        n_bases = len(t) - k - 1
        
        cmap = plt.colormaps["cool"].resampled(n_bases)
        
        ax = self.canvas_bspline_detail.fig.add_subplot(111)
        x_fine = np.linspace(t[k], t[-k - 1], 500)
        for i in range(n_bases):
            c = np.zeros(n_bases); c[i] = 1.0
            y_basis = BSpline(t, c, k)(x_fine)
            ax.plot(x_fine, y_basis, color=cmap(i / max(n_bases - 1, 1)), lw=1.8, label=f"B{i}", alpha=0.85)
        for kt in t[k:-k]: ax.axvline(kt, color="#f38ba8", lw=0.6, alpha=0.4, ls="--")
        ax.set_title(f"Basis Functions — Dim {d} (k={k}, {n_bases} bases)")
        ax.legend(fontsize=7, ncol=max(1, n_bases // 8), loc="upper right")
        self.bspline_info_text.setPlainText(f"Dim {d} | Degree k={k} | Bases={n_bases} | Knots={list(np.round(t[k:-k], 3))}")

    def _draw_knot_diagram(self):
        if self.vdb is None or self.vdb.splines_1d is None: return
        splines = self.vdb.splines_1d
        n_dims = len(splines)
        highlight_dim = self.bspline_dim_spin.value()
        fig = self.canvas_bspline_detail.fig
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        
        cmap = plt.colormaps["tab20"].resampled(n_dims)

        for d in range(n_dims):
            col = cmap(d / max(n_dims - 1, 1))
            lw = 2.5 if d == highlight_dim else 0.8
            alpha = 1.0 if d == highlight_dim else 0.35
            t = splines[d].t; k = splines[d].k
            t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-12)
            ax1.scatter(t_norm, np.full_like(t_norm, d), s=12, color=col, alpha=alpha, edgecolors='none')
            interior = t[k:-k]
            t_int_norm = (interior - t.min()) / max(t.max() - t.min(), 1e-12)
            ax2.scatter(t_int_norm, np.full_like(t_int_norm, d), s=18, color=col, alpha=alpha, edgecolors='white', linewidths=0.3)

        for ax, title in [(ax1, "All Knots (incl. clamped)"), (ax2, "Interior Knots Only")]:
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.5, n_dims - 0.5)
            ax.set_yticks(range(0, n_dims, max(1, n_dims // 15)))
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axhline(highlight_dim, color="#f9e2af", lw=1.5, alpha=0.4, ls="--")

        fig.suptitle("Knot Vector Diagram", fontsize=13, fontweight="bold", y=1.01)
        self.bspline_info_text.setPlainText(f"Highlight: Dim {highlight_dim} | Degree: {self.vdb.degree}")

    def _draw_cascading_2d_surfaces(self):
        if self.vdb is None: return
        ax = self.canvas_bspline_detail.fig.add_subplot(111)
        if self.vdb.mode == "Global 1D Additive":
            ax.text(0.5, 0.5, "Switch to 'Cascading Local 2D' to view.", transform=ax.transAxes, ha='center', va='center', fontsize=14, color="#f38ba8")
            ax.set_xticks([]); ax.set_yticks([])
            self.bspline_info_text.setPlainText("Not applicable in Global 1D Additive mode.")
            return

        n_clusters = self.vdb.n_clusters
        if n_clusters == 0:
            ax.text(0.5, 0.5, "No clusters found.", transform=ax.transAxes, ha='center', va='center', fontsize=14, color="#f38ba8")
            ax.set_xticks([]); ax.set_yticks([])
            return

        # Show only the first cluster in detail for the sub-view, or a grid if small
        ncols = min(3, n_clusters); nrows = int(np.ceil(n_clusters / ncols))
        self.canvas_bspline_detail.fig.clear()
        
        for i in range(n_clusters):
            ax = self.canvas_bspline_detail.fig.add_subplot(nrows, ncols, i + 1)
            if i >= len(self.vdb.local_splines_2d) or self.vdb.local_splines_2d[i] is None:
                ax.text(0.5, 0.5, f"Cluster {i}\n(no spline)", transform=ax.transAxes, ha='center', va='center', fontsize=10, color="#585b70")
                ax.set_xticks([]); ax.set_yticks([]); continue

            spline_2d = self.vdb.local_splines_2d[i]
            local_pca = self.vdb.local_pcas[i]
            mask = (self.vdb.cluster_labels == i)
            X_local_2d = local_pca.transform(self.vdb.X[mask])
            margin = 0.3
            xx, yy = np.meshgrid(np.linspace(X_local_2d[:,0].min()-margin, X_local_2d[:,0].max()+margin, 80),
                                 np.linspace(X_local_2d[:,1].min()-margin, X_local_2d[:,1].max()+margin, 80))
            try:
                zz = np.clip(spline_2d(xx, yy, grid=True), 0, 1)
                ax.contourf(xx, yy, zz, levels=20, cmap="viridis", alpha=0.9)
                ax.contour(xx, yy, zz, levels=10, colors="white", linewidths=0.3, alpha=0.4)
                ax.scatter(X_local_2d[:,0], X_local_2d[:,1], s=4, c="white", alpha=0.4, edgecolors='none')
            except Exception as e:
                ax.text(0.5, 0.5, f"Err: {e}", transform=ax.transAxes, ha='center', va='center', fontsize=8, color="#f38ba8")
                ax.set_xticks([]); ax.set_yticks([])
            var = local_pca.explained_variance_ratio_
            ax.set_title(f"C{i} ({int(mask.sum())} pts)\nVar: {var[0]*100:.1f}%, {var[1]*100:.1f}%", fontsize=9)
            ax.tick_params(labelsize=7)

        self.canvas_bspline_detail.fig.suptitle("Cascading 2D Bivariate Splines", fontsize=13, fontweight="bold", y=1.01)
        var_global = self.vdb.global_pca.explained_variance_ratio_.sum() * 100 if self.vdb.global_pca else 0
        self.bspline_info_text.setPlainText(f"Global PCA Var: {var_global:.1f}% | Clusters: {n_clusters}")

    def update_histogram(self):
        if self.current_scores is None: return
        fig = self.canvas_hist.fig; fig.clear()
        ax = fig.add_subplot(111)
        ax.hist(self.current_scores, bins=50, color='#89b4fa', alpha=0.8, edgecolor='#1e1e2e')
        ax.axvline(np.mean(self.current_scores), color='#f38ba8', ls='--', lw=2, label=f"Mean: {np.mean(self.current_scores):.3f}")
        ax.set_title("Score Distribution")
        ax.set_xlabel("Score"); ax.set_ylabel("Frequency")
        ax.legend(); ax.set_facecolor('#181825')
        fig.tight_layout(); self.canvas_hist.draw()

    def update_heatmap(self):
        if self.vdb is None: return
        fig = self.canvas_heatmap.fig; fig.clear()
        ax = fig.add_subplot(111)
        # Limit to 500 samples for performance
        X_sub = self.vdb.X[:500]
        dist = pairwise_distances(X_sub)
        im = ax.imshow(dist, cmap='inferno', aspect='auto')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Euclidean Distance')
        ax.set_title(f"Distance Heatmap (First {min(500, len(self.vdb.X))} samples)")
        fig.tight_layout(); self.canvas_heatmap.draw()

    def update_elbow(self):
        if self.vdb is None: return
        fig = self.canvas_elbow.fig; fig.clear()
        ax = fig.add_subplot(111)
        X_2d = self.pca_2d.fit_transform(self.vdb.X)
        max_k = min(15, len(self.vdb.X) - 1)
        inertias = []
        Ks = range(2, max_k + 1)
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(X_2d)
            inertias.append(km.inertia_)
        ax.plot(list(Ks), inertias, marker='o', color='#89b4fa', lw=2)
        ax.set_title("Elbow Method (K-Means Inertia)")
        ax.set_xlabel("K (Clusters)"); ax.set_ylabel("Inertia")
        ax.set_facecolor('#181825')
        fig.tight_layout(); self.canvas_elbow.draw()

    def update_parallel_coordinates(self):
        if not HAS_PANDAS or self.vdb is None: return
        fig = self.canvas_parallel.fig; fig.clear()
        ax = fig.add_subplot(111)
        # Limit dims to 20 for visibility
        max_dims = min(20, self.vdb.n_dims)
        X_sub = self.vdb.X[:200, :max_dims]
        y_sub = self.vdb.y[:200]
        df = pd.DataFrame(X_sub, columns=[f"D{i}" for i in range(max_dims)])
        df['Target'] = y_sub
        pd.plotting.parallel_coordinates(df, 'Target', ax=ax, color=['#f38ba8', '#89b4fa', '#a6e3a1', '#f9e2af'])
        ax.set_title("Parallel Coordinates (Sampled)")
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.set_facecolor('#181825')
        fig.tight_layout(); self.canvas_parallel.draw()

    def update_statistics(self):
        if self.vdb is None or self.current_scores is None: return
        X, y, scores = self.vdb.X, self.vdb.y, self.current_scores
        
        # Basic stats
        stats = f"═══ DATASET STATS ═══\n"
        stats += f"Samples: {X.shape[0]} | Dims: {X.shape[1]}\n"
        stats += f"X Mean/Std: {X.mean():.4f} / {X.std():.4f}\n"
        stats += f"Y Range: [{y.min():.3f}, {y.max():.3f}]\n\n"
        
        stats += f"═══ SCORE STATS ═══\n"
        stats += f"Mean: {scores.mean():.4f} | Std: {scores.std():.4f}\n"
        stats += f"Min: {scores.min():.4f} | Max: {scores.max():.4f}\n\n"
        
        stats += f"═══ DIMENSIONALITY REDUCTION ═══\n"
        stats += f"2D Vis Var: {self.var_2d * 100:.2f}%\n"
        stats += f"3D PCA Var: {self.var_3d * 100:.2f}%\n"
        
        if hasattr(self.reducer_2d, 'explained_variance_ratio_'):
            for i, v in enumerate(self.reducer_2d.explained_variance_ratio_):
                stats += f"  PC{i+1}: {v*100:.2f}%\n"
                
        stats += f"\n═══ SPLINE ARCHITECTURE ═══\n"
        stats += f"Mode: {self.vdb.mode}\n"
        stats += f"Degree: {self.vdb.degree}\n"
        if self.vdb.mode == "Global 1D Additive":
            stats += f"1D Splines Fit: {len(self.vdb.splines_1d) if self.vdb.splines_1d else 0}\n"
        else:
            stats += f"Global PCA Var: {self.vdb.global_pca.explained_variance_ratio_.sum()*100:.2f}%\n"
            stats += f"Clusters: {self.vdb.n_clusters}\n"
            
        self.stats_text.setPlainText(stats)

    # ── Queries & Interactions ───────────────────────────────────────────────
    def on_canvas_click(self, event):
        if self.vdb is None or event.inaxes != self.canvas_2d.fig.axes[0]: return
        click_2d = np.array([[event.xdata, event.ydata]])
        try:
            # Inverse transform back to high-D
            if hasattr(self.reducer_2d, 'inverse_transform'):
                x_highd = self.reducer_2d.inverse_transform(click_2d)[0]
            else:
                return
                
            self.last_query_points_2d = (event.xdata, event.ydata)
            k = self.k_neighbors_spin.value()
            results = self.vdb.query(x_highd, k=k)
            self.last_query_results = results
            
            txt = f"Click Query (K={k}):\n"
            for r in results:
                txt += f"  Idx:{r['index']:4d} | Dist:{r['distance']:.3f} | Score:{r['score']:.3f}\n"
            self.query_results_text.setPlainText(txt)
            self.update_2d_visualization()
            
            self.history_table.insertRow(0)
            self.history_table.setItem(0, 0, QTableWidgetItem(datetime.datetime.now().strftime("%H:%M:%S")))
            self.history_table.setItem(0, 1, QTableWidgetItem("Click"))
            avg_d = np.mean([r['distance'] for r in results])
            avg_s = np.mean([r['score'] for r in results])
            self.history_table.setItem(0, 2, QTableWidgetItem(f"{avg_d:.3f}"))
            self.history_table.setItem(0, 3, QTableWidgetItem(f"{avg_s:.3f}"))
        except Exception as e:
            logger.debug(f"Canvas click failed: {e}")

    def random_query(self):
        if self.vdb is None or self._is_busy(): return
        k = self.k_neighbors_spin.value()
        mean = self.vdb.X.mean(axis=0)
        std = self.vdb.X.std(axis=0) + 1e-6
        x_rand = np.random.randn(self.vdb.n_dims) * std + mean
        
        results = self.vdb.query(x_rand, k=k)
        self.last_query_results = results
        
        # Map to 2d for vis
        x_2d = self.reducer_2d.transform(x_rand.reshape(1, -1))[0]
        self.last_query_points_2d = (x_2d[0], x_2d[1])
        
        txt = f"Random Query (K={k}):\n"
        for r in results:
            txt += f"  Idx:{r['index']:4d} | Dist:{r['distance']:.3f} | Score:{r['score']:.3f}\n"
        self.query_results_text.setPlainText(txt)
        self.update_2d_visualization()
        
        self.history_table.insertRow(0)
        self.history_table.setItem(0, 0, QTableWidgetItem(datetime.datetime.now().strftime("%H:%M:%S")))
        self.history_table.setItem(0, 1, QTableWidgetItem(f"KNN({k})"))
        avg_d = np.mean([r['distance'] for r in results])
        avg_s = np.mean([r['score'] for r in results])
        self.history_table.setItem(0, 2, QTableWidgetItem(f"{avg_d:.3f}"))
        self.history_table.setItem(0, 3, QTableWidgetItem(f"{avg_s:.3f}"))
        logger.info(f"Random KNN query executed: avg_dist={avg_d:.3f}, avg_score={avg_s:.3f}")

    def random_range_query(self):
        if self.vdb is None or self._is_busy(): return
        radius = self.radius_spin.value()
        mean = self.vdb.X.mean(axis=0)
        std = self.vdb.X.std(axis=0) + 1e-6
        x_rand = np.random.randn(self.vdb.n_dims) * std + mean
        
        results = self.vdb.range_query(x_rand, radius=radius)
        
        x_2d = self.reducer_2d.transform(x_rand.reshape(1, -1))[0]
        self.last_query_points_2d = (x_2d[0], x_2d[1])
        self.last_query_results = results
        
        txt = f"Range Query (R={radius:.2f}, Found={len(results)}):\n"
        for r in results[:10]: # Limit text display
            txt += f"  Idx:{r['index']:4d} | Dist:{r['distance']:.3f} | Score:{r['score']:.3f}\n"
        if len(results) > 10: txt += f"  ... and {len(results)-10} more\n"
        self.query_results_text.setPlainText(txt)
        self.update_2d_visualization()
        
        self.history_table.insertRow(0)
        self.history_table.setItem(0, 0, QTableWidgetItem(datetime.datetime.now().strftime("%H:%M:%S")))
        self.history_table.setItem(0, 1, QTableWidgetItem(f"Range({radius:.1f})"))
        avg_d = np.mean([r['distance'] for r in results]) if results else 0
        avg_s = np.mean([r['score'] for r in results]) if results else 0
        self.history_table.setItem(0, 2, QTableWidgetItem(f"{avg_d:.3f}"))
        self.history_table.setItem(0, 3, QTableWidgetItem(f"{avg_s:.3f}"))
        logger.info(f"Random Range query executed: found={len(results)}, avg_dist={avg_d:.3f}")

    def evaluate_model(self):
        if self.vdb is None or self._is_busy(): return
        if len(np.unique(self.vdb.y)) < 2:
            self.eval_text.setPlainText("Need >1 class to evaluate purity.")
            return
            
        self._set_ui_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        k = self.k_neighbors_spin.value()

        def eval_op(progress_callback=None):
            acc = self.vdb.evaluate_knn_accuracy(k=k, progress_callback=progress_callback)
            return {'accuracy': acc}

        self._active_worker = ComputationThread(eval_op, op_name="evaluate")
        self._active_worker.finished.connect(self._on_eval_finished)
        self._active_worker.progress.connect(self.progress_bar.setValue)
        self._active_worker.start()

    def _on_eval_finished(self, result):
        self.progress_bar.setVisible(False)
        self._set_ui_busy(False)
        if 'error' in result:
            self.eval_text.setPlainText(f"Error: {result['error']}")
        elif result.get('accuracy') is not None:
            self.eval_text.setPlainText(f"KNN Purity: {result['accuracy']*100:.2f}%")
        else:
            self.eval_text.setPlainText("Could not compute purity.")

    # ── I/O & System ─────────────────────────────────────────────────────────
    def export_database(self):
        if self.vdb is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Data", "vdb_data.json", "JSON (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(self.vdb.to_dict(), f)
            logger.info(f"Data exported to {path}")

    def import_database(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Data", "", "JSON (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                X = np.array(data['X'])
                y = np.array(data.get('y', np.zeros(len(X))))
                self.current_dataset = {'X': X, 'y': y}
                self.vdb = VectorDatabase(X, y, use_gpu=self.gpu_checkbox.isChecked())
                self.bspline_dim_spin.setMaximum(max(0, self.vdb.n_dims - 1))
                self.fit_splines()
                logger.info(f"Data imported from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", str(e))

    def export_visualization(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "plot.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            idx = self.tabs.currentIndex()
            canvases = [self.canvas_2d, self.canvas_geom, self.canvas_3d, self.canvas_splines, 
                        self.canvas_bspline_detail, self.canvas_hist, self.canvas_heatmap, self.canvas_elbow]
            if HAS_PANDAS: canvases.append(self.canvas_parallel)
            if 0 <= idx < len(canvases):
                canvases[idx].fig.savefig(path, facecolor=canvases[idx].fig.get_facecolor(), edgecolor='none')
                logger.info(f"Plot exported to {path}")

    def _copy_results_to_clipboard(self):
        text = self.query_results_text.toPlainText()
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Results copied to clipboard", 2000)

    def clear_history(self):
        self.history_table.setRowCount(0)

    def load_settings(self):
        config = configparser.ConfigParser()
        if os.path.exists('settings.ini'):
            config.read('settings.ini')
            if 'UI' in config:
                self.reduction_combo.setCurrentText(config['UI'].get('reducer', 'PCA'))
                self.mode_combo.setCurrentText(config['UI'].get('mode', 'Global 1D Additive'))
                self.degree_spin.setValue(int(config['UI'].get('degree', 3)))
                self.n_knots_spin.setValue(int(config['UI'].get('knots', 8)))
                self.n_clusters_spin.setValue(int(config['UI'].get('clusters', 5)))

    def closeEvent(self, event):
        config = configparser.ConfigParser()
        config['UI'] = {
            'reducer': self.reduction_combo.currentText(),
            'mode': self.mode_combo.currentText(),
            'degree': str(self.degree_spin.value()),
            'knots': str(self.n_knots_spin.value()),
            'clusters': str(self.n_clusters_spin.value())
        }
        with open('settings.ini', 'w') as f:
            config.write(f)
        if self._is_busy():
            self._active_worker.terminate()
        event.accept()


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BplineVectorDBApp()
    window.show()
    sys.exit(app.exec())