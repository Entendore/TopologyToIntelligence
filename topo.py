import sys
import threading
import numpy as np
from contextlib import nullcontext

# ==========================================
# CUPY 1.26 IMPORT WITH NUMPY FALLBACK
# ==========================================
try:
    import cupy as cp
    USE_GPU = True
    def to_cpu(x): return cp.asnumpy(x)
    def to_gpu(x): return cp.asarray(x)
    print("[INFO] CuPy 1.26+ detected. GPU acceleration enabled.")
except ImportError:
    USE_GPU = False
    cp = np  # Fallback alias for API compatibility
    def to_cpu(x): return np.asarray(x)
    def to_gpu(x): return np.asarray(x)
    print("[WARN] CuPy not found. Falling back to NumPy (CPU mode).")

from sklearn.datasets import make_moons, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QPushButton, QTabWidget, QLabel)
from PySide6.QtCore import QThread, Signal, Qt


# ==========================================
# 1. HIGH-PERFORMANCE GEOMETRIC KAN (CuPy)
# ==========================================
# Fully replaced PyTorch with CuPy 1.26 for GPU-accelerated computation.
# Implements manual backpropagation for the RBF-KAN network.
class GeometricKAN:
    def __init__(self, input_dim, num_knots=80):
        self.num_knots = num_knots
        self.input_dim = input_dim
        
        # Initialize parameters on device (GPU if available)
        self.centers = (cp.random.rand(num_knots, input_dim).astype(cp.float32) * 2 - 1)
        self.weights = cp.random.rand(num_knots, 1).astype(cp.float32)
        self.beta = cp.ones(1, dtype=cp.float32) * 10.0
        
        # Adam optimizer state
        self.m_c = cp.zeros_like(self.centers)
        self.v_c = cp.zeros_like(self.centers)
        self.m_w = cp.zeros_like(self.weights)
        self.v_w = cp.zeros_like(self.weights)
        self.m_b = cp.zeros_like(self.beta)
        self.v_b = cp.zeros_like(self.beta)
        self.t = 0
    
    def forward(self, x):
        """Forward pass: RBF interpolation with Einstein-like summation."""
        diff = x[:, None, :] - self.centers[None, :, :]  # [B, K, D]
        dist_sq = cp.sum(diff ** 2, axis=2)  # [B, K] - replaces einsum
        rbf = cp.exp(-self.beta * dist_sq)  # [B, K]
        y_pred = rbf @ self.weights  # [B, 1]
        return y_pred, (diff, dist_sq, rbf)
    
    def backward(self, y_true, y_pred, cache):
        """Manual backpropagation for MSE loss."""
        diff, dist_sq, rbf = cache
        B = y_true.shape[0]
        
        # dL/dy_pred = 2 * (pred - true) / B
        dL_dypred = 2.0 * (y_pred - y_true) / B
        
        # dL/dw = rbf^T @ dL/dy_pred
        dL_dw = rbf.T @ dL_dypred  # [K, 1]
        
        # dL/drbf = dL/dy_pred @ w^T
        dL_drbf = dL_dypred @ self.weights.T  # [B, K]
        
        # dL/dbeta = sum(-dist_sq * rbf * dL/drbf)
        dL_dbeta = cp.sum(-dist_sq * rbf * dL_drbf)
        
        # dL/dcenters via chain rule through dist_sq
        dL_ddist_sq = -self.beta * rbf * dL_drbf  # [B, K]
        dL_ddiff = 2.0 * diff * dL_ddist_sq[:, :, None]  # [B, K, D]
        dL_dc = -cp.sum(dL_ddiff, axis=0)  # [K, D]
        
        return dL_dc, dL_dw, dL_dbeta
    
    def _adam_update(self, param, grad, m, v, lr, beta1, beta2, eps):
        """Single parameter Adam update (in-place for m, v)."""
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)
        param = param - lr * m_hat / (cp.sqrt(v_hat) + eps)
        return param
    
    def adam_step(self, grads, lr=0.015, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update for all parameters."""
        self.t += 1
        dL_dc, dL_dw, dL_dbeta = grads
        
        # Update parameters (m, v are modified in-place via _adam_update)
        self.centers = self._adam_update(
            self.centers, dL_dc, self.m_c, self.v_c, lr, beta1, beta2, eps)
        self.weights = self._adam_update(
            self.weights, dL_dw, self.m_w, self.v_w, lr, beta1, beta2, eps)
        self.beta = self._adam_update(
            self.beta, dL_dbeta, self.m_b, self.v_b, lr, beta1, beta2, eps)


# ==========================================
# 2. ASYNCHRONOUS DATA WORKER
# ==========================================
class DataWorker(QThread):
    finished = Signal(object, object, str)
    error = Signal(str)

    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type

    def run(self):
        try:
            if self.data_type == "moons_bool":
                X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
                X = StandardScaler().fit_transform(X)
                self.finished.emit(X, y.astype(np.float32).reshape(-1, 1), "Boolean (True/False)")
            
            elif self.data_type == "moons_cont":
                X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
                X = StandardScaler().fit_transform(X)
                y_cont = np.abs(y - (X[:, 0] > 0).astype(np.float32)).reshape(-1, 1)
                self.finished.emit(X, y_cont, "Continuous Wrongness")
            
            elif self.data_type == "digits":
                X, y = load_digits(return_X_y=True)
                X = StandardScaler().fit_transform(PCA(n_components=2).fit_transform(X))
                self.finished.emit(X, y.astype(np.float32).reshape(-1, 1), "Discrete Foliations (0-9)")
            
            elif self.data_type == "hf":
                ds = load_dataset("imdb", split="train[:500]")
                X_high = TfidfVectorizer(max_features=100).fit_transform(ds['text']).toarray()
                X = StandardScaler().fit_transform(PCA(n_components=2).fit_transform(X_high))
                y = (np.array(ds['label']) * 2 - 1).astype(np.float32).reshape(-1, 1)
                self.finished.emit(X, y, "Continuous Sentiment (-1 to 1)")
        except Exception as e:
            self.error.emit(str(e))


# ==========================================
# 3. CANCELABLE TRAINING WORKER (CuPy)
# ==========================================
class TrainingWorker(QThread):
    finished = Signal(object, object, str)
    progress = Signal(int)

    def __init__(self, X, y, title, epochs=400):
        super().__init__()
        # Transfer data to GPU (if available) at worker init
        self.X = to_gpu(X.astype(np.float32))
        self.y = to_gpu(y.astype(np.float32))
        self.title = title
        self.epochs = epochs
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        model = GeometricKAN(input_dim=2, num_knots=80)
        lr = 0.015
        
        for epoch in range(self.epochs):
            if self._stop_event.is_set():
                return  # Instant cancellation
            
            # StepLR schedule: decay by 0.5 every 100 epochs
            if epoch > 0 and epoch % 100 == 0:
                lr *= 0.5
            
            # Forward pass
            y_pred, cache = model.forward(self.X)
            
            # Backward pass
            grads = model.backward(self.y, y_pred, cache)
            
            # Adam update
            model.adam_step(grads, lr=lr)
            
            if epoch % 50 == 0:
                self.progress.emit(epoch)

        # Generate prediction grid for visualization
        x_min, x_max = float(to_cpu(self.X[:, 0].min())) - 0.5, float(to_cpu(self.X[:, 0].max())) + 0.5
        y_min, y_max = float(to_cpu(self.X[:, 1].min())) - 0.5, float(to_cpu(self.X[:, 1].max())) + 0.5
        
        linspace_x = cp.linspace(x_min, x_max, 100)
        linspace_y = cp.linspace(y_min, y_max, 100)
        grid_x, grid_y = cp.meshgrid(linspace_x, linspace_y, indexing='ij')
        grid = cp.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
        
        # GPU synchronization context (no-op for CPU fallback)
        ctx = cp.cuda.Stream.null if USE_GPU else nullcontext()
        with ctx:
            Z_pred, _ = model.forward(grid)
        
        # Transfer all data to CPU for matplotlib plotting
        # (cp.asnumpy implicitly synchronizes GPU operations)
        Z = to_cpu(Z_pred).reshape(100, 100)
        xx = to_cpu(grid_x)
        yy = to_cpu(grid_y)
        X_cpu = to_cpu(self.X)
        y_cpu = to_cpu(self.y)
        
        self.finished.emit((xx, yy, Z), (X_cpu, y_cpu), self.title)


# ==========================================
# 4. OPTIMIZED PYSIDE6 VISUALIZER
# ==========================================
class ManifoldCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self.ax.set_title("Awaiting Geometric Projection...")

    def plot_geometric_boundary(self, grid_data, point_data, title):
        self.ax.cla()
        xx, yy, Z = grid_data
        X, y = point_data
        y_flat = y.flatten()
        
        is_discrete = len(np.unique(y_flat)) > 3 and np.all(y_flat.astype(int) == y_flat)
        
        if is_discrete:
            self.ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.6)
            self.ax.scatter(X[:, 0], X[:, 1], c=y_flat, cmap='Spectral', 
                          edgecolor='k', s=15, alpha=0.8)
            self.ax.set_title(f"Discrete Foliations\n{title}")
        else:
            self.ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.6)
            self.ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)
            self.ax.scatter(X[:, 0], X[:, 1], c=y_flat, cmap='coolwarm', 
                          edgecolor='white', s=10, alpha=0.9)
            self.ax.set_title(f"SDF / Boolean Seam\n{title}")

        self.ax.set_xlabel("Proj. Axis 1")
        self.ax.set_ylabel("Proj. Axis 2")
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"High-Performance Geo-KAN-Former (CuPy {'GPU' if USE_GPU else 'CPU'})")
        self.setGeometry(100, 100, 950, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.active_workers = {}

        self.setup_moons_tab()
        self.setup_digits_tab()
        self.setup_hf_tab()

        self.statusBar().showMessage(f"Ready. Acceleration: {'CUDA GPU' if USE_GPU else 'CPU (NumPy fallback)'}")

    def create_tab(self, name):
        tab = QWidget()
        h_layout = QHBoxLayout(tab)
        canvas = ManifoldCanvas(tab)
        h_layout.addWidget(canvas, stretch=3)
        
        control_panel = QVBoxLayout()
        h_layout.addLayout(control_panel, stretch=1)
        self.tabs.addTab(tab, name)
        return canvas, control_panel

    def setup_moons_tab(self):
        canvas, controls = self.create_tab("Moons")
        controls.addWidget(QLabel("Exact non-linear seams & gradients."))
        
        btn_bool = QPushButton("Train Boolean Boundary")
        btn_bool.clicked.connect(lambda: self.start_pipeline(canvas, "moons_bool", "Moons", btn_bool))
        controls.addWidget(btn_bool)

        btn_cont = QPushButton("Train Continuous Wrongness")
        btn_cont.clicked.connect(lambda: self.start_pipeline(canvas, "moons_cont", "Moons", btn_cont))
        controls.addWidget(btn_cont)

    def setup_digits_tab(self):
        canvas, controls = self.create_tab("Digits")
        controls.addWidget(QLabel("64D to 2D Foliations."))
        btn = QPushButton("Train Discrete Foliations")
        btn.clicked.connect(lambda: self.start_pipeline(canvas, "digits", "Digits", btn))
        controls.addWidget(btn)

    def setup_hf_tab(self):
        canvas, controls = self.create_tab("HuggingFace IMDB")
        controls.addWidget(QLabel("Semantic space boundary\n(Downloads data on first run)."))
        btn = QPushButton("Train Semantic Boundary")
        btn.clicked.connect(lambda: self.start_pipeline(canvas, "hf", "IMDB", btn))
        controls.addWidget(btn)

    def set_tab_enabled(self, tab_index, enabled):
        tab_widget = self.tabs.widget(tab_index)
        for child in tab_widget.findChildren(QPushButton):
            child.setEnabled(enabled)

    def start_pipeline(self, canvas, data_type, dataset_name, trigger_btn):
        tab_index = self.tabs.currentIndex()
        
        # Lock UI and cancel any existing workers
        self.set_tab_enabled(tab_index, False)
        if tab_index in self.active_workers:
            self.active_workers[tab_index].stop()
            self.active_workers[tab_index].wait(500)

        self.statusBar().showMessage(f"Loading data for {dataset_name}...")
        
        # Chain: Data Worker -> Training Worker
        self.data_worker = DataWorker(data_type)
        self.data_worker.error.connect(lambda e: self.handle_error(e, tab_index))
        self.data_worker.finished.connect(
            lambda X, y, t: self.on_data_ready(canvas, X, y, t, tab_index))
        self.data_worker.start()

    def on_data_ready(self, canvas, X, y, title, tab_index):
        self.statusBar().showMessage(f"Fitting KAN Splines for {title}...")
        
        self.train_worker = TrainingWorker(X, y, title)
        self.train_worker.progress.connect(
            lambda e: self.statusBar().showMessage(f"Training Epoch {e}..."))
        self.train_worker.finished.connect(
            lambda g, p, t: self.on_train_complete(canvas, g, p, t, tab_index))
        self.active_workers[tab_index] = self.train_worker
        self.train_worker.start()

    def on_train_complete(self, canvas, grid_data, point_data, title, tab_index):
        canvas.plot_geometric_boundary(grid_data, point_data, title)
        self.set_tab_enabled(tab_index, True)
        self.statusBar().showMessage("Training Complete. Geometric Level-Sets Rendered.", 5000)

    def handle_error(self, err_msg, tab_index):
        self.statusBar().showMessage(f"Error: {err_msg}")
        self.set_tab_enabled(tab_index, True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())