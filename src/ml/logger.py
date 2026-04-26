import os
import threading
import sys
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import dash
    from dash import dcc, html, Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class TrainingLogger:
    def __init__(self, mode="gui", config_name="training"):
        self.mode = mode if mode != "tui" else "gui"  # Map tui to gui as requested
        self.config_name = config_name
        self.metrics = {
            "loss": [],
            "time_per_batch": [],
            "inference_time": [],
            "train_loss_epoch": [],
            "test_loss_epoch": [],
            "epoch": []
        }
        
        self.pbar = None
        self.dash_app = None
        
        if self.mode == "gui":
            if not DASH_AVAILABLE:
                print("Dash/Plotly not installed. Falling back to plain logging.")
                self.mode = "plain"
            else:
                self._setup_dash()

    def _setup_dash(self):
        """Initialize and start the Dash server in a background thread."""
        self.dash_app = dash.Dash(__name__)
        
        # Layout
        self.dash_app.layout = html.Div([
            html.H1(f"Training Monitor: {self.config_name}"),
            dcc.Graph(id='live-plots'),
            dcc.Interval(
                id='interval-component',
                interval=1*1000, # in milliseconds
                n_intervals=0
            ),
            html.Div(id='stats-panel', style={'fontSize': '20px', 'marginTop': '20px'})
        ])

        @self.dash_app.callback(
            [Output('live-plots', 'figure'), Output('stats-panel', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graph(n):
            # Create subplots: 2x2
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Batch Loss", "Batch Time (s)", "Inference Time (ms)", "Epoch Losses")
            )

            # Batch Loss
            fig.add_trace(go.Scatter(y=self.metrics["loss"], mode='lines', name='Loss'), row=1, col=1)
            # Batch Time
            fig.add_trace(go.Scatter(y=self.metrics["time_per_batch"], mode='lines', name='Batch Time'), row=1, col=2)
            # Inf Time
            fig.add_trace(go.Scatter(y=self.metrics["inference_time"], mode='lines', name='Inf Time'), row=2, col=1)
            # Epoch Losses
            fig.add_trace(go.Scatter(x=self.metrics["epoch"], y=self.metrics["train_loss_epoch"], 
                                     mode='lines+markers', name='Train Loss'), row=2, col=2)
            fig.add_trace(go.Scatter(x=self.metrics["epoch"], y=self.metrics["test_loss_epoch"], 
                                     mode='lines+markers', name='Test Loss'), row=2, col=2)

            fig.update_layout(height=800, title_text="Real-time Training Metrics", showlegend=False)
            
            stats = f"Epochs: {len(self.metrics['epoch'])} | Last Batch Loss: {self.metrics['loss'][-1]:.6f}" if self.metrics["loss"] else "Initializing..."
            return fig, stats

        # Run Dash in a separate thread to avoid blocking training
        threading.Thread(target=lambda: self.dash_app.run(debug=False, use_reloader=False, port=8080), daemon=True).start()
        print("GUI Dashboard started at http://127.0.0.1:8080")

    def set_progress_bar(self, pbar):
        self.pbar = pbar

    def log(self, event, **kwargs):
        """Single callback for all logging events."""
        if event == "batch":
            loss = kwargs.get("loss")
            batch_time = kwargs.get("batch_time")
            inf_time = kwargs.get("inf_time", 0) * 1000  # to ms
            lr = kwargs.get("lr", "N/A")

            self.metrics["loss"].append(loss)
            self.metrics["time_per_batch"].append(batch_time)
            self.metrics["inference_time"].append(inf_time)

            if self.mode == "plain":
                if len(self.metrics["loss"]) % 100 == 0:
                    print(f"Batch {len(self.metrics['loss'])} - Loss: {loss:.4f}, Inf Time: {inf_time:.2f}ms")

            if self.pbar:
                self.pbar.set_postfix({"loss": f"{loss:.4f}", "inf_t": f"{inf_time:.2f}ms"})

        elif event == "epoch":
            epoch = kwargs.get("epoch")
            train_loss = kwargs.get("train_loss")
            test_loss = kwargs.get("test_loss")
            msg = f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
            
            self.metrics["epoch"].append(epoch)
            self.metrics["train_loss_epoch"].append(train_loss)
            self.metrics["test_loss_epoch"].append(test_loss)
            
            # Always print epoch results to console as requested
            print(msg)

    def save_final(self):
        # Plotly doesn't have a simple 'save image' without kaleido, 
        # so we just notify the user.
        if self.mode == "gui":
            print("Training complete. You can view final results on the dashboard at http://127.0.0.1:8050")
