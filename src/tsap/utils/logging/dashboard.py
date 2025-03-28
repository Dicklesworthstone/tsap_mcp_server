"""
Terminal-based dashboard for TSAP using Textual.

This module provides a Text User Interface (TUI) dashboard for monitoring
TSAP operations, logs, and statistics in real-time. The dashboard uses
the Textual library to create a rich, interactive terminal interface.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

try:
    from textual.app import App
    from textual.widgets import Header, Footer, Static, DataTable, Log
    from textual.containers import Container, Horizontal, Grid
    from textual.screen import Screen
    from textual.message import Message
    from textual.binding import Binding
    TEXTUAL_AVAILABLE = True
except ImportError:
    # Create stub classes for typechecking when Textual is not available
    class App: 
        pass
    class Screen: 
        pass
    class Message: 
        pass
    TEXTUAL_AVAILABLE = False

from rich.text import Text
from rich.panel import Panel

from tsap.utils.logging import logger
from tsap.utils.metrics import get_all_metrics
from tsap.version import get_version_info
from tsap.performance_mode import get_performance_mode, describe_current_mode


class LogMonitor:
    """
    Monitor and buffer recent log messages.
    
    This class captures log messages and maintains a buffer of recent messages
    for display in the dashboard.
    
    Attributes:
        max_messages: Maximum number of messages to keep in the buffer
        messages: List of captured log messages
    """
    def __init__(self, max_messages: int = 1000) -> None:
        """
        Initialize a new log monitor.
        
        Args:
            max_messages: Maximum number of messages to keep in the buffer
        """
        self.max_messages = max_messages
        self.messages: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._install_handler()
    
    def _install_handler(self) -> None:
        """Install a handler to capture log messages."""
        import logging
        
        class CaptureHandler(logging.Handler):
            def __init__(self, callback: Callable[[Dict[str, Any]], None]) -> None:
                super().__init__()
                self.callback = callback
            
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    message = {
                        'timestamp': record.created,
                        'level': record.levelname,
                        'logger': record.name,
                        'component': getattr(record, 'component', 'unspecified'),
                        'message': self.format(record),
                        'exception': record.exc_text if record.exc_text else None
                    }
                    self.callback(message)
                except Exception:
                    self.handleError(record)
        
        # Create and install the handler
        handler = CaptureHandler(self.add_message)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Get the root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the buffer.
        
        Args:
            message: Log message to add
        """
        with self._lock:
            self.messages.append(message)
            
            # Trim buffer if necessary
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, count: Optional[int] = None, level: Optional[str] = None, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent log messages, optionally filtered.
        
        Args:
            count: Maximum number of messages to return
            level: Filter by log level
            component: Filter by component
            
        Returns:
            List of log messages
        """
        with self._lock:
            # Apply filters
            filtered = self.messages
            
            if level:
                filtered = [m for m in filtered if m['level'] == level]
            
            if component:
                filtered = [m for m in filtered if m['component'] == component]
            
            # Apply count limit
            if count is not None:
                filtered = filtered[-count:]
            
            return filtered


class MetricsCollector:
    """
    Collect and buffer system and operation metrics.
    
    This class periodically collects metrics from all registered metric
    collectors and maintains a history of metric values.
    
    Attributes:
        history_size: Maximum number of historical data points to keep
        update_interval: Interval between metric updates in seconds
        current_metrics: Most recent metric values
        metric_history: Historical metric values
    """
    def __init__(self, history_size: int = 100, update_interval: float = 1.0) -> None:
        """
        Initialize a new metrics collector.
        
        Args:
            history_size: Maximum number of historical data points to keep
            update_interval: Interval between metric updates in seconds
        """
        self.history_size = history_size
        self.update_interval = update_interval
        self.current_metrics: Dict[str, Dict[str, Any]] = {}
        self.metric_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """Start the metrics collection thread."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._thread.start()
    
    def stop(self) -> None:
        """Stop the metrics collection thread."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)
    
    def _collect_loop(self) -> None:
        """Metrics collection loop."""
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Sleep until next collection
                self._stop_event.wait(self.update_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
                # Sleep a bit after an error
                time.sleep(1.0)
    
    def _collect_metrics(self) -> None:
        """Collect current metrics."""
        metrics = get_all_metrics()
        
        with self._lock:
            # Update current metrics
            self.current_metrics = metrics
            
            # Add to history
            history_entry = {
                'timestamp': time.time(),
                'metrics': metrics
            }
            self.metric_history.append(history_entry)
            
            # Trim history if necessary
            if len(self.metric_history) > self.history_size:
                self.metric_history = self.metric_history[-self.history_size:]
    
    def get_current_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the most recent metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            return self.current_metrics.copy()
    
    def get_metric_history(self, metric_path: str, count: Optional[int] = None) -> List[float]:
        """
        Get historical values for a specific metric.
        
        Args:
            metric_path: Path to the metric (e.g., 'system.cpu.usage')
            count: Maximum number of data points to return
            
        Returns:
            List of metric values
        """
        with self._lock:
            # Parse the metric path
            parts = metric_path.split('.')
            
            # Extract values from history
            values = []
            for entry in self.metric_history:
                # Navigate the metrics dictionary
                value = entry['metrics']
                for part in parts:
                    if part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                if value is not None:
                    values.append(value)
            
            # Apply count limit
            if count is not None:
                values = values[-count:]
            
            return values


if TEXTUAL_AVAILABLE:
    class LogPanel(Container):
        """A panel displaying log messages."""
        DEFAULT_CSS = """
        LogPanel {
            height: 100%;
            border: round $primary;
            padding: 1;
        }
        
        LogPanel > .header {
            dock: top;
            height: 1;
            color: $text;
            background: $primary;
            text-align: center;
            width: 100%;
        }
        
        LogPanel > .log {
            height: 1fr;
            overflow-y: scroll;
        }
        
        LogPanel > .filters {
            dock: top;
            height: 1;
            margin-top: 1;
            margin-bottom: 1;
        }
        """
        
        def __init__(self, log_monitor: LogMonitor, name: Optional[str] = None) -> None:
            """
            Initialize a new log panel.
            
            Args:
                log_monitor: Monitor providing log messages
                name: Optional name for the panel
            """
            super().__init__(name=name)
            self.log_monitor = log_monitor
            self.level_filter: Optional[str] = None
            self.component_filter: Optional[str] = None
        
        def compose(self) -> None:
            """Compose the widget."""
            # Header
            yield Static("Logs", classes="header")
            
            # Filters
            filter_container = Horizontal(classes="filters")
            yield filter_container
            
            # Log display
            self.log_display = Log(highlight=True, markup=True)
            yield self.log_display
        
        def update_logs(self) -> None:
            """Update the log display with new messages."""
            messages = self.log_monitor.get_messages(
                count=100,
                level=self.level_filter,
                component=self.component_filter
            )
            
            # Clear the log display
            self.log_display.clear()
            
            # Add messages
            for message in messages:
                level = message['level']
                component = message['component']
                msg_text = message['message']
                
                # Format based on level
                if level == 'DEBUG':
                    self.log_display.write(f"[dim][blue][{level}][/blue] [{component}] {msg_text}[/dim]")
                elif level == 'INFO':
                    self.log_display.write(f"[blue][{level}][/blue] [{component}] {msg_text}")
                elif level == 'WARNING':
                    self.log_display.write(f"[yellow][{level}][/yellow] [{component}] {msg_text}")
                elif level == 'ERROR':
                    self.log_display.write(f"[red][{level}][/red] [{component}] {msg_text}")
                elif level == 'CRITICAL':
                    self.log_display.write(f"[bold red][{level}][/bold red] [{component}] {msg_text}")
                else:
                    self.log_display.write(f"[{level}] [{component}] {msg_text}")
        
        def set_level_filter(self, level: Optional[str]) -> None:
            """
            Set the log level filter.
            
            Args:
                level: Log level to filter by, or None for all levels
            """
            self.level_filter = level
            self.update_logs()
        
        def set_component_filter(self, component: Optional[str]) -> None:
            """
            Set the component filter.
            
            Args:
                component: Component to filter by, or None for all components
            """
            self.component_filter = component
            self.update_logs()


    class MetricsPanel(Container):
        """A panel displaying system metrics."""
        DEFAULT_CSS = """
        MetricsPanel {
            height: 100%;
            border: round $primary;
            padding: 1;
        }
        
        MetricsPanel > .header {
            dock: top;
            height: 1;
            color: $text;
            background: $primary;
            text-align: center;
            width: 100%;
        }
        
        MetricsPanel > .metrics-grid {
            height: 1fr;
            grid-size: 2;
            grid-gutter: 1;
            padding: 1;
        }
        
        MetricsPanel > .metrics-grid > .metric {
            border: solid $primary;
            padding: 1;
        }
        """
        
        def __init__(self, metrics_collector: MetricsCollector, name: Optional[str] = None) -> None:
            """
            Initialize a new metrics panel.
            
            Args:
                metrics_collector: Collector providing metric data
                name: Optional name for the panel
            """
            super().__init__(name=name)
            self.metrics_collector = metrics_collector
        
        def compose(self) -> None:
            """Compose the widget."""
            # Header
            yield Static("System Metrics", classes="header")
            
            # Metrics grid
            self.metrics_grid = Grid(classes="metrics-grid")
            yield self.metrics_grid
        
        def update_metrics(self) -> None:
            """Update the metrics display with new data."""
            metrics = self.metrics_collector.get_current_metrics()
            
            # Clear the grid
            self.metrics_grid.remove_children()
            
            # Add CPU metric
            cpu_panel = Panel(f"CPU: {metrics.get('system', {}).get('cpu_usage', 0):.1f}%", title="CPU Usage")
            cpu_container = Static(cpu_panel, classes="metric")
            self.metrics_grid.mount(cpu_container)
            
            # Add memory metric
            memory_used = metrics.get('system', {}).get('memory_used', 0)
            memory_total = metrics.get('system', {}).get('memory_total', 1)
            memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
            
            memory_panel = Panel(
                f"Memory: {memory_used / (1024 * 1024):.1f} MB / {memory_total / (1024 * 1024):.1f} MB ({memory_percent:.1f}%)",
                title="Memory Usage"
            )
            memory_container = Static(memory_panel, classes="metric")
            self.metrics_grid.mount(memory_container)
            
            # Add process count metric
            process_panel = Panel(
                f"Processes: {metrics.get('system', {}).get('process_count', 0)}",
                title="Process Count"
            )
            process_container = Static(process_panel, classes="metric")
            self.metrics_grid.mount(process_container)
            
            # Add uptime metric
            uptime = metrics.get('system', {}).get('uptime', 0)
            uptime_str = str(timedelta(seconds=uptime)).split('.')[0]  # Remove microseconds
            
            uptime_panel = Panel(
                f"Uptime: {uptime_str}",
                title="System Uptime"
            )
            uptime_container = Static(uptime_panel, classes="metric")
            self.metrics_grid.mount(uptime_container)


    class OperationsPanel(Container):
        """A panel displaying active and recent operations."""
        DEFAULT_CSS = """
        OperationsPanel {
            height: 100%;
            border: round $primary;
            padding: 1;
        }
        
        OperationsPanel > .header {
            dock: top;
            height: 1;
            color: $text;
            background: $primary;
            text-align: center;
            width: 100%;
        }
        
        OperationsPanel > .active-label, OperationsPanel > .recent-label {
            height: 1;
            margin-top: 1;
            margin-bottom: 1;
            text-align: left;
            width: 100%;
        }
        
        OperationsPanel > .active-table, OperationsPanel > .recent-table {
            height: 1fr;
            margin-bottom: 1;
        }
        """
        
        def __init__(self, name: Optional[str] = None) -> None:
            """
            Initialize a new operations panel.
            
            Args:
                name: Optional name for the panel
            """
            super().__init__(name=name)
            self.active_operations: List[Dict[str, Any]] = []
            self.recent_operations: List[Dict[str, Any]] = []
        
        def compose(self) -> None:
            """Compose the widget."""
            # Header
            yield Static("Operations", classes="header")
            
            # Active operations section
            yield Static("Active Operations", classes="active-label")
            
            self.active_table = DataTable(
                ("ID", "Operation", "Start Time", "Duration", "Progress"),
                classes="active-table"
            )
            yield self.active_table
            
            # Recent operations section
            yield Static("Recent Operations", classes="recent-label")
            
            self.recent_table = DataTable(
                ("ID", "Operation", "Status", "Duration"),
                classes="recent-table"
            )
            yield self.recent_table
        
        def update_operations(self, active_operations: List[Dict[str, Any]], recent_operations: List[Dict[str, Any]]) -> None:
            """
            Update the operations display with new data.
            
            Args:
                active_operations: List of active operations
                recent_operations: List of recent operations
            """
            self.active_operations = active_operations
            self.recent_operations = recent_operations
            
            # Update active operations table
            self.active_table.clear()
            for op in active_operations:
                op_id = op.get('id', 'Unknown')
                op_name = op.get('name', 'Unknown')
                start_time = datetime.fromtimestamp(op.get('start_time', 0)).strftime("%H:%M:%S")
                duration = str(timedelta(seconds=op.get('duration', 0))).split('.')[0]
                progress = f"{op.get('progress', 0):.1f}%"
                
                self.active_table.add_row(op_id, op_name, start_time, duration, progress)
            
            # Update recent operations table
            self.recent_table.clear()
            for op in recent_operations:
                op_id = op.get('id', 'Unknown')
                op_name = op.get('name', 'Unknown')
                status = op.get('status', 'Unknown')
                duration = str(timedelta(seconds=op.get('duration', 0))).split('.')[0]
                
                # Format status with color
                if status == 'completed':
                    status_text = Text(status, style="green")
                elif status == 'failed':
                    status_text = Text(status, style="red")
                elif status == 'canceled':
                    status_text = Text(status, style="yellow")
                else:
                    status_text = Text(status)
                
                self.recent_table.add_row(op_id, op_name, status_text, duration)


    class StatusPanel(Container):
        """A panel displaying system status information."""
        DEFAULT_CSS = """
        StatusPanel {
            height: 100%;
            border: round $primary;
            padding: 1;
        }
        
        StatusPanel > .header {
            dock: top;
            height: 1;
            color: $text;
            background: $primary;
            text-align: center;
            width: 100%;
        }
        
        StatusPanel > .status-grid {
            height: 1fr;
            grid-size: 2;
            grid-gutter: 1;
            padding: 1;
        }
        
        StatusPanel > .status-grid > .status-item {
            border: solid $primary;
            padding: 1;
        }
        """
        
        def __init__(self, name: Optional[str] = None) -> None:
            """
            Initialize a new status panel.
            
            Args:
                name: Optional name for the panel
            """
            super().__init__(name=name)
        
        def compose(self) -> None:
            """Compose the widget."""
            # Header
            yield Static("System Status", classes="header")
            
            # Status grid
            self.status_grid = Grid(classes="status-grid")
            yield self.status_grid
        
        def update_status(self) -> None:
            """Update the status display with new data."""
            # Get version information
            version_info = get_version_info()
            version = version_info.get('version', 'Unknown')
            protocol_version = version_info.get('protocol_version', 'Unknown')
            
            # Get performance mode
            performance_mode = get_performance_mode()
            mode_info = describe_current_mode()
            
            # Clear the grid
            self.status_grid.remove_children()
            
            # Add version panel
            version_panel = Panel(
                f"Version: {version}\nProtocol: {protocol_version}",
                title="Version Info"
            )
            version_container = Static(version_panel, classes="status-item")
            self.status_grid.mount(version_container)
            
            # Add performance mode panel
            mode_panel = Panel(
                f"Mode: {performance_mode.upper()}\n"
                f"Depth: {mode_info.get('search_depth', 'Unknown')}\n"
                f"Concurrency: {mode_info.get('max_concurrency', 'Unknown')}",
                title="Performance Mode"
            )
            mode_container = Static(mode_panel, classes="status-item")
            self.status_grid.mount(mode_container)
            
            # Add server status panel
            server_panel = Panel(
                "Status: Running\n"
                f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                title="Server Status"
            )
            server_container = Static(server_panel, classes="status-item")
            self.status_grid.mount(server_container)
            
            # Add plugin status panel
            plugin_panel = Panel(
                "Active Plugins: 5\n"
                "Total Plugins: 8",
                title="Plugin Status"
            )
            plugin_container = Static(plugin_panel, classes="status-item")
            self.status_grid.mount(plugin_container)


    class TsapDashboard(App):
        """
        TSAP Terminal Dashboard App.
        
        This is the main Textual app for the TSAP dashboard.
        """
        TITLE = "TSAP Dashboard"
        CSS = """
        Screen {
            background: $surface;
        }
        
        #dashboard {
            layout: grid;
            grid-size: 3 2;
            grid-gutter: 1;
            padding: 1;
        }
        
        #log-panel {
            row-span: 2;
        }
        """
        
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "refresh", "Refresh"),
            Binding("l", "toggle_logs", "Toggle Logs"),
            Binding("m", "toggle_metrics", "Toggle Metrics"),
            Binding("o", "toggle_operations", "Toggle Operations"),
            Binding("s", "toggle_status", "Toggle Status")
        ]
        
        def __init__(self) -> None:
            """Initialize the dashboard app."""
            super().__init__()
            self.log_monitor = LogMonitor()
            self.metrics_collector = MetricsCollector()
        
        def compose(self) -> None:
            """Compose the app layout."""
            # Create the header and footer
            yield Header()
            yield Footer()
            
            # Create the main container
            dashboard = Container(id="dashboard")
            
            # Create panels
            log_panel = LogPanel(self.log_monitor, id="log-panel")
            metrics_panel = MetricsPanel(self.metrics_collector)
            operations_panel = OperationsPanel()
            status_panel = StatusPanel()
            
            # Add panels to the dashboard
            dashboard.mount(log_panel)
            dashboard.mount(metrics_panel)
            dashboard.mount(operations_panel)
            dashboard.mount(status_panel)
            
            yield dashboard
        
        def on_mount(self) -> None:
            """Handle the mount event."""
            # Start metrics collection
            self.metrics_collector.start()
            
            # Set up periodic refresh
            self.set_interval(1.0, self.refresh_dashboard)
        
        def refresh_dashboard(self) -> None:
            """Refresh all dashboard panels."""
            # Update logs
            log_panel = self.query_one("#log-panel", LogPanel)
            log_panel.update_logs()
            
            # Update metrics
            metrics_panel = self.query_one(MetricsPanel)
            metrics_panel.update_metrics()
            
            # Update operations (placeholder data)
            operations_panel = self.query_one(OperationsPanel)
            active_ops = [
                {
                    'id': '1',
                    'name': 'ripgrep_search',
                    'start_time': time.time() - 30,
                    'duration': 30,
                    'progress': 75
                },
                {
                    'id': '2',
                    'name': 'code_analyzer',
                    'start_time': time.time() - 10,
                    'duration': 10,
                    'progress': 40
                }
            ]
            recent_ops = [
                {
                    'id': '0',
                    'name': 'parallel_search',
                    'status': 'completed',
                    'duration': 15
                }
            ]
            operations_panel.update_operations(active_ops, recent_ops)
            
            # Update status
            status_panel = self.query_one(StatusPanel)
            status_panel.update_status()
        
        def action_refresh(self) -> None:
            """Handle the refresh action."""
            self.refresh_dashboard()
        
        def action_toggle_logs(self) -> None:
            """Toggle the visibility of the logs panel."""
            log_panel = self.query_one("#log-panel", LogPanel)
            log_panel.toggle_class("hidden")
        
        def action_toggle_metrics(self) -> None:
            """Toggle the visibility of the metrics panel."""
            metrics_panel = self.query_one(MetricsPanel)
            metrics_panel.toggle_class("hidden")
        
        def action_toggle_operations(self) -> None:
            """Toggle the visibility of the operations panel."""
            operations_panel = self.query_one(OperationsPanel)
            operations_panel.toggle_class("hidden")
        
        def action_toggle_status(self) -> None:
            """Toggle the visibility of the status panel."""
            status_panel = self.query_one(StatusPanel)
            status_panel.toggle_class("hidden")
        
        def on_unmount(self) -> None:
            """Handle the unmount event."""
            # Stop metrics collection
            self.metrics_collector.stop()


def is_dashboard_available() -> bool:
    """
    Check if the dashboard is available.
    
    The dashboard requires the Textual library to be installed.
    
    Returns:
        True if the dashboard is available, False otherwise
    """
    return TEXTUAL_AVAILABLE


def start_dashboard() -> None:
    """
    Start the TSAP dashboard.
    
    This function launches the Textual-based dashboard in the current terminal.
    """
    if not TEXTUAL_AVAILABLE:
        logger.error("Cannot start dashboard: Textual library is not installed")
        print("Error: Textual library is not installed. Please install it using:")
        print("  pip install textual")
        return
    
    app = TsapDashboard()
    app.run()


def launch_dashboard_in_thread() -> threading.Thread:
    """
    Launch the dashboard in a separate thread.
    
    Returns:
        Thread running the dashboard
    """
    if not TEXTUAL_AVAILABLE:
        logger.error("Cannot launch dashboard: Textual library is not installed")
        raise ImportError("Textual library is not installed. Please install it using: pip install textual")
    
    def _run_dashboard() -> None:
        app = TsapDashboard()
        app.run()
    
    thread = threading.Thread(target=_run_dashboard, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    # If run directly, start the dashboard
    start_dashboard()