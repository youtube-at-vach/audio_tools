class SessionManager:
    """
    Manages the current measurement session state.
    """
    def __init__(self):
        self.current_module = None
        self.is_running = False
        self.results = []

    def set_module(self, module):
        """Sets the current measurement module."""
        self.current_module = module

    def start_measurement(self):
        """Starts the measurement process."""
        if self.current_module:
            self.is_running = True
            # Logic to start module
            pass

    def stop_measurement(self):
        """Stops the measurement process."""
        self.is_running = False
        # Logic to stop module
        pass
