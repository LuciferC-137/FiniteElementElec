import time

YELLOW = '\033[93m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

class Logger:
    _instance : 'Logger' = None
    info = '[INFO]'
    warning = '[WARNING]'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._logs = []
            cls._instance._init_time = time.time()
        return cls._instance

    def log(self, message: str, level: str = info):
        current_time = time.time()
        elapsed = current_time - self._init_time
        formatted = f"[+{elapsed:.3f}s] {level} {message}"
        self._logs.append(formatted)
        print(formatted)
    
    def log_prc(self, message: str, current: float,
                max: float, level: str = info):
        current_time = time.time()
        elapsed = current_time - self._init_time
        formatted = f"[+{elapsed:.3f}s] {level} {message} {100*current/max:.1f}%"
        formatted = formatted + "                     "
        if (len(self._logs) > 0 and formatted != self._logs[-1]):
            self._logs.append(formatted)
            print(formatted, end = '\r')

    def log_prc_done(self, message: str, level: str = info):
        current_time = time.time()
        elapsed = current_time - self._init_time
        formatted = f"[+{elapsed:.3f}s] {level} {message} {GREEN}{100}%{RESET}"
        formatted = formatted + "                     "
        print(formatted)
        self._logs.append(formatted)

    def get_logs(self):
        return self._logs.copy()