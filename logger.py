import time

YELLOW = '\033[93m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

class Logger:
    _instance : 'Logger' = None
    info = '[INFO]'
    warning = '[WARNING]'
    error = '[ERROR]'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._logs = []
            cls._instance._init_time = time.time()
            cls._last_prc = 0
        return cls._instance

    def log(self, message: str, level: str = info):
        formatted = self._formate(message, level)
        self._logs.append(formatted)
        print(formatted)
    
    def log_prc(self, message: str, current: float,
                max: float, level: str = info):
        prc = int(100*current/max)
        if prc != self._last_prc:
            formatted = self._formate(f"{message} {prc}%", level)
            self._logs.append(formatted)
            print(formatted, end = '\r')
            self._last_prc = prc

    def log_prc_done(self, message: str, level: str = info):
        formatted = self._formate(message + f"{GREEN} 100%{RESET}", level)
        print(formatted)
        self._logs.append(formatted)

    def _formate(self, message: str, level: str = info) -> str:
        current_time = time.time()
        elapsed = current_time - self._init_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        if (level == Logger.warning):
            return f"[{formatted_time}]{YELLOW} {self.warning} {RESET}{message}"
        elif (level == Logger.error):
            return f"[{formatted_time}]{RED} {self.error} {message} {RESET}"
        return f"[{formatted_time}] {self.info} {message}                   "

    def get_logs(self):
        return self._logs.copy()

    def raise_error(self, message: str):
        raise InternalError(self._formate(message, self.error))
    
    def ask_for_continue(self, message: str):
        input(self._formate(f"{YELLOW}{message} Continue ? "
                            f"[Press ENTER or use ctrl C] {RESET}",
                             self.warning))


class InternalError(Exception):
    """Custom exception class for errors in this package."""
    def __init__(self, message: str):
        super().__init__(f"\n{RED}{message}{RESET}")