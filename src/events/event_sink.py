import os
import json
import threading

class EventSink:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.buffer = []
        self.lock = threading.Lock()

    def write(self, event):
        with self.lock:
            self.buffer.append(event)

    def flush(self):
        with self.lock:
            with open(self.path, 'w') as f:
                for e in self.buffer:
                    f.write(json.dumps(e) + '\n')
