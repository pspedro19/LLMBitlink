import time
from functools import wraps

def measure_time(histogram):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - start_time
            histogram.observe(elapsed)
            return result
        return wrapper
    return decorator