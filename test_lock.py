from filelock import Timeout, FileLock
import time

# First, establish lock so only one instance of build_algorithm_history will run at a time
lock = FileLock('algorithm_history.lock')
with lock:
    open('arbitrary_lockfile.lock', 'w').close()
lock.acquire()
open('arbitrary_lockfile.lock', 'w').close()    # this step will block until the lock is released
print('start')
time.sleep(10)
raise RuntimeError('intentional')
lock.release()
print('release')
time.sleep(10)
print('stop')
