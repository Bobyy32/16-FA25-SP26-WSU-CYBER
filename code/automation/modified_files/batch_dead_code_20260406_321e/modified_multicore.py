def terminate(self):
    """Stop all workers."""
    if not self.join_signal.is_set():
        self.join_signal.set()
    
    # Give more time for graceful shutdown
    time.sleep(0.1)

    if self.main_worker_thread.is_alive():
        self.main_worker_thread.join(timeout=5)
    
    if self.threaded:
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5)
    else:
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)

        while not self.all_finished():
            time.sleep(0.001)
    
    # Clean up queues
    if self.queue.full():
        try:
            self.queue.get()
        except:
            pass
    self.queue.put(pickle.dumps(None, protocol=-1))
    time.sleep(0.1)