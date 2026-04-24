if self.threaded:
    for worker in self.workers:
        if worker.is_alive():
            worker.join()  # Waits for graceful shutdown
else:
    for worker in self.workers:
        if worker.is_alive():
            worker.terminate()
            worker.join()  # May not wait long enough

    # wait until all workers are fully terminated
    while not self.all_finished():
        time.sleep(0.001)