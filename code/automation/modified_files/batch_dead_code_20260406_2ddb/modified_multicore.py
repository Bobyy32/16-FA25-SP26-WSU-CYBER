# Wait a bit longer after sending the signal
self.queue.put(pickle.dumps(None, protocol=-1))
time.sleep(0.1)  # Increase timeout to match worker wait times
while self.nb_workers_finished < self.nb_workers:
    time.sleep(0.005)
# Ensure all workers have exited before joining threads
while self.nb_workers_finished < self.nb_workers:
    time.sleep(0.001)