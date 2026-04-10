if self.queue.full():
    self.queue.get()
self.queue.put(pickle.dumps(None, protocol=-1))