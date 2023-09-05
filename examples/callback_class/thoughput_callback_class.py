from transformers import TrainerCallback
import time

class ThroughputCallback(TrainerCallback):
    def __init__(self, logging_steps):
        self.logging_steps = logging_steps
        self.start_time = time.time()
        self.step_times = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.logging_steps == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            self.step_times.append(elapsed_time)
            throughput = self.logging_steps / elapsed_time
            print(f"Throughput after {state.global_step} steps: {throughput:.2f} steps/second")
            self.start_time = current_time