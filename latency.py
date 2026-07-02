import time

from predict import predict

IMAGE = "dataset/test/screen/20260702_155155.jpg"

# Warm-up
for _ in range(10):
    predict(IMAGE)

runs = 100

start = time.perf_counter()

for _ in range(runs):
    predict(IMAGE)

end = time.perf_counter()

avg_ms = (end - start) * 1000 / runs

print(f"Average Latency: {avg_ms:.2f} ms/image")