import time
from app.services.classifier import classify

sample_clause = (
    "Either party may terminate this Agreement upon thirty (30) days written "
    "notice to the other party if the other party materially breaches any "
    "provision of this Agreement and fails to cure such breach within the "
    "notice period."
)

def time_one_by_one(texts):
    start = time.perf_counter()
    for t in texts:
        classify([t])  # one clause at a time
    return time.perf_counter() - start

def time_batched(texts, batch_size=16):
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        classify(texts[i:i+batch_size])  # batch of clauses at once
    return time.perf_counter() - start

texts = [sample_clause] * 100

print("Warming up model...")
classify([sample_clause])  # first call loads the model, exclude from timing

print("\nTiming one-by-one classification...")
t1 = time_one_by_one(texts)

print("Timing batched classification...")
t2 = time_batched(texts)

speedup = t1 / t2
print(f"\nOne-by-one: {t1:.2f}s ({len(texts)/t1:.1f} clauses/sec)")
print(f"Batched:    {t2:.2f}s ({len(texts)/t2:.1f} clauses/sec)")
print(f"Speedup:    {speedup:.2f}x")