f} -> {s.end:.2f}] {s.speaker}: {s.text}")

print(sorted({s.speaker for s in res.segments}))
