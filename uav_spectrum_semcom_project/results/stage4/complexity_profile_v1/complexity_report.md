# Stage-4 complexity profile

CPU, one thread; latency is local-machine evidence and not UAV-board latency.

| Component | Params | FP32 parameter bytes | Linear MACs | Median / p95 latency |
|---|---:|---:|---:|---:|
| C1 head, one scene | 1740 | 6960 | 1664 | 0.4238 / 0.7075 ms |
| C2 network, 8 candidates, one seed | 5697 | 22788 | 44800 | 0.0695 / 0.1201 ms |
| C2 network, 8 candidates, five seeds | 28485 | 113940 | 224000 | 0.3502 / 0.5706 ms |

> one multiply-accumulate per Linear weight; activations, softmax, quantization, Python control, detector, codec, and link simulation excluded
