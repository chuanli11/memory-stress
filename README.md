Memory Capacity Stress Test
===


Goal: A synthetic training job that try to run a single iterm batch that eats all the GPU memory.

V100 (32GB) failed the test. (see log file V100.txt)

QuadroRTX8000 (48GB) passed the test. (see log file QuadroRTX8000.txt)

__example__

```bash
python keras_test.py > V100.txt 2>&1
python keras_test.py > QuadroRTX8000.txt 2>&1
```