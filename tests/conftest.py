"""测试环境限制：避免 BLAS/OpenMP 为每个进程创建大量线程。

容器可能对 PID/线程数有较低配额；测试只做小规模数值运算，单线程足够，
同时让并发/时序测试更稳定。
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
