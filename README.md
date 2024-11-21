# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


---

## Task 3.1 and 3.2: Diagnostics Output

```
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (163)

================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (163)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        # raise NotImplementedError("Need to implement for Task 3.1")        |
                                                                             |
        out_size = len(out)                                                  |
                                                                             |
        if np.array_equal(out_strides, in_strides) and np.array_equal(       |
            out_shape, in_shape                                              |
        ):                                                                   |
            for i in prange(out_size):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   |
        else:                                                                |
            for i in prange(out_size):---------------------------------------| #3
                out_index = np.zeros(len(out_shape), dtype=np.int32)---------| #0
                in_index = np.zeros(len(in_shape), dtype=np.int32)-----------| #1
                                                                             |
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                                                                             |
                out_pos = index_to_position(out_index, out_strides)          |
                in_pos = index_to_position(in_index, in_strides)             |
                                                                             |
                out[out_pos] = fn(in_storage[in_pos])                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (183)
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (184)
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (220)

================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (220)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # TODO: Implement for Task 3.1.                                    |
        # raise NotImplementedError("Need to implement for Task 3.1")      |
                                                                           |
        out_size = len(out)                                                |
                                                                           |
        if (                                                               |
            np.array_equal(out_strides, a_strides)                         |
            and np.array_equal(out_strides, b_strides)                     |
            and np.array_equal(out_shape, a_shape)                         |
            and np.array_equal(out_shape, b_shape)                         |
        ):                                                                 |
            for i in prange(out_size):-------------------------------------| #7
                out[i] += fn(a_storage[i], b_storage[i])                   |
        else:                                                              |
            for i in prange(out_size):-------------------------------------| #8
                out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #4
                a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #5
                b_index = np.zeros(len(b_shape), dtype=np.int32)-----------| #6
                                                                           |
                to_index(i, out_shape, out_index)                          |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                                                                           |
                # flat position                                            |
                out_pos = index_to_position(out_index, out_strides)        |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                                                                           |
                out[out_pos] += fn(a_storage[a_pos], b_storage[b_pos])     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4, #5, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial)
   +--6 (serial)



Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (246)
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (247)
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (248)
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (285)

================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (285)
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        reduce_dim: int,                                                 |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
                                                                         |
        out_size = len(out)                                              |
        reduce_size = a_shape[reduce_dim]                                |
                                                                         |
        for i in prange(out_size):---------------------------------------| #10
            out_index = np.zeros(len(out_shape), dtype=np.int32)---------| #9
            to_index(i, out_shape, out_index)                            |
                                                                         |
            out_pos = index_to_position(out_index, out_strides)          |
                                                                         |
            reduction_val = out[out_pos]                                 |
            for j in range(reduce_size):                                 |
                out_index[reduce_dim] = j                                |
                a_pos = index_to_position(out_index, a_strides)          |
                reduction_val = fn(reduction_val, a_storage[a_pos])      |
                                                                         |
            out[out_pos] = reduction_val                                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (301)
is hoisted out of the parallel loop labelled #10 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (317)

================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/EmmaWang/Desktop/MLE/workplace/mod3-imyixinw/minitorch/fast_ops.py (317)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    # raise NotImplementedError("Need to implement for Task 3.2")                         |
                                                                                          |
    batch_size = out_shape[0]                                                             |
    m, n = out_shape[-2], out_shape[-1]                                                   |
    k = a_shape[-1]  # inner dimension shared by a and b                                  |
                                                                                          |
    for batch in prange(batch_size):------------------------------------------------------| #11
        a_batch_offset = batch * a_batch_stride                                           |
        b_batch_offset = batch * b_batch_stride                                           |
        out_batch_offset = batch * out_strides[0] if len(out_shape) > 2 else 0            |
                                                                                          |
        for i in range(m):  # rows                                                        |
            for j in range(n):  # columns                                                 |
                out_pos = out_batch_offset + i * out_strides[-2] + j * out_strides[-1]    |
                out[out_pos] = 0                                                          |
                                                                                          |
                for p in range(k):  # dot product                                         |
                    a_pos = a_batch_offset + i * a_strides[-2] + p * a_strides[-1]        |
                    b_pos = b_batch_offset + p * b_strides[-2] + j * b_strides[-1]        |
                    out[out_pos] += a_storage[a_pos] * b_storage[b_pos]                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Task 5 - GPU
For PTS = 50, HIDDEN = 100, and RATE = 0.05

### simple
```
Epoch  0  loss  6.742519406905529 correct 42
Epoch  10  loss  2.3670620497649937 correct 48
Epoch  20  loss  0.9772114142038978 correct 50
Epoch  30  loss  1.55740301614055 correct 49
Epoch  40  loss  0.7915650332133963 correct 49
Epoch  50  loss  1.4864985224583944 correct 50
Epoch  60  loss  0.39448778101934584 correct 50
Epoch  70  loss  0.8382297956346229 correct 49
Epoch  80  loss  1.6980054850686526 correct 50
Epoch  90  loss  0.5950492907993452 correct 49
Epoch  100  loss  0.16265572544011164 correct 50
Epoch  110  loss  0.38595931486038487 correct 50
Epoch  120  loss  0.7044030528689083 correct 49
Epoch  130  loss  0.30760204870293306 correct 50
Epoch  140  loss  0.19859839414969507 correct 49
Epoch  150  loss  0.2635160315418787 correct 50
Epoch  160  loss  1.5841328320143644 correct 50
Epoch  170  loss  0.46204979368974086 correct 50
Epoch  180  loss  0.46615941100266034 correct 49
Epoch  190  loss  0.2502466260079963 correct 50
Epoch  200  loss  1.0749506668965838 correct 50
Epoch  210  loss  0.13502436597164522 correct 50
Epoch  220  loss  0.9698044713775705 correct 50
Epoch  230  loss  0.0915974990949531 correct 50
Epoch  240  loss  0.025119042779842747 correct 50
Epoch  250  loss  0.7489758747255436 correct 50
Epoch  260  loss  0.187362357524778 correct 50
Epoch  270  loss  0.8296330106011377 correct 50
Epoch  280  loss  0.5720266323482275 correct 49
Epoch  290  loss  0.4490642710918149 correct 49
Epoch  300  loss  0.011672697283942721 correct 50
Epoch  310  loss  0.14202214358851625 correct 50
Epoch  320  loss  0.44668938082240406 correct 50
Epoch  330  loss  0.004637350378096153 correct 50
Epoch  340  loss  0.0959212137123766 correct 50
Epoch  350  loss  0.7865632007255254 correct 50
Epoch  360  loss  0.7927630765677541 correct 50
Epoch  370  loss  0.04457684196148353 correct 50
Epoch  380  loss  0.6135203409851095 correct 49
Epoch  390  loss  0.2031529558312831 correct 50
Epoch  400  loss  0.16655623328104552 correct 50
Epoch  410  loss  0.00884312223247446 correct 50
Epoch  420  loss  0.5192192352014627 correct 50
Epoch  430  loss  0.14122179233066404 correct 50
Epoch  440  loss  0.32306838880055005 correct 50
Epoch  450  loss  0.21694471645471178 correct 50
Epoch  460  loss  0.6792836140609121 correct 50
Epoch  470  loss  0.2583017898719253 correct 50
Epoch  480  loss  0.17751479746747373 correct 50
Epoch  490  loss  0.004996543739903875 correct 50
Epoch  500  loss  0.037448615562044284 correct 50
Time per epoch (s): 1.6045570120811463
```

### xor
```
Epoch  0  loss  7.736814839910217 correct 22
Epoch  10  loss  4.525981632591744 correct 32
Epoch  20  loss  6.0249310982003275 correct 42
Epoch  30  loss  2.701662788850274 correct 46
Epoch  40  loss  3.4124635474464577 correct 46
Epoch  50  loss  3.9215830137540353 correct 47
Epoch  60  loss  2.6533524955616556 correct 48
Epoch  70  loss  1.1684121393922324 correct 47
Epoch  80  loss  3.515425340557946 correct 47
Epoch  90  loss  0.8533128621040946 correct 48
Epoch  100  loss  3.0509431834397027 correct 47
Epoch  110  loss  2.4402111903439048 correct 48
Epoch  120  loss  2.6117284782164627 correct 48
Epoch  130  loss  2.218404338325043 correct 48
Epoch  140  loss  2.468874283916258 correct 48
Epoch  150  loss  1.1265462452631947 correct 48
Epoch  160  loss  1.7488266680544047 correct 49
Epoch  170  loss  0.8387874070578755 correct 49
Epoch  180  loss  0.9810379339211807 correct 48
Epoch  190  loss  2.061887235790256 correct 49
Epoch  200  loss  0.8126397873425846 correct 49
Epoch  210  loss  0.5381296984896498 correct 48
Epoch  220  loss  1.7706649501618918 correct 49
Epoch  230  loss  1.2187097050754727 correct 50
Epoch  240  loss  1.0556135395683417 correct 49
Epoch  250  loss  0.34625745829341065 correct 50
Epoch  260  loss  0.2935886701666934 correct 49
Epoch  270  loss  1.0424749336167791 correct 49
Epoch  280  loss  0.22725261935867153 correct 50
Epoch  290  loss  1.5852344760787367 correct 49
Epoch  300  loss  0.24332816411306218 correct 49
Epoch  310  loss  0.5007181522431411 correct 49
Epoch  320  loss  0.06893388958330676 correct 47
Epoch  330  loss  0.6797406180980818 correct 49
Epoch  340  loss  0.6407809851212082 correct 50
Epoch  350  loss  0.32264835866265973 correct 48
Epoch  360  loss  1.2670231336091256 correct 50
Epoch  370  loss  1.0832867503868635 correct 50
Epoch  380  loss  0.7934961213632492 correct 50
Epoch  390  loss  2.0427658879175703 correct 48
Epoch  400  loss  1.5124257627438003 correct 48
Epoch  410  loss  1.0592724196931664 correct 49
Epoch  420  loss  1.1170199378094732 correct 48
Epoch  430  loss  0.15689921748838587 correct 50
Epoch  440  loss  0.20502464723099628 correct 50
Epoch  450  loss  0.6112209565416968 correct 50
Epoch  460  loss  0.6022874582240592 correct 50
Epoch  470  loss  0.15307988887659754 correct 49
Epoch  480  loss  1.7680465909185827 correct 48
Epoch  490  loss  0.08846736683762922 correct 48
Epoch  500  loss  0.13020465322785602 correct 50
Time per epoch (s): 1.6160142564773559
```

### split
```
Epoch  0  loss  9.738907896937185 correct 31
Epoch  10  loss  5.561375632177957 correct 42
Epoch  20  loss  3.799060342154119 correct 47
Epoch  30  loss  2.6562400312852854 correct 45
Epoch  40  loss  2.2035079746380264 correct 49
Epoch  50  loss  2.4818688493775047 correct 50
Epoch  60  loss  1.4962677578466728 correct 49
Epoch  70  loss  1.76755877981879 correct 50
Epoch  80  loss  0.8245723300692923 correct 50
Epoch  90  loss  0.415417292935137 correct 48
Epoch  100  loss  2.151569156102277 correct 48
Epoch  110  loss  1.1835309565280951 correct 50
Epoch  120  loss  1.2182624000916444 correct 50
Epoch  130  loss  0.6845530546016427 correct 50
Epoch  140  loss  0.7481809577906366 correct 50
Epoch  150  loss  0.9595017337772371 correct 50
Epoch  160  loss  0.3583326872863583 correct 50
Epoch  170  loss  0.2534621207015807 correct 50
Epoch  180  loss  1.1661757333082183 correct 50
Epoch  190  loss  1.007711768872081 correct 50
Epoch  200  loss  1.0911615101804208 correct 49
Epoch  210  loss  0.665414694013984 correct 50
Epoch  220  loss  0.18476160613271728 correct 50
Epoch  230  loss  0.6975359240181984 correct 50
Epoch  240  loss  0.14285687070929162 correct 50
Epoch  250  loss  0.4015859889739782 correct 50
Epoch  260  loss  0.6043101713120733 correct 50
Epoch  270  loss  0.38773336394277125 correct 50
Epoch  280  loss  1.0333994836619698 correct 49
Epoch  290  loss  0.886414274528872 correct 50
Epoch  300  loss  0.3712657195206649 correct 50
Epoch  310  loss  0.8169856608211289 correct 50
Epoch  320  loss  0.9467594047489636 correct 50
Epoch  330  loss  0.3529952871741907 correct 50
Epoch  340  loss  0.5427234882477671 correct 50
Epoch  350  loss  0.5546708619017467 correct 50
Epoch  360  loss  0.09395681847287243 correct 50
Epoch  370  loss  0.46382209248773704 correct 50
Epoch  380  loss  0.790576433199924 correct 50
Epoch  390  loss  0.06799231093533721 correct 50
Epoch  400  loss  0.016416125495727914 correct 50
Epoch  410  loss  0.07832616556822669 correct 50
Epoch  420  loss  0.7519676277517436 correct 50
Epoch  430  loss  0.1073726560574696 correct 50
Epoch  440  loss  0.04655211245441959 correct 49
Epoch  450  loss  0.04037958742929973 correct 50
Epoch  460  loss  0.03691816139066906 correct 50
Epoch  470  loss  0.11702109437486795 correct 50
Epoch  480  loss  0.957258551763546 correct 50
Epoch  490  loss  0.6371099039812038 correct 50
Epoch  500  loss  0.6152783965033096 correct 50
Time per epoch (s): 1.610368073940277
```


## Task 5 - CPU
For PTS = 50, HIDDEN = 100, and RATE = 0.05

### simple
```
Epoch  0  loss  7.06956554577408 correct 40
Epoch  10  loss  1.8256997842996656 correct 49
Epoch  20  loss  1.5481083359786243 correct 50
Epoch  30  loss  0.44932146125485867 correct 50
Epoch  40  loss  0.28236796088643124 correct 50
Epoch  50  loss  0.6582568285055388 correct 49
Epoch  60  loss  0.48036402440324355 correct 50
Epoch  70  loss  0.2541516456128164 correct 50
Epoch  80  loss  0.5909818132464881 correct 50
Epoch  90  loss  0.1952862416406868 correct 50
Epoch  100  loss  0.3797943312410518 correct 50
Epoch  110  loss  0.586543168499006 correct 50
Epoch  120  loss  0.773109976602183 correct 50
Epoch  130  loss  0.04660494079143406 correct 50
Epoch  140  loss  0.7459383066867105 correct 50
Epoch  150  loss  0.7906260330359174 correct 50
Epoch  160  loss  0.01672966344615856 correct 50
Epoch  170  loss  0.7094896604363997 correct 50
Epoch  180  loss  0.49923648119999875 correct 50
Epoch  190  loss  0.013146686788715872 correct 50
Epoch  200  loss  0.22410193157564406 correct 50
Epoch  210  loss  0.16914127394429163 correct 50
Epoch  220  loss  0.33719850039338817 correct 50
Epoch  230  loss  0.009534645123533535 correct 50
Epoch  240  loss  0.008517286491058105 correct 50
Epoch  250  loss  0.01345519402015432 correct 50
Epoch  260  loss  0.12172509906036988 correct 50
Epoch  270  loss  0.31002678846801374 correct 50
Epoch  280  loss  0.16526108576390686 correct 50
Epoch  290  loss  0.007744155971597701 correct 50
Epoch  300  loss  0.0025263479033815887 correct 50
Epoch  310  loss  0.6444573651674091 correct 50
Epoch  320  loss  0.003080404140182459 correct 50
Epoch  330  loss  0.020798581951595122 correct 50
Epoch  340  loss  0.0009676920007757664 correct 50
Epoch  350  loss  0.4293890182793752 correct 50
Epoch  360  loss  0.019776205823207986 correct 50
Epoch  370  loss  0.32880332978634785 correct 50
Epoch  380  loss  0.38724418231687374 correct 50
Epoch  390  loss  0.4270759714651376 correct 50
Epoch  400  loss  0.375424127751212 correct 50
Epoch  410  loss  0.0019873644463063657 correct 50
Epoch  420  loss  0.0065743009498458686 correct 50
Epoch  430  loss  0.037196287977483616 correct 50
Epoch  440  loss  0.06440286482093611 correct 50
Epoch  450  loss  0.041883054054761344 correct 50
Epoch  460  loss  0.0014889973326700284 correct 50
Epoch  470  loss  0.12060444516766076 correct 50
Epoch  480  loss  0.4008062810559275 correct 50
Epoch  490  loss  0.16310264471267005 correct 50
Epoch  500  loss  0.0015558016106423302 correct 50
Time per epoch (s): 0.16352708768844604
```

### xor
```
Epoch  0  loss  6.20039102914655 correct 32
Epoch  10  loss  5.6191527652406155 correct 41
Epoch  20  loss  5.418882887675495 correct 39
Epoch  30  loss  3.4250955417209736 correct 38
Epoch  40  loss  4.387775340297341 correct 40
Epoch  50  loss  4.179115242440993 correct 44
Epoch  60  loss  1.906140726329996 correct 43
Epoch  70  loss  5.763568659141893 correct 44
Epoch  80  loss  3.3269683309834055 correct 47
Epoch  90  loss  3.6327711885611516 correct 46
Epoch  100  loss  3.5148610944254446 correct 42
Epoch  110  loss  3.471571246880372 correct 44
Epoch  120  loss  1.70885747294066 correct 47
Epoch  130  loss  2.548543627663432 correct 44
Epoch  140  loss  2.155165343864623 correct 48
Epoch  150  loss  1.364852311201901 correct 49
Epoch  160  loss  3.732796525726192 correct 43
Epoch  170  loss  1.7827599855520968 correct 48
Epoch  180  loss  0.8941720791736986 correct 48
Epoch  190  loss  1.4203519745531006 correct 49
Epoch  200  loss  2.2588510137776385 correct 46
Epoch  210  loss  1.5272531391359634 correct 48
Epoch  220  loss  1.0231941807256002 correct 50
Epoch  230  loss  1.012010466230424 correct 49
Epoch  240  loss  0.6217439248036666 correct 49
Epoch  250  loss  0.3726432874564438 correct 46
Epoch  260  loss  0.29029663121986043 correct 48
Epoch  270  loss  1.0739155815461927 correct 50
Epoch  280  loss  1.1493175023668059 correct 50
Epoch  290  loss  0.9242025705170873 correct 50
Epoch  300  loss  1.3796785524368615 correct 50
Epoch  310  loss  0.4326733323147213 correct 50
Epoch  320  loss  1.454854979805485 correct 50
Epoch  330  loss  1.2454818639739402 correct 50
Epoch  340  loss  0.8573492831120004 correct 48
Epoch  350  loss  0.5538124287378089 correct 50
Epoch  360  loss  0.7029846725182334 correct 49
Epoch  370  loss  0.09693635733598746 correct 50
Epoch  380  loss  1.2772328878596326 correct 50
Epoch  390  loss  0.8304244000059703 correct 50
Epoch  400  loss  1.7059677494391314 correct 48
Epoch  410  loss  1.2590798452960945 correct 50
Epoch  420  loss  0.7669605037473038 correct 50
Epoch  430  loss  0.42531217917171965 correct 50
Epoch  440  loss  0.22288350698416431 correct 50
Epoch  450  loss  0.8529361138009528 correct 50
Epoch  460  loss  0.27929433319730496 correct 50
Epoch  470  loss  0.46070567662134565 correct 50
Epoch  480  loss  0.850244731532931 correct 50
Epoch  490  loss  0.2412033568959602 correct 50
Epoch  500  loss  0.6606176054696647 correct 50
Time per epoch (s): 0.16422242832183837
```

### split
```
Epoch  0  loss  6.50180942150871 correct 33
Epoch  10  loss  8.172223932717824 correct 17
Epoch  20  loss  5.093400788303077 correct 36
Epoch  30  loss  5.633159119026533 correct 47
Epoch  40  loss  3.8956620583546195 correct 36
Epoch  50  loss  3.782976274321898 correct 48
Epoch  60  loss  3.214738933669382 correct 48
Epoch  70  loss  4.062763086644247 correct 47
Epoch  80  loss  2.482140103530833 correct 42
Epoch  90  loss  2.512930820318402 correct 50
Epoch  100  loss  2.2598266072971374 correct 49
Epoch  110  loss  2.5186235786892013 correct 50
Epoch  120  loss  1.1752866308012153 correct 49
Epoch  130  loss  1.79315293414534 correct 50
Epoch  140  loss  0.6347025847470852 correct 50
Epoch  150  loss  1.1122307529560191 correct 50
Epoch  160  loss  0.6207073897297368 correct 50
Epoch  170  loss  0.6987737506805379 correct 50
Epoch  180  loss  0.7944828227690275 correct 50
Epoch  190  loss  0.3166107019592051 correct 50
Epoch  200  loss  0.2791150865421208 correct 48
Epoch  210  loss  0.6450451513860475 correct 50
Epoch  220  loss  0.6981877922876049 correct 50
Epoch  230  loss  0.67674246468693 correct 50
Epoch  240  loss  0.5588280039125069 correct 50
Epoch  250  loss  0.40005591888796554 correct 50
Epoch  260  loss  0.49411535322498457 correct 50
Epoch  270  loss  0.2859337284207108 correct 50
Epoch  280  loss  0.6797008188828022 correct 50
Epoch  290  loss  0.12966819258407328 correct 50
Epoch  300  loss  0.2343617323569899 correct 50
Epoch  310  loss  0.1719765024106293 correct 50
Epoch  320  loss  0.2699418070316175 correct 50
Epoch  330  loss  0.5462103146384327 correct 50
Epoch  340  loss  0.14992150164310317 correct 50
Epoch  350  loss  0.2566812579089721 correct 50
Epoch  360  loss  0.4462053322926579 correct 50
Epoch  370  loss  0.1970658603056104 correct 50
Epoch  380  loss  0.31915914942128804 correct 50
Epoch  390  loss  0.04254891453549043 correct 50
Epoch  400  loss  0.09642175148293898 correct 50
Epoch  410  loss  0.18672827887832155 correct 50
Epoch  420  loss  0.11105171537504163 correct 50
Epoch  430  loss  0.12688569137610167 correct 50
Epoch  440  loss  0.21757764918143457 correct 50
Epoch  450  loss  0.14229485214838494 correct 50
Epoch  460  loss  0.24498590614014484 correct 50
Epoch  470  loss  0.3292199956147726 correct 50
Epoch  480  loss  0.0354380401144623 correct 50
Epoch  490  loss  0.015795499302837933 correct 50
Epoch  500  loss  0.22189660205518225 correct 50
Time per epoch (s): 0.16355704164505006
```

## Task 5 - Bigger Model (simple)
For PTS = 200, HIDDEN = 200, and RATE = 0.05

### GPU
```
Epoch  0  loss  1.1269893000551185 correct 188
Epoch  10  loss  0.08413630070883621 correct 199
Epoch  20  loss  1.9060785961238078 correct 192
Epoch  30  loss  0.8108486644907531 correct 195
Epoch  40  loss  0.04416532600994214 correct 198
Epoch  50  loss  0.22884415973455746 correct 199
Epoch  60  loss  0.0665868753469212 correct 200
Epoch  70  loss  0.1627889502844587 correct 200
Epoch  80  loss  0.09024163139129901 correct 199
Epoch  90  loss  0.0016177955945049134 correct 199
Epoch  100  loss  0.025018913932853822 correct 199
Epoch  110  loss  0.49490240968024585 correct 199
Epoch  120  loss  0.0023672716614148146 correct 200
Epoch  130  loss  0.013506868854200006 correct 199
Epoch  140  loss  0.1363551751773143 correct 200
Epoch  150  loss  0.07136930087998032 correct 200
Epoch  160  loss  0.023814453974418608 correct 200
Epoch  170  loss  0.5641905560813382 correct 198
Epoch  180  loss  0.009012193585058037 correct 200
Epoch  190  loss  0.02635514614808819 correct 200
Epoch  200  loss  0.006260662302913087 correct 200
Epoch  210  loss  0.03293259751187938 correct 200
Epoch  220  loss  0.08545771246259627 correct 199
Epoch  230  loss  0.013020338483542485 correct 200
Epoch  240  loss  7.872168949977823e-06 correct 200
Epoch  250  loss  0.0004272298154256862 correct 199
Epoch  260  loss  0.0002637077949640951 correct 200
Epoch  270  loss  5.685041042589046e-06 correct 200
Epoch  280  loss  0.0054838110957100405 correct 200
Epoch  290  loss  0.027089163014845284 correct 200
Epoch  300  loss  1.909319639766419e-05 correct 199
Epoch  310  loss  0.5144191742030442 correct 198
Epoch  320  loss  0.003565357470209348 correct 200
Epoch  330  loss  0.041655885073962985 correct 200
Epoch  340  loss  0.05952779638639636 correct 200
Epoch  350  loss  0.5334690789161147 correct 199
Epoch  360  loss  0.05576852839592955 correct 200
Epoch  370  loss  0.003772439946628276 correct 200
Epoch  380  loss  2.5514366119941128e-06 correct 200
Epoch  390  loss  1.6288957712692726e-06 correct 200
Epoch  400  loss  0.027688302419871854 correct 199
Epoch  410  loss  0.05590197671621058 correct 200
Epoch  420  loss  0.000182044070282177 correct 200
Epoch  430  loss  0.03552495314362153 correct 200
Epoch  440  loss  6.322040175555784e-06 correct 200
Epoch  450  loss  0.008127015813836248 correct 200
Epoch  460  loss  0.24705649988462236 correct 200
Epoch  470  loss  1.585601927057469e-05 correct 200
Epoch  480  loss  1.25001416915988e-05 correct 200
Epoch  490  loss  0.0009302671890427147 correct 200
Epoch  500  loss  0.0021065966106640806 correct 200
Time per epoch: 6.864321283340454
```

### CPU
```
Epoch  0  loss  1.3508192985435348 correct 173
Epoch  10  loss  0.14633371390705988 correct 197
Epoch  20  loss  0.7563808796427848 correct 194
Epoch  30  loss  0.3287398857501861 correct 196
Epoch  40  loss  0.27503177223111336 correct 195
Epoch  50  loss  0.2237707006209538 correct 198
Epoch  60  loss  0.34486548170167886 correct 198
Epoch  70  loss  0.0763961889352941 correct 198
Epoch  80  loss  0.2035803334154839 correct 196
Epoch  90  loss  0.9862214477213012 correct 197
Epoch  100  loss  0.6313706955287457 correct 199
Epoch  110  loss  0.03752738536385432 correct 200
Epoch  120  loss  0.07562047840163462 correct 200
Epoch  130  loss  3.18786380496589e-05 correct 198
Epoch  140  loss  8.063408432318266e-05 correct 200
Epoch  150  loss  0.00036928862912873447 correct 198
Epoch  160  loss  0.25487440329130956 correct 200
Epoch  170  loss  0.5202828674202334 correct 198
Epoch  180  loss  0.011357653338581644 correct 196
Epoch  190  loss  0.0004149426360631558 correct 197
Epoch  200  loss  0.004094663806569004 correct 197
Epoch  210  loss  0.2969238091086398 correct 200
Epoch  220  loss  0.3132721974180337 correct 200
Epoch  230  loss  0.45838556429530514 correct 199
Epoch  240  loss  0.3389613640504534 correct 196
Epoch  250  loss  0.5935349027926982 correct 200
Epoch  260  loss  0.49713096281568553 correct 198
Epoch  270  loss  7.053315424316926e-05 correct 200
Epoch  280  loss  0.004397860646873121 correct 197
Epoch  290  loss  1.3066058568730218e-05 correct 200
Epoch  300  loss  4.828672830434067e-06 correct 199
Epoch  310  loss  0.8664746550691068 correct 198
Epoch  320  loss  0.803106937709501 correct 197
Epoch  330  loss  0.01052649748351653 correct 200
Epoch  340  loss  4.519766113418631e-05 correct 190
Epoch  350  loss  0.3146422566726859 correct 200
Epoch  360  loss  2.6808625419533102e-05 correct 200
Epoch  370  loss  0.0022093204365901003 correct 200
Epoch  380  loss  3.606207157700133e-05 correct 200
Epoch  390  loss  1.9913371988521374e-08 correct 200
Epoch  400  loss  0.00090899214686642 correct 192
Epoch  410  loss  4.9278387200091635e-06 correct 200
Epoch  420  loss  0.01041607429762791 correct 200
Epoch  430  loss  1.2762773830046401e-05 correct 200
Epoch  440  loss  0.22548629280686566 correct 200
Epoch  450  loss  5.298135256755837e-07 correct 200
Epoch  460  loss  7.02425238939313e-05 correct 200
Epoch  470  loss  0.010740617354067396 correct 200
Epoch  480  loss  0.10695624819512775 correct 198
Epoch  490  loss  0.004155175449100537 correct 200
Epoch  500  loss  0.3163849237362998 correct 200
Time per epoch: 1.1968778700828553
```