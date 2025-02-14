# ADL Coursework mrb124

## Lab 0:

### Tutorial 1

> Task: Delete the call to 'replace\_all\_uses\_with' to verify that FX will report a RuntimeError

**Why use** `**replace_all_uses_with(node.args[0])**`**?**

* **Ensures all nodes that depended on dropout still function**.
* If not used, those nodes will try to access a now-deleted dropout node.

If you **remove** `replace_all_uses_with(node.args[0])`, the dropout node is deleted **without redirecting its outputs**. This leaves dangling references in the graph.

![alt text](Tutorial%20Images/Tutorial%201/Task1_removefunction.png)

### Tutorial 2:

---

**BERT MODEL:** Deep learning model developed by Google designed for natural language processing (NLP) tasks.

> **Task 2:** Remove the \`attention\_mask\` and \`labels\` arguments from the \`hf\_input\_names\` list and re-run the following cell. Use \`mg.draw()\` to visualize the graph in each case. Can you see any changes in the graph topology? Can you explain why this happens?

- `labels` is used in training to compute the EntropyLoss at the final layer. When removing it, the cross-entropy calculation is omitted meaning the graph will now only output logits. there Is a removal of the ground truth label module.
- `attention_mask` is used to ignore padding tokens during attention compitations. Without it, all tokens (including padding) are considered equal, leading to incorrect attention weighting and worse performance. Basically, it is used to indicate which tokens are padding (0) and which are actual words (1). Removing it, the model will treat all tokens equally, including the padding tokens, which can intoduce noise in the computations. When no mask is specified, more infromation from the model is used as an input to the masking process.

<table>
  <tr>
    <th>With Last Layer</th>
    <th>Remove Last Layer</th>
  </tr>
  <tr>
    <td><img src="Tutorial Images/Tutorial 2/difference_in_graphs/with_last_layer.png" width="300"></td>
    <td><img src="Tutorial Images/Tutorial 2/difference_in_graphs/remove_last_layer.png" width="300"></td>
  </tr>
  <tr>
    <th>With Attention</th>
    <th>Remove Attention</th>
  </tr>
  <tr>
    <td><img src="Tutorial Images/Tutorial 2/difference_in_graphs/with_attention.png" width="300"></td>
    <td><img src="Tutorial Images/Tutorial 2/difference_in_graphs/Remove_attention.png" width="300"></td>
  </tr>
</table>

**Difference in the graph:** The difference in the graph is that now the nodes related to masking perations have dissapeared, as well as the loss computation node is removed. the final graoh becomes simpler, showing only forward comptations without loss.

## Lab 1: Model Compression (Quantization and Pruning)

### Tutorial 3

---

Use BERT model fine tuned for sequence classification and run Mase quantization pass. 

* Post-Training Quantisation (**PTQ**): Once the model is trained, quantise the model and evaluate accuracy
* Run further training iterations of the quantized model (**QAT**): Once quantized, train more to see difference in accuracy.

In this tutorial, use the model from Tutorial 2 trained with LoRA. 

| BaseLine LoRA Accuracy | PTQ Accuracy | QAT Accuracy |
| :--------------------: | :----------: | :----------: |
|         83.8%         |    82.3%    |    84.2%    |

Quantized model accuracy exceeds the full precision model, with much lower memory requirement to store the weights.

1. In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32.
   1. Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3.
   2. Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning

![alt text](Tutorial%20Images/Tutorial%203/ComparePTQandQAT.png)

As shown, fixed point widths were controlled to be 4, 8, 12, 16, 20, 24, 28, 32 bits. It is shown that post-quantization training is more effective than quantization without training. This is because it alows the model to adapt to the lower precisions before being evaluated. The fixed point width which acheived the highest accuracy whilst keeping the model complexity low was found to be approximately 8 bits. This created a good balance between accuracy and lower model complexity. This will be kept for the rest of the lab.

The training after quantisation was set to 4. This value was chosen by testing the accuracy of the model after post-quantization training whilst changing the training epochs. It was found that an epoch number of around 4-5 was best, achieving the highest accuracy wihtout overfitting the data which was found once more epochs were used. This is shown in the figure below, **validating our choice of 4 epoch of post-quantization training.**

![alt text](Tutorial%20Images/Tutorial%203/TestingOptimalEpochNumber.png)

### Tutorial 4

---

**Pruning:** used to reduce size and complexity of neural networks by removing unnecessay parameters. Used to reduce model size, decease inference time, impove generalisation and being more energy efficieny. 

* Structued pruning: removes entire structures (channels, filters, layers)
* Unstructured pruning: removes individual weights or connections.

Unstructured pruning:

* Sparcity: value from 0 to 1, express portion of element in the model that should be pruned 
* Method: pruning methods, including random and L1-norm 
* Scope: defines whether to consider each weight/activation individually (local) or all tensors in the model (global)

1. Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9.
   1. Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4.
   2. Plot separate curves for `Random` and `L1-Norm` methods to evaluate the effect of different pruning strategies.

![alt text](Tutorial%20Images/Tutorial%204/RandomvsL1Norm.png)

A range of sparsity levels were applied to the model using both l1-norm and random method. Once the model was pruned using the 2 techniques, post-quantisation training was conducted for 4 epoch (due to similar reason talked about In the previous task). It can be Interpreted that:

* **L1-NORM Pruning is more effective** : It maintains high accuracy at lower sparsity levels and degrades more gracefully than random pruning. L1-Norm pruning identifies the least important neurons, filters, or channels based on the L1 norm (sum of absolute values of weights). It makes sure that Important structures are not pruned, keeping accuracy high.
* **Random Pruning leads to a steep accuracy drop** : The model collapses much faster, showing that removing weights randomly is inefficient. Random pruning removes weights without considering their Importance, It deletes connections which are of high Importance In the network.

## Lab 2: Neural Architecture Search

### Tutorial 5

---

Neural Architecture Search (NAS) with Mase and Optuna:

`trial.suggest_int` and `trial.suggest_category` to trigger the chosen sampler to choose parameter choices and layer types. Basically goes through the suggested parameters and tries different combinations. 

The objective function is used to create a new model instance with the chosen parameters according to a sampler, which is trained on the IMDb dataset for a number of epochs. 

If the layer has **equal input and output features**, it **randomly selects** (via `trial.suggest_categorical`) whether to **keep it as** `nn.Linear` or **replace it with** `Identity`.

* **It depends on the** `trial` **execution.** Each layer with `in_features == out_features` is individually **decided at random** to be either:
  * `nn.Linear` (default case, keeps it as is).
  * `Identity()` (replaces the linear transformation with an identity operation). This simply passes the input forward unchanged, removes the learnable transformation at the later.

### Task 1

1. Tutorial 5 shows how to use random search to find the optimal configuration of hyperparameters and layer choices for the Bert model.
   1. Now, explore using the GridSampler and TPESampler in Optuna.
   2. Plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each sampler to compare their performance.

![alt text](Tutorial%20Images/Tutorial%205/Task%201%20best%20samplers/SamplerComparison.png)

Optuna's Sampler chooses between nn.Linear or Identity for each layer using TPE, Gridsampler or Random. Sampling between nn.Linear or Identity is useful to reduce model complexity, simulate sparse architectures, parameter search for optimal depth, and reducing overfitting.

**Results:**

Overall, TPE was the sampler with the best achieving accuracy, as it builds a probabilistic model for promising parameter regions and actively searches for better parameters based on past evaluations. Unlike Gridsampler, which explores a fixed set of paramters blindly and RandomSampler whcih selects parameters randomly, TPE focuses more on areas that are expected to yield higher accuracy.

Sampler type:

* **TPESampler**:

  * dynamically adapts to previous trials, learning which parameters It must change to find the most optimal results.
  * This sampler balances testing new hyperparameters with refining the ones which are In a promising search area already.
  * It only functions well for high trial numbers as It needs time to Initialise Its search and become familiar with the hyperparameters.
* **GridSampler**:

  * This sampler completes an exhaustive search of all combinations of hyperparameters In a set grid.
  * It Is very computationally expensive, and may take time to reach an optimal search.
  * It starts slow, but eventually finds a good configuration after several trials.
* **RandomSampler**:

  * Samples parameters randomly without learning from previous trials.
  * There is no refinement based on past accuacy results.

Once the best model with TPESampler was found, more training epochs were given to the model to see if accuracy can be increased further. As shown in the graph below, it was found that the accuracy decreased, highly due to overfitting when epochs were further increased.

The figures above were completed by searching for parameters using Optuna, and then completing 1 epoch of training before evaluating the model to see its accuracy. The value of 1 epoch was selected as It was found that more than 1 epoch of training overfit the data, decreasing the model's accuracy as shown In the figure below.

![alt text](Tutorial%20Images/Tutorial%205/TestingMoreEpochs.png)

Additionally, to see the changes between each epoch, the following plot was made, showing clear overfitting even after the first iteration.

![alt text](Tutorial%20Images/Tutorial%205/SmallEpochValidationTest.png)

To further investigate if overfitting was occuring with more epochs, a plot of the loss function at each step of training was competed. Even after 50% of 1 epoch, the loss started increasing again, demonstrating that the accuracy values would decrease in values, proving overfitting occured.

![alt text](Tutorial%20Images/Tutorial%205/TrainingLoss.png)

### Task 2

Once the model is constructed and trained for some iterations, call the CompressionPipeline to qunatize and pune the model, the continue training for a few more epochs. Use the sampler tat yeods best results in Task 1 to run the compression aware seach. The obectve function should retun gthe final accuracy fo the mdoe after compression. consider also th case wehre final training si performed after qunatixation. 

Get the best model from task 1, see how training affects it. 

Compression-aware search without post-compression training and compression aware with post compression training. 

1. In Tutorial 5, NAS is used to find an optimal configuration of hyperparameters, then we use the CompressionPipeline in Mase to quantize and prune the model after search is finished. However, the final compressed model may not be optimal, since different model architectures may have different sensitivities to quantization and pruning. Ideally, we want to run a compression-aware search flow, where the quantization and pruning is considered in each trial.
   1. In the objective function, after the model is constructed and trained for some iterations, call the CompressionPipeline to quantize and prune the model, then continue training for a few more epochs. Use the sampler that yielded the best results in Task 1 to run the compression-aware search. The objective function should return the final accuracy of the model after compression. Consider also the case where final training is performed after quantization/pruning.
   2. Plot a new figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. There should be three curves: 1. the best performance from Task 1 (without compression), compression-aware search without post-compression training, and compression-aware search with post-compression training.

![alt text](Tutorial%20Images/Tutorial%205/CompressionComparison.png)

Before this test was completed, no more training epochs were added to the data, this is because it was found previously that with more training, the model overfit, decreasing accuracy. It was shown that post quantisation training lead to higher accuracy than the baseline, which may be due to reduced complexity reducing overfitting.

## Lab 3: Mixed Precision Search

Defining a search space with linear\_layer\_choices as well as `width_choices` and `fractional_width_choices`. 

#### Task 1

1. In Tutorial 6, all layers allocated to IntegerLinear are allocated the same width and fractional width. This is suboptimal, as different layers may have different sensitivities to quantization.
   1. Modify the code to allow different layers to have widths in the range \[8, 16, 32\] and fractional widths in the range \[2, 4, 8\]. Expose this choice as an additional hyperparameter for the Optuna sampler.
   2. Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.

![alt text](Tutorial%20Images/Tutorial%206/LinearPrecision.png)

As shown, using TPE search, the accuracy increased significantly after the 2nd trial. Due to no further increase after 10 trials, the test was stopped.

#### Task 2

1. In Section 1 of Tutorial 6, when defining the search space, a number of layers are imported, however only LinearInteger and the full precision nn.Linear are selected.
   1. Now, extend the search to consider all supported precisions for the Linear layer in Mase, including Minifloat, BlockFP, BlockLog, Binary, etc. This may also require changing the model constructor so the required arguments are passed when instantiating each layer.
   2. Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each precision to compare their performance.

During this test, the model was able to replace the original nn.Linear layers using `trial.suggest_categorical` to one of the other precision types per run.

The **figure on the Left** represent the overall search where each line represents each precision.

The **figure on the right** represents a zoomed In version of the figure on the left for precision types which acheived similar results.

<p align="center">
  <img src="Tutorial Images/Tutorial 6/Task 2/ComparingPrecisions2.png" alt="Comparing Precisions" width="45%">
  <img src="Tutorial Images/Tutorial 6/Task 2/ZoomedInComparison2.png" alt="Zoomed In Comparison" width="45%">
</p>

**Observations:** As shown, the best performing model was found to be **LinearBinary** although only able to change weight values to either 0 or 1 which should have reduced the accuracy of the model. This Is due to the `trial.suggest_categorical` preferring to keep most layers as nn.Linear (full precision) as the only goal of the search was to obtain the highest accuracy.

| Precision Type        | Performance in the Graph        | Reason for Performance                                                                                                                                       |
| --------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LinearBinary          | **Highest accuracy (≈0.869)** | Retained most layers as nn.Linear due to `trial.suggest_categorica`l, maintaining full precision and avoiding information loss from quantization.          |
| LinearBinaryScaling   | Moderate accuracy (≈0.864)     | Uses binary weights with scaling factors, which helps retain some precision while reducing memory footprint, leading to better performance than pure binary. |
| LinearBlockFP         | High accuracy (≈0.867)         | Block floating point representation allows dynamic exponent sharing across weight groups, preserving more precision and maintaining strong performance.      |
| LinearInteger         | High accuracy (≈0.866)         | Integer quantization is efficient but introduces fixed-point rounding errors, leading to minor accuracy loss compared to floating-point approaches.          |
| LinearLog             | Low accuracy (≈0.70)           | Logarithmic quantization compresses values heavily, particularly small ones, leading to loss of critical information and accuracy degradation.               |
| LinearBlockLog        | **Worst accuracy (≈0.50)**    | Logarithmic quantization with block encoding severely limits representation for small values, leading to significant information loss and minimal learning.  |
| LinearMinifloatDenorm | High accuracy (≈0.866)         | Minifloat with denorms allows representation of very small values, improving precision close to full floating-point representation.                          |
| LinearMinifloatIEEE   | High accuracy (≈0.866)         | IEEE Minifloat uses fewer bits while retaining floating-point behavior, achieving near full-precision accuracy while reducing memory and computational cost. |

---

#### Mixed Precision tested all at once:

Additionally, another test was conducted where all the parameters were Included In the same search space, so nn.Linear could be replaced by any precision type. This was done to get a clearer understanding on the overall structure of the model when given full access to the data.

![alt text](Tutorial%20Images/Tutorial%206/Task%202/All_at_Once/optuna_progress_MixedPrecisionAllAtOnce.png)

The best-performing model used the following counts of each precision type, achieving a final accuracy of **87.2%.** This confirms that a combination of Minifloat, BlockFP, Integer, and Binary precision types was optimal, balancing accuracy and efficiency.

| **Precision Type**             | **Count** |
| ------------------------------------ | --------------- |
| **LinearMinifloatIEEE**        | 6               |
| **LinearBlockFP**              | 5               |
| **LinearInteger**              | 4               |
| **LinearBinary**               | 4               |
| **LinearBinaryScaling**        | 3               |
| **nn.Linear (Full Precision)** | 2               |
| **LinearBlockLog**             | 1               |
| **LinearMinifloatDenorm**      | 1               |

#### Implementing a Cost function:

Finally, investigation into the behaviour of the search with a **cost function was investigated**. This was done using function from mase named `calculate_avg_bits_mg_analysis_pass` which finds the average bit width of the weights and data In each model used to find the best accuracy. A composite metric (accuracy - lambda x average bits) was used to control the search. This was done using only LinearInteger as It provides a good balance between high precision and

It rewards models with lower precisions as It reduces model complexity, whilst still trying to obtain the highest accuracy.

- The **figure on left** shows the overall accuracy of LinearInteger using the cost function
- The **figure on the right** shows the value of the cost function which very quickly converges.

Overall: This showed that the model was able to keep accuracy constrant whilst decreasing the average bit size of the model, showing the cost function worked correctly.

<p align="center">
  <img src="Tutorial Images/Tutorial 6/Task 2/CostFunctionInvestigation/Accuracy.png" alt="Accuracy" width="45%">
  <img src="Tutorial Images/Tutorial 6/Task 2/CostFunctionInvestigation/Reward Obtained.png" alt="Reward Obtained" width="45%">
</p>

## Lab 4: (Software Stream) Performance Engineering

torch.compile: Makes Pytorch models run faster. In-time compiler that optimizes the model and the input data for the specific hardware.

* **JIT compilation** – converts python code into machine code at runtime. Continuously analyses the code being executed and identifies parts of the code where the speedup gained from compilation or recompilation would outweight the overhead of compiling that code.

#### 1\. In the first part of Lab 4 (torch.compile), we did not really observe real run-time speedups with torch.compile.

1. Modify the code and investigate why this is the case?

```python
# Run multiple times and compare
for i in range(3):
    print(f"Run {i+1}")
    avg_t = time_model(model, n=n, device=device)
    opt_avg_t = time_model(opt_model, n=n, device=device)
    print(f"Original model: {avg_t:.4f} s")
    print(f"Optimized model: {opt_avg_t:.4f} s")
```

```
Run 1
Original model: 9.0700 s
Optimized model: 6.1027 s

Run 2
Original model: 9.2037 s
Optimized model: 5.8397 s

Run 3
Original model: 9.0928 s
Optimized model: 5.5648 s
```

Without running successive passes through the model, runs through the pre-compiled model were faster than the JIT-compiled model. This Is due to a number of factors which were Investigated by modifying the code:

* **Compilation overhead**: when conducting JIT-compile, a compilation step Is necessary, adding overhead for short runs when the model Is ran only a few times e.g. `n=5`. This Is because the first runs Include tracing and graph transofmrations (adding overhead). When running more iterations (e.g., `n=100`), the overhead is amortized, leading to more stable performance.

As shown In the code above, when using n=100, the more times the the compiler was used, the faster the optimized model became, proving the point above.

#### 2. If you change the device to cuda, do you observe the same thing?

```python
# Function to move data to the same device
def get_data(batch_size=1, input_size=(3, 224, 224)):
    """Generate a dummy input tensor and move it to the target device."""
    return torch.randn(batch_size, *input_size).to(device)

# Define function to time model execution
def time_model(model, n=10, device="cuda"):
    """Measure the average inference time of the model."""
    model = model.to(device)
    data = get_data()
  
    # Warm-up to ensure accurate timing
    with torch.no_grad():
        for _ in range(3):
            model(data)

    # Measure execution time
    start = time.time()
    with torch.no_grad():
        for _ in range(n):
            model(data)
    end = time.time()
  
    return (end - start) / n
```

```
Original model (GPU): 0.0057 s 
Optimized model (GPU): 0.0019 s
```

Originally, when running the code on cuda without any code modifications, the optimized GPU model took more time to run that the original model. This Is likewise explained by overhead, where the model needs to be traced, then apply a graph transformation. With only  **`n=10`** , the overhead from tracing may not be fully amortized, making the compiled model appear slower.

Modifications were added to the code to ensure both the model and Input tensors are moved to the GPU for optimal performance. Additionally

- Ensuring both the **model and input tensors** are moved to the GPU is crucial for optimal performance.
- Additionally, **Warm-up runs** help mitigate initial compilation overhead, making timing measurements more reliable

---

### In the second part of Lab 4 (kernel fusion), we looked at a fused SDPA kernel.

Kernel Fusion fuses both kernel in 1, matrix multiplication and ReLU activation are executed within the same kernel. The ReLU transformation is applied directly after multiplication, before writing to global memory. 

`F.scaled_dot_product_attention()` is used to fuse all the three steps in the example into one GPU kernel. Uses **FlashAttention optimizations**. 

1. Loads small tiles of Q, K, V into GPU shared memory
2. Computes attention scores and softmax within registers (fast local memory)
3. Uses trick called “online softmax”  
   1. Rather than computing softmax at once, stores only the row max & row sum to accumulate results more efficiently. 
4. Computes weighted sum of V efficiencly without writing intermediate results to global memory. 

#### Now, extend the profiling to the SDPA kernel, compare its runtime behavior with the naive implementation.

```
Running on: cpu 
Naïve SDPA Time: 0.023348 s 
Fused SDPA Time: 0.020229 s 
Speedup Factor: 1.15x (Higher is better)
```

The fused kernels significantly outperform the naïve SDPA. These results prove that the fusing kernels reduce memory access, as they store Intermediate results In registers Instead of global memory. Instead of computing softmax all at once, It accumulates the row max and row sum Incrementally.

#### If you change the device to cuda, do you observe the same thing?

```
Running on: cuda
Naïve SDPA Time: 0.000502 s
Fused SDPA Time: 0.000274 s
Speedup Factor: 1.83x (Higher is better)
```

On CUDA, the fused version show an even larger speedup factor compared to CPU. FlashAttention tricks (online softmax, memory tiling) are GPU-optimized, leading to greater efficiency.

---

## Part 3: Custom Kernels (MXINT)

In the third part of lab4 (Custom kernel), we go through how to write MXINT8 dequantization kernel and bind it to Python.

### How does MXINT8 benefit custom hardware if both the activation and weights in a linear layer are quantized to MXINT8?

There are several benefits for custom hardware when using MXINT8:

* **Faster Computation:** Specialised hardware can be used to perform matrix multiplication. Hardware such as TPUs support lower precision than FP32 (e.g. INT8). Therefore lower bit matrix multiplication can run with reduced power consumption.
* **Reduced Memory:** FP32 weights and activations require 32 bits per value. On the other hand, MXINT uses only 8 bits per value (4 bits for mantissa + 8 bit shared exponents across a group). This allows models to fit In smaller and faster chips Instead on replying on global memory (e.g. DRAM)

Now, when BOTH the activation and weights are quantised In the the same MXINT method, advantages Inlude:

* **Optimal fusing from matrix multiplication**: If all the data types are the same, there Is less computational demand for fusing them Into one optimized kernel. If activations and weights used different quantization methods, hardware would require **separate processing units** or additional **format conversion steps**, increasinglatency and power consumption .
* **Reduced memory use**: as there are no extra tensors being stored and unused during matrix multiplication
  * If activations were stored in floating-point while weights were in MXINT, the hardware would need runtime conversions between the two formats.

### What is the purpose of the variable dont\_need\_abs and bias in the C++ for loop?

In MXINT8, the use of `dont_need_abs` and `bias` variable Is to correctly reconstruct the dequantized floating-point values from the MXINT8 format. Adjustments are needed to Interpret the mantissa.

The final dequantized value Is computed as:

Dequantized Value = Mantissa x 2^(exponent-127)

If the mantissa Is not porperly normalized, applying the exponent can result In Incorrect values.

 **`dont_need_abs`:** as the mantissa In MXINT8 does not have an Implicit leading bit, the 6th bit (represented by X in this binary: 0X000000) of the `mantissa_abs` determines whether the value Is already correctly scaled. IEEE floating point Includes a hidden leading 1 for normalized numbers, but MXINT does not.

```python
auto dont_need_abs = bool(mantissa_abs & 0x40);
```

If the 6th bit is 0 , the mantissa is too small to be directly scaled by the exponent. The  exponent already accounts for some scaling, so retaining an unnormalized mantissa would  introduce an incorrect order of magnitude.

* When the 6th bit Is 1: mantissa fully expressed and does not require additional adjustments.
* When the 6th bit Is 0: bias substraction Is needed to correctly dequantize the value.

 **`bias`:** This Is used to handle cases where adjustments are needed.  It shows the minimal possible dequantized value for a given exponent, used to adjust for missing Implicit bits as discussed above when `dont_need_abs` Is FALSE.

```python
auto bias = cutlass::bfloat16_t::bitcast(sign | exp | uint16_t(0));
```

If the mantissa is too small, the `bias` correction ensures that it is properly normalized before applying the exponent. The bias is subtracted from the mantissa to shift it into the correct range .

### How does cta_tiler partition data for copying to shared memory in CUDA kernel?

`cta_tiler` in CUTE library is used to partition data into tiles that are distributed across Cooperative Thread Arrays (CTA) in CUDA. This allows each CTA to handle a specific protion of the data, giving the ability of parallel processing. This then enables the CTA to work on a subset of data, reducing memory access latency. An Important feature to note Is that It retains Indexing structure.

**How It work:**

The tile size depends on the group size and CTA dimensions (`BLK_M`, `BLK_K`).

The code below converts a 16 x 16 matrix into 4x16 tiles. Assuming the matrix at the start Is 16x16

```python
auto BLK_M = Int<4>{};
auto BLK_K = Int<16>{};
auto cta_tiler = make_shape(BLK_M, BLK_K);
```

Once the matrix Is split, each CTA processes one of the tiles In parallel. Each CTA gest a unique tile index, maked `blockIdx.x, blockIdx.y`

```python
auto cta_coord = make_coord(blockIdx.x, blockIdx.y);
```

```python
auto tile = local_tile(A, cta_tiler, cta_coord);
```

Then `local_tile()` fetches the correct tile from global memory.

Since shared memory is **faster than global memory,** each CTA copies Its tiles Into shared memory before computation.

### How does layout\_sX partition threads in a threadlock for computation? (Challenge)

`layout_sX` plays an Important role in organizing threads within a thread block for computation. `BLK_M` and `BLK_K` like the previous question represent the dimensions of the tile that a thread block (CTA) will process. The `make_layout` function creates a layout object that describes how data is organized in shared memory.

Assume `BLK_M = 4` and `BLK_K = 16`, resulting in a 4x16 tile. If the thread block consists of 64 threads, `layout_sX` will partition the 4x16 shared memory tile such that each thread is responsible for a specific element or a subset of elements within this tile. This partitioning allows for correct memory accesses and efficient parallel processing. By partitioning the data according to the specified layout, it ensures that each thread handles a specific portion of the workload, leading to optimized memory access patterns and improved computational efficiency.

The `local_partition` function partitions the shared memory tile `sX` according to `layout_sX`, assigning each thread (`threadIdx.x`) a specific portion of the data to work on. This ensures that threads operate on distinct segments of data, facilitating parallel computation and efficient memory access.

This Is a simple example of the strcture of the ouput

```python
Thread (0) processes tile row 0, col 0
Thread (1) processes tile row 1, col 0
Thread (2) processes tile row 2, col 0
...
Thread (31) processes tile row 3, col 15
```

### Why the saved GPU memory is not exactly (32 - (4+8/32))/32 = 86.7% of the FP32 model?

The GPU memory Is not exactly (32 - (4+8/32))/32 = 86.7% because the code uses MXINT8 and not MXINT4, making the equation used to calculate the saved GPU memory Is Incorrect. The correct equation should be (32 - (8 + 8/32))/32 = 74.2% due to the 8 mantissas used compared to 4. Even with MXINT8 quantization , the GPU does not achieve the full 74.2% savings. This is because only specific layers (linear layers) are quantized, while other layers remain in full precision (FP32 or FP16) In the quantisation process. The code shows that only linear layers are quantized, whilst other layers keep their full precision.

```python
for layer_name, layer in model.named_modules():
    if not isinstance(layer, torch.nn.Linear):
        continue
    if "classifier" in layer_name:
        continue
    layer.cuda()
    layer_q = QLinearPacked.build_from_linear(layer, group_size=mxint8_group_size)
    set_layer_by_name(model, layer_name, layer_q)
    del layer
    torch.cuda.empty_cache()
```

Therefore although quantisation weights are less precise (take up less memory) the **GPU memory is reduced by 66.4% Instead of the calulcation of 74.2% due to non quantization layers being used.
