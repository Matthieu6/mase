## Tutorial 2:


Parameter Efficient Finetuning (PEFT) with LoRA.

LoRA involves replacing W with A and B which are low-rank matrices. We freeze W parameters and only allow the optimizer to train the parameters in A and B. This enables to acheive accuracies comparable to full fine tuning, while only training a fraction of the parameters. 

Total Trainable Parameters: 14480258 without LoRA

Total Trainable Parameters: 3169816 with LoRA

Once lora is conduncted, we can pass `fuse_lora_weights_transform_pass` to optimize the model for inference, replacing each LoRA instance with `nn.Linear` module, where the AB product added to the original weight matrix. 

By adjusting the rank number of LoRA, we can control the trade-off between memory usage and fine-tuned accuracy. Such parameter-efficient fine-tuning techniques are very useful in the area of large language models (LLMs), where the memory requirement for training is a significant bottleneck.
