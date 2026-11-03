# Whole-state files per engine in the disk tier

The disk tier stores one `.safetensors` file per whole cached state, named by the state's key and the engine that
computed it, rather than oMLX-style 256-token blocks shared between engines. `StateCache` already stores whole states
and finds prefixes through chained block hashes, so a prefix hit loads one file; the two engines' states of one text
differ in their last bits (bf16 kernels), so each keeps its own file and cross-engine reuse is opt-in
(`--state-cache-import any-engine`). Cost: a state present on both engines takes twice the disk.
