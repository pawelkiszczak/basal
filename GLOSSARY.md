# basal serving

How basal-serve answers typed questions about a text and keeps the work it already did for that text across requests.

## Language

### State cache

**State**:
The text a request's questions are about. The state cache's unit is its rendered token prefix: the template up to the
end of the state.

**State cache**:
The keys and values of request states, kept across requests so a later request over the same state computes only its
questions and options.
_Avoid_: KV cache, prefix cache (that is `--prefix-cache`: the template prefix only)

**Memory tier**:
The state cache's entries on the device, bounded by `--state-cache-gb`.

**Disk tier**:
The state cache's files under `--state-cache-dir`, one directory per signature.
_Avoid_: SSD cache, cold tier

**Budget**:
The bytes a tier may hold; beyond it the least recently used states go first.

**Signature**:
What makes two servers' state files interchangeable: model, revision, weights, adapter, quantisation, KV dtype and file
format version.

### Managing it

**Clear**:
Removing the cached states of a tier.
_Avoid_: flush (that is writing the queued files), purge

**Reset**:
Zeroing the state cache's counters; never removes a state.

**No-store**:
A request that reads cached states but stores none of its own.
_Avoid_: bypass

**Prune**:
Removing the cache directories of other models.
