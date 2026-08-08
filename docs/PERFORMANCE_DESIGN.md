# Inference performance design

This framework uses large-batch execution and cross-prompt continuous batching. Independent
algorithm workers submit synchronous calls to one shared backend; a background dispatcher merges requests that
become ready at nearly the same time. Every generation request already owns its seed, so changing scheduling order
does not change its random stream. Score results are split back into original request order.

The exact score cache is keyed by model wrapper, complete sampling configuration, prefix, and continuation. This is
important for replay: the same stored completion is often rescored under the base model and several behavior
policies, while a score from one temperature or truncation policy must never be reused for another. Random
generations are deliberately not cached.

Replay generation is flattened across candidates as well: a decision with candidate-specific fresh counts emits
one heterogeneous generation batch, and post-selection reserve completions use the same path. This removes the
candidate-by-candidate synchronization point while leaving replay keys, seeds, and behavior log-probabilities
unchanged.

The remaining distribution-preserving optimization layers are:

1. retain and fork prefix KV state for candidate blocks and MH suffixes where the concrete backend supports it;
2. overlap CPU reward parsing with the next ready GPU batch;
3. bucket variable-length work while retaining request-local seeds and actual sampling probabilities;
4. keep consumed replay records in the design pool so variance and cost estimates improve without leaking values
   into future evaluation decisions.

Hard proposal truncation, unrecorded sampling transforms, and data-dependent reuse of current evaluation rollouts
are intentionally excluded because they can change the estimator or its support.
