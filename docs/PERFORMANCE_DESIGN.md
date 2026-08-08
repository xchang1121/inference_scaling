# Inference performance design

This framework uses large-batch execution and cross-prompt continuous batching. Independent
algorithm workers submit synchronous calls to one shared backend; a background dispatcher merges requests that
become ready at nearly the same time. Every generation request already owns its seed, so changing scheduling order
does not change its random stream. Score results are split back into original request order.

The remaining distribution-preserving optimization layers are:

1. cache exact behavior and base scores by model, sampling policy, prefix, and continuation;
2. retain and fork prefix KV state for candidate blocks and MH suffixes where the concrete backend supports it;
3. overlap CPU reward parsing with the next ready GPU batch;
4. bucket variable-length work while retaining request-local seeds and actual sampling probabilities;
5. keep consumed replay records in the design pool so variance and cost estimates improve without leaking values
   into future evaluation decisions.

Hard proposal truncation, unrecorded sampling transforms, and data-dependent reuse of current evaluation rollouts
are intentionally excluded because they can change the estimator or its support.
