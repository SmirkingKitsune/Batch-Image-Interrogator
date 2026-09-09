# Inquiry inference controls

Read Metadata reports the declared context limit and an f16 KV estimate. It
preserves Context Size and Max Tokens. Apply Suggested changes only Context
Size; changing the model or token settings invalidates the report.

The suggestion assumes an 8,192-token prompt plus the configured reply budget,
rounded up and limited by model and UI bounds. This prompt allowance is a
placeholder, not measured usage. Persisting usage.prompt_tokens and deriving a
workload percentile is deferred. A limit below prompt plus reply budget cannot
guarantee a complete response. The Context Size control stops at 131,072, so a
model declaring more cannot be suggested or set beyond that here.

The KV estimate is computed from the declared attention shape, not from weight
file size. It reads head_count_kv, so grouped-query models are costed at their
actual cache width rather than as dense attention; it falls back to head_count
only when head_count_kv is absent, which is genuine multi-head attention. Do
not treat the figure as a padded upper bound.

KV estimates assume full-context attention in every declared layer. Hybrid,
recurrent, sliding-window, and runtime-specific allocations may differ. They
exclude model weights, the vision encoder, and compute buffers. The reported
figure is always f16: quantized KV types are implemented in core/context_sizing
but the UI does not yet expose a cache-type control, and q8_0 would be roughly
half the number shown. The UI does not infer GPU capacity from host
MemAvailable or guarantee memory fit; core.context_sizing.available_memory_bytes
exists for the deferred workload work and is not called by the UI. Confirm
actual allocations from the runtime for a hardware-specific budget.

Skip model reasoning sends enable_thinking=false to compatible chat templates.
It rides on each request, so it takes effect on the next request with no
reload; an in-flight request keeps its existing setting. Do not
carry reasoning between turns adds --no-reasoning-preserve at server launch and
requires reloading the model to restart the server with that flag. Both options default off. Custom llama-server builds
must support the selected flag; an unsupported flag makes llama-server exit at
startup, reported as "llama-server exited during startup" with the tail of the
server log attached, rather than being silently ignored.

Streaming requests use a 180-second inactivity guard. Content, tool arguments,
and reasoning deltas count as progress; keepalives do not. Buffered fallback
requests retain the existing 120/300-second timeout behavior. Long prefill
before the first token can still exceed the streaming guard on slow hardware.

These controls improve inference configuration and diagnostics. They do not
establish or fix the cause of the reported system shutdowns. No sustained GPU
crash reproduction was performed as part of this change.
