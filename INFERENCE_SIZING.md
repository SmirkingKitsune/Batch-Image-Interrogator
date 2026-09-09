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

Reasoning Budget caps thinking at N tokens via --reasoning-budget; -1 passes no
flag and leaves it unrestricted, 0 ends thinking immediately. It is a launch
flag, so changing it requires reloading the model. It has no effect while Skip
model reasoning is checked, since that removes the thinking block the budget
would bound; the control is disabled in that state, but a settings file can
still hold both and the budget is silently ignored.

Measured on Qwen3-VL 27B with WD/Camie context, two images, per image:

| mode | tags | approx s/image |
|---|---|---|
| skip reasoning | 14 / 18 | 45 |
| budget 512 | 30 / 52 | 155 |
| budget 1024 | 30 / 56 | 235 |
| unrestricted | 30 / 39 | 900-1900 |

At budget 1024 one image reproduced the unrestricted tag list exactly, so
unrestricted reasoning was spending 6,500-13,700 tokens to reach an answer 1024
tokens reaches. Skipping reasoning also changed an answer, not just its length:
it emitted two_boys where WD (0.97), unrestricted, 512 and 1024 all agree on
1boy. Higher budgets are not monotonically better - budget 1024 on the second
image produced dog_ears and wolf_ears together, and every mode emits synonym
pairs the prompt forbids. Treat these numbers as two images on one model, not a
general result.

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
