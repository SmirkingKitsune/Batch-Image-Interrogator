# Inquiry inference controls

Read Metadata reports the declared context limit and a dense-attention f16 KV
estimate. It preserves Context Size and Max Tokens. Apply Suggested changes
only Context Size; changing the model or token settings invalidates the report.

The suggestion assumes an 8,192-token prompt plus the configured reply budget,
rounded up and limited by model and UI bounds. This prompt allowance is a
placeholder, not measured usage. Persisting usage.prompt_tokens and deriving a
workload percentile is deferred. A limit below prompt plus reply budget cannot
guarantee a complete response.

KV estimates assume full-context attention in every declared layer. Hybrid,
recurrent, sliding-window, quantized, and runtime-specific allocations may
differ. They exclude model weights, the vision encoder, and compute buffers.
The UI does not infer GPU capacity from host MemAvailable or guarantee memory
fit. Confirm actual allocations from the runtime for a hardware-specific budget.

Skip model reasoning sends enable_thinking=false to compatible chat templates.
Do not carry reasoning between turns adds --no-reasoning-preserve at server
launch. Both options default off; reload the model after changing these controls.
Custom llama-server builds must support the selected flag.

Streaming requests use a 180-second inactivity guard. Content, tool arguments,
and reasoning deltas count as progress; keepalives do not. Buffered fallback
requests retain the existing 120/300-second timeout behavior. Long prefill before
the first token can still exceed the streaming guard on slow hardware.

These controls improve inference configuration and diagnostics. They do not
establish or fix the cause of the reported system shutdowns. No sustained GPU
crash reproduction was performed as part of this change.
