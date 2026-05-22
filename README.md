# nx_arm

Nx backend + Nx-tensor model wrappers for ARM CPUs.

Built on top of [`arm_ai`](https://github.com/marclainez/arm_ai),
which ships the underlying NIF (NEON kernels, candle, tract-onnx,
symphonia, image) and the Nx-free APIs (LlamaCandle, Phonemizer).

## What's in here

- `NxArm.Backend` — `Nx.Backend` implementation. Set as default:

      Nx.global_default_backend(NxArm.Backend)

- `NxArm.Compiler` — `Nx.Defn.Compiler` with pattern fusion
  (softmax, GELU, LayerNorm) for Bumblebee / Axon graphs.
- Helpers: `NxArm.LLM`, `NxArm.KVCache`, `NxArm.Sampling`,
  `NxArm.Quantized`, `NxArm.QuantizedConv`, `NxArm.FFT`,
  `NxArm.Image`, `NxArm.Audio`, `NxArm.Vision`,
  `NxArm.Detection`, `NxArm.Embeddings`.
- `NxArm.Models.*` — Nx-tensor wrappers over `arm_ai`'s NIF for
  Whisper, ONNX, YOLO, Silero VAD, Piper, OCR, Face, Stable
  Diffusion.

## Companion package: `arm_ai`

If you DON'T need Nx tensors (e.g. you're driving the LLM
directly with token id lists), depend on `arm_ai` instead of
`nx_arm` — same NIF, smaller surface, no Nx dep.

## License

Apache-2.0.
