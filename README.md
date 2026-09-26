# nx_arm

> ### ⚠️ Very early work — built for a workshop, not for production
>
> This package was written for the **Goatmire Elixir workshop** on running
> Nerves on Fairphone 3 hardware. It exists for tinkering and teaching.
> There are no stability guarantees and APIs will change without notice.
>
> See [`nerves_ai`](https://github.com/mlainez/nerves_ai) for the full
> stack and the workshop context.

`Nx.Backend` and `Nx.Defn.Compiler` for ARM CPUs, built on the NEON
kernels in [`arm_ai`](https://github.com/mlainez/arm_ai).

## What's in here

- `NxArm.Backend` — an `Nx.Backend` implementation. f32 elementwise ops,
  `dot`, reductions, `conv`, shape ops, gather/scatter, windows, sort and
  1-D FFT run natively; everything else falls back to `Nx.BinaryBackend`.

      Nx.global_default_backend(NxArm.Backend)

- `NxArm.Compiler` — an `Nx.Defn.Compiler` that evaluates graphs on
  `NxArm.Backend` and fuses softmax, GELU and LayerNorm subgraphs into
  single NIF calls. Works with Bumblebee / Axon via `compiler:`.
- `NxArm.softmax/2` — fused last-axis softmax.

## Install

```elixir
defp deps do
  [{:nx_arm, github: "mlainez/nx_arm"}]
end
```

This pulls in `arm_ai`, whose NIF builds from source, so the build
machine needs a Rust toolchain.

## Companion package: `arm_ai`

If you don't need Nx tensors (for example you drive an LLM with token id
lists), depend on `arm_ai` alone and use `ArmAI.LlamaCandle`.

## Toolchain

Built and tested with Erlang/OTP 29.1.1 and Elixir 1.20.4 against Nx 0.12,
matching the official Nerves systems (see `.tool-versions`).

## License

Apache-2.0.
