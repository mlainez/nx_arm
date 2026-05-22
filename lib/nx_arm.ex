defmodule NxArm do
  @moduledoc """
  Nx backend + Nx-tensor model wrappers for ARM CPUs, built on the
  `:arm_ai` NEON inference NIF.

  ## Backend

      Nx.global_default_backend(NxArm.Backend)

  See `NxArm.Backend` for the supported ops + fallbacks, and
  `NxArm.Compiler` for the Defn pattern fuser.

  ## Nx-tensor model wrappers

  All under `NxArm.Models.*` — Whisper, ONNX, YOLO, Silero VAD,
  Piper, OCR, Face, Stable Diffusion. They take/return Nx tensors
  and assume the user has set NxArm.Backend as the default.

  ## Companion package

  `:arm_ai` ships the underlying NIF + Nx-free APIs (LlamaCandle,
  Phonemizer). nx_arm depends on it. If you only need the
  pure-binary inference API without Nx, depend on `:arm_ai`
  directly.
  """

  @doc """
  Fused softmax along `axis` (default `-1`). Bypasses the
  primitive-by-primitive Axon/Nx.Defn decomposition for the
  common case.
  """
  def softmax(%Nx.Tensor{} = tensor, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    shape = Nx.shape(tensor) |> Tuple.to_list()
    rank = length(shape)
    axis = if axis < 0, do: axis + rank, else: axis

    if axis != rank - 1 do
      raise ArgumentError,
            "NxArm.softmax currently only supports the last axis (got #{axis} for rank #{rank})"
    end

    if Nx.type(tensor) != {:f, 32} do
      raise ArgumentError, "NxArm.softmax requires :f32 input"
    end

    inner = elem(Nx.shape(tensor), rank - 1)
    n_outer = div(Nx.size(tensor), inner)

    bin = NxArm.Backend.__bin_of__(tensor)
    out_bin = ArmAI.Native.softmax_f32_op(bin, n_outer, inner)

    %{tensor | data: %NxArm.Backend{bin: out_bin}}
  end
end
