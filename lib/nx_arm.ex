defmodule NxArm do
  @moduledoc """
  `Nx.Backend` and `Nx.Defn.Compiler` for ARM CPUs, built on the
  `:arm_ai` NEON kernels.

      Nx.global_default_backend(NxArm.Backend)
      Nx.Defn.default_options(compiler: NxArm.Compiler)

  See `NxArm.Backend` for the natively supported ops (everything else
  falls back to `Nx.BinaryBackend`) and `NxArm.Compiler` for the Defn
  evaluator with softmax / GELU / LayerNorm fusion.

  If you only need quantized LLM inference without Nx, depend on
  `:arm_ai` directly and use `ArmAI.LlamaCandle`.
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
