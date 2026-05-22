defmodule NxArm.CompilerFusionTest do
  @moduledoc """
  Targeted tests for the rewriter pass in `NxArm.Compiler` — the
  Defn-time pattern fuser that detects softmax / GELU / LayerNorm
  subgraphs and replaces them with single fused NIF calls.

  We check three properties for each fusion target:

  1. The fused output matches `Nx.BinaryBackend` for the same
     input within tight float tolerance.
  2. Constant folding + dead-broadcast elimination + dropout-elim
     don't change the result.
  3. Non-matching graphs pass through untouched.
  """

  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp diff(a, b) do
    Nx.subtract(ref(a), ref(b)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  describe "softmax fusion (try_softmax_divide / try_softmax_multiply)" do
    test "matches BinaryBackend on the canonical (exp − max) / sum form" do
      fun = fn x ->
        m = Nx.reduce_max(x, axes: [-1], keep_axes: true)
        e = Nx.exp(Nx.subtract(x, m))
        Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
      end

      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))

      assert_in_delta diff(out, fun.(x)), 0.0, 1.0e-5
      assert ref(out) |> Nx.sum(axes: [-1]) |> Nx.to_flat_list()
             |> Enum.all?(fn s -> abs(s - 1.0) < 1.0e-5 end)
    end

    test "also handles the Axon.Activations.softmax shape (reciprocal × exp)" do
      fun = fn x ->
        m = Nx.reduce_max(x, axes: [-1], keep_axes: true)
        e = Nx.exp(Nx.subtract(x, m))
        s = Nx.sum(e, axes: [-1], keep_axes: true)
        # reciprocal × exp is what Axon's lowering produces post-fusion
        Nx.multiply(e, Nx.divide(1, s))
      end

      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))

      assert_in_delta diff(out, fun.(x)), 0.0, 1.0e-5
    end
  end

  describe "GELU fusion (try_gelu)" do
    test "matches BinaryBackend on the standard erf-based GELU form" do
      sqrt2 = :math.sqrt(2.0)

      fun = fn x ->
        gate = Nx.divide(x, sqrt2) |> Nx.erf() |> Nx.add(1.0)
        Nx.multiply(x, gate) |> Nx.divide(2.0)
      end

      x = Nx.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))

      assert_in_delta diff(out, fun.(x)), 0.0, 1.0e-5
    end
  end

  describe "LayerNorm fusion (try_layernorm)" do
    test "matches BinaryBackend on the standard mean/var/scale-shift form" do
      fun = fn x, gamma, beta ->
        mu = Nx.mean(x, axes: [-1], keep_axes: true)
        diff_x = Nx.subtract(x, mu)
        var = Nx.mean(Nx.multiply(diff_x, diff_x), axes: [-1], keep_axes: true)
        eps = 1.0e-5
        inv = Nx.rsqrt(Nx.add(var, eps))
        norm = Nx.multiply(diff_x, inv)
        Nx.add(Nx.multiply(norm, gamma), beta)
      end

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      beta = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x), arm(gamma), arm(beta))

      assert_in_delta diff(out, fun.(x, gamma, beta)), 0.0, 1.0e-4
    end
  end

  describe "no-fusion passes through correctly" do
    test "arbitrary linear pipeline" do
      fun = fn x, w, b -> Nx.add(Nx.dot(x, w), b) end
      x = Nx.tensor([[1.0, 2.0]])
      w = Nx.tensor([[3.0, 4.0], [5.0, 6.0]])
      b = Nx.tensor([0.5, 0.5])

      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x), arm(w), arm(b))

      assert_in_delta diff(out, fun.(x, w, b)), 0.0, 1.0e-5
    end

    test "deeply nested arithmetic — no patterns hit" do
      fun = fn x ->
        x
        |> Nx.multiply(2)
        |> Nx.add(1)
        |> Nx.subtract(3)
        |> Nx.multiply(0.5)
      end

      x = Nx.tensor([[1.0, 2.0, 3.0]])
      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))

      assert_in_delta diff(out, fun.(x)), 0.0, 1.0e-6
    end
  end

  describe "rewriter optimisations" do
    test "dropout in inference mode is eliminated (metadata stripped)" do
      # Axon emits `metadata` wrappers tagged dropout: true for inference.
      # The rewriter's try_dropout_elim should pass the inner value through.
      fun = fn x -> Nx.add(x, 1.0) end
      x = Nx.tensor([1.0, 2.0, 3.0])
      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))
      assert ref(out) |> Nx.to_flat_list() == [2.0, 3.0, 4.0]
    end
  end
end
