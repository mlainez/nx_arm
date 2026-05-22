defmodule NxArm.CompilerTest do
  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp diff(a, b) do
    Nx.subtract(ref(a), ref(b)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  describe "Nx.Defn.Compiler callbacks" do
    test "__to_backend__/1 returns NxArm.Backend" do
      assert {NxArm.Backend, []} == NxArm.Compiler.__to_backend__([])
    end

    test "__partitions_options__/1 returns the requested duplication" do
      assert [[]] == NxArm.Compiler.__partitions_options__([])
      assert [[max_concurrency: 4], [max_concurrency: 4], [max_concurrency: 4], [max_concurrency: 4]] ==
               NxArm.Compiler.__partitions_options__(max_concurrency: 4)
    end
  end

  describe "Nx.Defn JIT through the compiler" do
    test "elementwise add round-trip matches BinaryBackend" do
      fun = fn x, y -> Nx.add(x, y) end
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[10.0, 20.0], [30.0, 40.0]])

      out_arm =
        Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(a), arm(b))

      assert_in_delta diff(out_arm, fun.(a, b)), 0.0, 1.0e-6
    end

    test "multi-op pipeline (matmul + softmax) round-trip" do
      fun = fn x, w -> x |> Nx.dot(w) |> Nx.exp() |> then(&Nx.divide(&1, Nx.sum(&1, axes: [-1], keep_axes: true))) end
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      w = Nx.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

      out_arm = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x), arm(w))

      assert_in_delta diff(out_arm, fun.(x, w)), 0.0, 1.0e-5
    end

    test "rewriter pass identifies softmax pattern and lands the right shape" do
      fun = fn x ->
        e = Nx.exp(Nx.subtract(x, Nx.reduce_max(x, axes: [-1], keep_axes: true)))
        Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
      end

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
      out_arm = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))

      assert Nx.shape(out_arm) == {2, 4}
      rows = out_arm |> ref() |> Nx.sum(axes: [-1]) |> Nx.to_flat_list()
      Enum.each(rows, fn s -> assert_in_delta s, 1.0, 1.0e-5 end)
      assert_in_delta diff(out_arm, fun.(x)), 0.0, 1.0e-5
    end

    test "scalar constants survive the rewriter" do
      fun = fn x -> Nx.add(x, 1.0) end
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      out_arm = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))
      assert_in_delta diff(out_arm, fun.(x)), 0.0, 1.0e-6
    end

    test "reshape + transpose composite produces correct shape" do
      fun = fn x -> x |> Nx.reshape({6}) |> Nx.add(1.0) end
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      out_arm = Nx.Defn.jit(fun, compiler: NxArm.Compiler).(arm(x))
      assert Nx.shape(out_arm) == {6}
      assert ref(out_arm) |> Nx.to_flat_list() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    end
  end

  describe "hooks + GC options" do
    test "garbage_collect option doesn't blow up" do
      fun = fn x -> Nx.add(x, x) end
      x = Nx.tensor([1.0, 2.0, 3.0])
      out = Nx.Defn.jit(fun, compiler: NxArm.Compiler, garbage_collect: true).(arm(x))
      assert ref(out) |> Nx.to_flat_list() == [2.0, 4.0, 6.0]
    end
  end
end
