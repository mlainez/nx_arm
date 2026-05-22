defmodule NxArm.CompilerOptimisationsTest do
  @moduledoc """
  Exercises the compile-time optimisations in `NxArm.Compiler`'s
  rewriter pass that aren't model-pattern fusion:

  * `try_constant_fold/1` — fold pure-constant arithmetic
    (`add/multiply/subtract/divide` over two `:constant` nodes)
  * `try_dead_broadcast/1` — eliminate broadcasts whose output
    shape equals the input shape
  * Composite ops + indices ops survive the rewriter
  """

  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp run(fun, args) do
    arm_args = Enum.map(args, &arm/1)
    out = apply(Nx.Defn.jit(fun, compiler: NxArm.Compiler), arm_args)
    ref_out = apply(fun, args)
    diff = Nx.subtract(ref(out), ref(ref_out)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    {out, diff}
  end

  describe "try_constant_fold" do
    test "constant + constant folds at compile time" do
      fun = fn x -> Nx.add(x, Nx.add(2, 3)) end
      x = Nx.tensor([1.0, 2.0])
      {out, diff} = run(fun, [x])
      assert ref(out) |> Nx.to_flat_list() == [6.0, 7.0]
      assert diff < 1.0e-6
    end

    test "constant - constant fold" do
      fun = fn x -> Nx.subtract(x, Nx.subtract(10, 3)) end
      x = Nx.tensor([1.0, 2.0, 3.0])
      {out, _} = run(fun, [x])
      assert ref(out) |> Nx.to_flat_list() == [-6.0, -5.0, -4.0]
    end

    test "constant * constant fold" do
      fun = fn x -> Nx.multiply(x, Nx.multiply(2, 3)) end
      x = Nx.tensor([1.0])
      {out, _} = run(fun, [x])
      assert ref(out) |> Nx.to_flat_list() == [6.0]
    end

    test "non-constant operand doesn't fold" do
      fun = fn x, y -> Nx.add(x, y) end
      x = Nx.tensor([1.0, 2.0])
      y = Nx.tensor([10.0, 20.0])
      {out, _} = run(fun, [x, y])
      assert ref(out) |> Nx.to_flat_list() == [11.0, 22.0]
    end
  end

  describe "try_dead_broadcast" do
    test "broadcast to the same shape is a no-op (passes through)" do
      fun = fn x -> Nx.broadcast(x, {2, 3}) end
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      {out, diff} = run(fun, [x])
      assert Nx.shape(out) == {2, 3}
      assert diff < 1.0e-6
    end

    test "broadcast that adds dims still works (not eliminated)" do
      fun = fn x -> Nx.broadcast(x, {3, 2, 3}) end
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      {out, diff} = run(fun, [x])
      assert Nx.shape(out) == {3, 2, 3}
      assert diff < 1.0e-6
    end
  end

  describe "composite + indices ops survive the rewriter" do
    test "concatenate two tensors" do
      fun = fn a, b -> Nx.concatenate([a, b], axis: 0) end
      a = Nx.tensor([[1.0, 2.0]])
      b = Nx.tensor([[3.0, 4.0]])
      {out, diff} = run(fun, [a, b])
      assert Nx.shape(out) == {2, 2}
      assert diff < 1.0e-6
    end

    test "slice" do
      fun = fn x -> Nx.slice(x, [0, 1], [2, 1]) end
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      {out, _} = run(fun, [x])
      assert ref(out) |> Nx.to_flat_list() == [2.0, 5.0]
    end

    test "iota at compile time" do
      fun = fn -> Nx.iota({4}, type: :f32) end
      out =
        Nx.Defn.jit(fun, compiler: NxArm.Compiler).()

      assert ref(out) |> Nx.to_flat_list() == [0.0, 1.0, 2.0, 3.0]
    end
  end
end
