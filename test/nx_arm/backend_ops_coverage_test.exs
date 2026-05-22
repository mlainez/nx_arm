defmodule NxArm.BackendOpsCoverageTest do
  @moduledoc """
  Hits as many `Nx.Backend` callbacks as possible to exercise the
  fallback paths in `NxArm.Backend`. Where we have a fast path,
  the inputs match it; where we don't, the same call routes to
  `Nx.BinaryBackend` via `fallback/2`. Either way: the answer
  must equal the BinaryBackend reference.
  """

  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp assert_match(arm_out, ref_out, tol \\ 1.0e-5) do
    assert Nx.shape(arm_out) == Nx.shape(ref_out)
    diff = Nx.subtract(ref(arm_out), ref_out) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff <= tol, "diff #{diff} exceeded tol #{tol}"
  end

  describe "unary math (inputs in [0, 1])" do
    @inputs_unit Nx.tensor([0.1, 0.5, 0.7, 0.9])

    for op <- [:asin] do
      test "#{op}/1 on [0, 1]" do
        ref = apply(Nx, unquote(op), [@inputs_unit])
        out = apply(Nx, unquote(op), [arm(@inputs_unit)])
        assert_match(out, ref, 1.0e-4)
      end
    end
  end

  describe "unary math (general range)" do
    @inputs Nx.tensor([0.1, 0.5, 1.0, 2.0])

    for op <- [:exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh,
               :sqrt, :rsqrt, :atan, :erf, :negate, :abs, :sign,
               :floor, :ceil, :round] do
      test "#{op}/1" do
        ref = apply(Nx, unquote(op), [@inputs])
        out = apply(Nx, unquote(op), [arm(@inputs)])
        assert_match(out, ref, 1.0e-4)
      end
    end
  end

  describe "binary math" do
    @a Nx.tensor([1.0, 2.0, 3.0, 4.0])
    @b Nx.tensor([0.5, 1.5, 2.5, 3.5])

    for op <- [:add, :subtract, :multiply, :divide, :pow, :min, :max,
               :remainder, :atan2] do
      test "#{op}/2" do
        ref = apply(Nx, unquote(op), [@a, @b])
        out = apply(Nx, unquote(op), [arm(@a), arm(@b)])
        assert_match(out, ref, 1.0e-4)
      end
    end
  end

  describe "comparison" do
    @a Nx.tensor([1.0, 2.0, 3.0])
    @b Nx.tensor([1.0, 3.0, 2.0])

    for op <- [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] do
      test "#{op}/2" do
        ref = apply(Nx, unquote(op), [@a, @b])
        out = apply(Nx, unquote(op), [arm(@a), arm(@b)])
        assert ref(out) |> Nx.to_flat_list() == Nx.to_flat_list(ref)
      end
    end
  end

  describe "reductions" do
    @x Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    for op <- [:sum, :reduce_max, :reduce_min, :mean] do
      test "#{op}/1 (full)" do
        ref = apply(Nx, unquote(op), [@x])
        out = apply(Nx, unquote(op), [arm(@x)])
        assert_match(out, ref)
      end

      test "#{op}/2 axis = 0" do
        ref = apply(Nx, unquote(op), [@x, [axes: [0]]])
        out = apply(Nx, unquote(op), [arm(@x), [axes: [0]]])
        assert_match(out, ref)
      end

      test "#{op}/2 axis = 1" do
        ref = apply(Nx, unquote(op), [@x, [axes: [1]]])
        out = apply(Nx, unquote(op), [arm(@x), [axes: [1]]])
        assert_match(out, ref)
      end
    end
  end

  describe "shape manipulation" do
    @x Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    test "reshape" do
      out = Nx.reshape(arm(@x), {6})
      assert ref(out) |> Nx.to_flat_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "transpose default (reverses axes)" do
      out = Nx.transpose(arm(@x))
      assert Nx.shape(out) == {3, 2}
      assert_match(out, Nx.transpose(@x))
    end

    test "transpose with explicit axes" do
      x = Nx.iota({2, 3, 4}, type: :f32)
      out = Nx.transpose(arm(x), axes: [2, 0, 1])
      assert_match(out, Nx.transpose(x, axes: [2, 0, 1]))
    end

    test "broadcast scalar to shape" do
      x = Nx.tensor(2.0)
      out = Nx.broadcast(arm(x), {2, 3})
      assert_match(out, Nx.broadcast(x, {2, 3}))
    end

    test "broadcast 1-D to 2-D" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      out = Nx.broadcast(arm(x), {2, 3})
      assert_match(out, Nx.broadcast(x, {2, 3}))
    end

    test "concatenate along axis 0" do
      a = Nx.tensor([[1.0, 2.0]])
      b = Nx.tensor([[3.0, 4.0]])
      out = Nx.concatenate([arm(a), arm(b)], axis: 0)
      assert_match(out, Nx.concatenate([a, b], axis: 0))
    end

    test "concatenate along axis -1" do
      a = Nx.tensor([[1.0, 2.0]])
      b = Nx.tensor([[3.0, 4.0]])
      out = Nx.concatenate([arm(a), arm(b)], axis: -1)
      assert_match(out, Nx.concatenate([a, b], axis: -1))
    end

    test "stack two tensors into a higher dim" do
      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0, 4.0])
      out = Nx.stack([arm(a), arm(b)])
      assert_match(out, Nx.stack([a, b]))
    end
  end

  describe "selection" do
    test "select with broadcastable mask" do
      mask = Nx.tensor([true, false, true])
      on_true = Nx.tensor([1.0, 2.0, 3.0])
      on_false = Nx.tensor([10.0, 20.0, 30.0])

      out = Nx.select(arm(mask), arm(on_true), arm(on_false))
      ref_out = Nx.select(mask, on_true, on_false)
      assert ref(out) |> Nx.to_flat_list() == Nx.to_flat_list(ref_out)
    end
  end

  describe "dtype conversions" do
    test "f32 → s32" do
      x = Nx.tensor([1.7, 2.3, -0.5])
      out = Nx.as_type(arm(x), :s32)
      assert ref(out) |> Nx.to_flat_list() == Nx.as_type(x, :s32) |> Nx.to_flat_list()
    end

    test "s64 → f32" do
      x = Nx.tensor([1, 2, 3], type: :s64)
      out = Nx.as_type(arm(x), :f32)
      assert_match(out, Nx.as_type(x, :f32))
    end
  end

  describe "argmax / argmin" do
    test "argmax 1-D" do
      x = Nx.tensor([1.0, 5.0, 2.0, 4.0])
      out = Nx.argmax(arm(x))
      assert Nx.to_number(out) == 1
    end

    test "argmax 2-D along axis 1" do
      x = Nx.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]])
      out = Nx.argmax(arm(x), axis: 1) |> ref()
      assert Nx.to_flat_list(out) == [1, 2]
    end
  end

  describe "clip" do
    test "clip f32 to a tighter range" do
      x = Nx.tensor([-5.0, 0.5, 10.0])
      out = Nx.clip(arm(x), -1.0, 1.0)
      assert_match(out, Nx.clip(x, -1.0, 1.0))
    end
  end

  describe "edge case shapes" do
    test "empty axis-of-1 reduction" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      assert_match(Nx.sum(arm(x), axes: [-1]), Nx.sum(x, axes: [-1]))
    end

    test "scalar add" do
      x = Nx.tensor(5.0)
      out = Nx.add(arm(x), arm(Nx.tensor(3.0)))
      assert Nx.to_number(ref(out)) == 8.0
    end

    test "1-element tensor binary op" do
      x = Nx.tensor([7.0])
      out = Nx.multiply(arm(x), arm(Nx.tensor([2.0])))
      assert ref(out) |> Nx.to_flat_list() == [14.0]
    end
  end
end
