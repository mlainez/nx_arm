defmodule NxArm.SoftmaxTest do
  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  describe "NxArm.softmax/2" do
    test "1-D probabilities sum to 1" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32) |> arm()
      out = NxArm.softmax(x)
      s = out |> ref() |> Nx.sum() |> Nx.to_number()
      assert_in_delta s, 1.0, 1.0e-5
    end

    test "2-D row-wise softmax — each row sums to 1" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32) |> arm()
      out = NxArm.softmax(x)
      sums = out |> ref() |> Nx.sum(axes: [-1]) |> Nx.to_flat_list()
      Enum.each(sums, fn s -> assert_in_delta s, 1.0, 1.0e-5 end)
    end

    test "matches BinaryBackend reference within tight tolerance" do
      x = Nx.tensor([[0.1, 0.2, 0.3, 0.4]], type: :f32)
      out_arm = NxArm.softmax(arm(x))

      ref_e = Nx.exp(Nx.subtract(x, Nx.reduce_max(x, axes: [-1], keep_axes: true)))
      ref_out = Nx.divide(ref_e, Nx.sum(ref_e, axes: [-1], keep_axes: true))

      diff = Nx.subtract(ref(out_arm), ref_out) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "axis: -1 is the default" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> arm()
      assert ref(NxArm.softmax(x)) == ref(NxArm.softmax(x, axis: -1))
    end

    test "raises on non-last axis (not supported)" do
      x = Nx.tensor([[1.0, 2.0, 3.0]]) |> arm()

      assert_raise ArgumentError, ~r/last axis/, fn ->
        NxArm.softmax(x, axis: 0)
      end
    end

    test "raises on non-f32 input" do
      x = Nx.tensor([[1.0, 2.0]], type: :f64) |> arm()

      assert_raise ArgumentError, ~r/:f32/, fn ->
        NxArm.softmax(x)
      end
    end
  end
end
