defmodule NxArm.BackendPathsTest do
  @moduledoc """
  Targeted coverage for `NxArm.Backend` paths that the conformance
  suite doesn't exercise — matmul shape variants, creation
  callbacks (eye / iota / constant), backend_transfer, the inspect
  hook, etc.
  """

  use ExUnit.Case, async: true

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  describe "creation callbacks" do
    test "eye/2 produces an identity matrix" do
      m = Nx.eye(4, type: :f32, backend: NxArm.Backend)
      assert Nx.shape(m) == {4, 4}
      assert ref(m) |> Nx.to_flat_list() == [
               1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0
             ]
    end

    test "iota/2 produces a ramp" do
      x = Nx.iota({5}, type: :f32, backend: NxArm.Backend)
      assert ref(x) |> Nx.to_flat_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    end

    test "iota/2 along axis 1 in 2-D" do
      x = Nx.iota({2, 3}, type: :f32, axis: 1, backend: NxArm.Backend)
      assert ref(x) |> Nx.to_flat_list() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    end

    test "tensor with explicit backend constant" do
      x = Nx.tensor(7.0, backend: NxArm.Backend)
      assert ref(x) |> Nx.to_number() == 7.0
    end
  end

  describe "backend_transfer + backend_copy" do
    test "round-trip through NxArm.Backend preserves shape + values" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      armed = arm(x)
      back = ref(armed)
      assert Nx.shape(back) == Nx.shape(x)
      assert Nx.to_flat_list(back) == Nx.to_flat_list(x)
    end

    test "backend_deallocate is a no-op" do
      x = arm(Nx.tensor([1.0, 2.0]))
      # The Nx public API for releasing a tensor is Nx.backend_deallocate/1
      assert :ok = Nx.backend_deallocate(x)
    end

    test "Nx.to_binary on an NxArm tensor returns the same bytes as on BinaryBackend" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
      assert Nx.to_binary(arm(x)) == Nx.to_binary(x)
    end
  end

  describe "matmul shape variants" do
    test "2-D × 2-D right-transposed (Axon Dense layout: W is {N, K})" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      # weight as {n, k} -> Nx.dot(x, [1], w, [1]) is x @ w^T
      w = Nx.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
      arm_out = Nx.dot(arm(x), [1], arm(w), [1])
      ref_out = Nx.dot(x, [1], w, [1])

      diff = Nx.subtract(ref(arm_out), ref_out) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "3-D × 2-D (B, M, K) × (K, N) — Bumblebee Linear pattern" do
      x = Nx.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
      w = Nx.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
      arm_out = Nx.dot(arm(x), [-1], arm(w), [0])
      ref_out = Nx.dot(x, [-1], w, [0])

      assert Nx.shape(arm_out) == Nx.shape(ref_out)
      diff = Nx.subtract(ref(arm_out), ref_out) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "3-D × 2-D right-transposed" do
      x = Nx.tensor([[[1.0, 2.0, 3.0]]])
      w = Nx.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
      arm_out = Nx.dot(arm(x), [-1], arm(w), [1])
      ref_out = Nx.dot(x, [-1], w, [1])

      diff = Nx.subtract(ref(arm_out), ref_out) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end
  end

  describe "squeeze + put_in callbacks" do
    test "squeeze removes a 1-dim axis" do
      x = Nx.tensor([[[1.0], [2.0]]]) |> arm()
      out = Nx.squeeze(x)
      assert Nx.shape(out) == {2}
      assert ref(out) |> Nx.to_flat_list() == [1.0, 2.0]
    end
  end
end
