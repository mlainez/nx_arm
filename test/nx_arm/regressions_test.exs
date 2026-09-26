defmodule NxArm.RegressionsTest do
  use ExUnit.Case, async: true

  defp arm(list, type \\ :f32), do: Nx.tensor(list, type: type, backend: NxArm.Backend)
  defp bin(list, type \\ :f32), do: Nx.tensor(list, type: type, backend: Nx.BinaryBackend)

  defp assert_close(got, want, tol \\ 1.0e-5) do
    assert Nx.shape(got) == Nx.shape(want)
    got = Nx.backend_copy(got, Nx.BinaryBackend)
    want = Nx.backend_copy(want, Nx.BinaryBackend)
    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(got, want)))) <= tol
  end

  describe "remainder" do
    test "follows the sign of the dividend, like Nx.BinaryBackend" do
      a = [-7.5, -1.0, 0.0, 3.25, 7.5]
      b = [2.0, 3.0, 5.0, -2.0, -2.0]
      assert_close(Nx.remainder(arm(a), arm(b)), Nx.remainder(bin(a), bin(b)))
    end

    test "works with a scalar operand on either side" do
      a = [-7.5, -1.0, 3.25, 7.5]
      assert_close(Nx.remainder(arm(a), 2.0), Nx.remainder(bin(a), 2.0))
      assert_close(Nx.remainder(10.0, arm([3.0, -4.0])), Nx.remainder(10.0, bin([3.0, -4.0])))
    end
  end

  describe "atan2" do
    test "works with a scalar operand on either side" do
      a = [-1.0, 0.5, 2.0]
      assert_close(Nx.atan2(arm(a), 1.0), Nx.atan2(bin(a), 1.0))
      assert_close(Nx.atan2(1.0, arm(a)), Nx.atan2(1.0, bin(a)))
    end
  end

  describe "fft / ifft" do
    defp signal(shape) do
      re = Nx.iota(shape, type: :f32) |> Nx.divide(3) |> Nx.sin()
      Nx.complex(re, Nx.cos(re))
    end

    test "1-D matches Nx.BinaryBackend" do
      x = signal({16})
      assert_close(Nx.fft(Nx.backend_copy(x, NxArm.Backend)), Nx.fft(x), 1.0e-3)
      assert_close(Nx.ifft(Nx.backend_copy(x, NxArm.Backend)), Nx.ifft(x), 1.0e-3)
    end

    test "rank-2 input transforms each row independently" do
      x = signal({3, 8})
      assert_close(Nx.fft(Nx.backend_copy(x, NxArm.Backend)), Nx.fft(x), 1.0e-3)
    end

    test "honours :length" do
      x = signal({8})
      assert_close(Nx.fft(Nx.backend_copy(x, NxArm.Backend), length: 16), Nx.fft(x, length: 16), 1.0e-3)
    end
  end
end
