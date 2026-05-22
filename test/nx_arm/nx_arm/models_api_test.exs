defmodule ArmAI.ModelsApiTest do
  use ExUnit.Case, async: true

  # These tests cover the *contract* every model wrapper presents to
  # the caller. They run on any host (no Nerves toolchain, no NIF
  # features enabled, no real model file) — what they prove is that
  # the Elixir surface area never raises on the unhappy path.
  #
  # The actual inference paths are tested on-device via `mix upload`
  # + the runnable examples in `examples/`.

  describe "Models.LlamaCandle.load/2" do
    test "missing GGUF returns {:error, _}, never raises" do
      assert {:error, _} = ArmAI.LlamaCandle.load("/tmp/__no_such_gguf.gguf")
    end

    test "accepts a :tokenizer keyword without crashing on missing tokenizer.json" do
      # Even if both files are missing the wrapper should fail
      # gracefully — the tokenizer fallback in load/2 is best-effort.
      assert {:error, _} =
               ArmAI.LlamaCandle.load("/tmp/__nope.gguf",
                 tokenizer: "/tmp/__nope_tokenizer.json"
               )
    end
  end

  describe "Models.Onnx.load/1" do
    test "missing file returns {:error, _}" do
      assert {:error, _} = ArmAI.Onnx.load("/tmp/__no_such_file.onnx")
    end
  end

  describe "Models.YOLO.load/2" do
    test "missing file → error tuple" do
      assert {:error, _} = ArmAI.YOLO.load("/tmp/__no_yolo.onnx")
    end

    test "layout option round-trips on the error path" do
      assert {:error, _} = ArmAI.YOLO.load("/tmp/__no.onnx", layout: :v8)
      assert {:error, _} = ArmAI.YOLO.load("/tmp/__no.onnx", layout: :v5)
    end
  end

  # Wrappers that exist only on the development branch where the
  # underlying module is shipped — guarded with `if Code.ensure_loaded?/1`
  # so the test file works against any compilation profile.

  if Code.ensure_loaded?(ArmAI.WhisperCandle) do
    describe "Models.WhisperCandle.load/1" do
      test "missing GGUF returns {:error, _}" do
        assert {:error, _} =
                 ArmAI.WhisperCandle.load(
                   gguf_path: "/tmp/__nope.gguf",
                   tokenizer_path: "/tmp/__nope.json",
                   mel_filters_path: "/tmp/__nope.bin"
                 )
      end
    end
  end

  if Code.ensure_loaded?(ArmAI.SileroVAD) do
    describe "Models.SileroVAD.load/1" do
      test "missing ONNX returns {:error, _}" do
        assert {:error, _} = ArmAI.SileroVAD.load("/tmp/__no_vad.onnx")
      end
    end
  end

  if Code.ensure_loaded?(ArmAI.Piper) do
    describe "Models.Piper.load/2" do
      test "missing ONNX returns {:error, _}" do
        assert {:error, _} = ArmAI.Piper.load("/tmp/__no_piper.onnx")
      end
    end
  end

  describe "Models.Onnx struct contract" do
    test "fields are stable" do
      m = %ArmAI.Onnx{handle: nil, input_names: ["images"], output_names: ["out"]}
      assert m.input_names == ["images"]
      assert m.output_names == ["out"]
    end
  end
end
