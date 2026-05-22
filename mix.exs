defmodule NxArm.MixProject do
  use Mix.Project

  @version "0.2.0"

  def project do
    [
      app: :nx_arm,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "NxArm",
      description:
        "Nx backend + Nx-tensor model wrappers for ARM CPUs, built on the arm_ai NEON inference NIF",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:nx, "~> 0.9"},
      # arm_ai owns the NIF. nx_arm provides the Nx.Backend impl
      # and Nx.Defn.Compiler over arm_ai's primitives.
      {:arm_ai, path: "../arm_ai"},
      # nx_arm doesn't itself bridge a model, but rustler must be
      # visible at compile time because :arm_ai pulls in a build of
      # the NIF when no precompiled tarball matches the runtime
      # triple (i.e., when developing locally).
      {:rustler, "~> 0.36", optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:axon, "~> 0.7", only: [:test]},
      {:bumblebee, "~> 0.6", only: [:test]},
      {:stream_data, "~> 1.1", only: [:test]}
    ]
  end

  defp package do
    [
      name: :nx_arm,
      licenses: ["Apache-2.0"],
      files: ~w(lib mix.exs README.md),
      links: %{"GitHub" => "https://github.com/marclainez/nx_arm"}
    ]
  end
end
