defmodule NxArm.MixProject do
  use Mix.Project

  @version "0.2.0"

  def project do
    [
      app: :nx_arm,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "NxArm",
      description:
        "Nx backend + Defn compiler for ARM CPUs, built on the arm_ai NEON kernels",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:nx, "~> 0.12.0"},
      # arm_ai owns the NIF. nx_arm provides the Nx.Backend impl
      # and Nx.Defn.Compiler over arm_ai's primitives.
      {:arm_ai, github: "mlainez/arm_ai"},
      {:axon, "~> 0.7", only: [:test]},
      {:bumblebee, "~> 0.6", only: [:test]},
      {:stream_data, "~> 1.1", only: [:test]}
    ]
  end

  defp package do
    [
      name: :nx_arm,
      licenses: ["Apache-2.0"],
      files: ~w(lib mix.exs README.md LICENSE),
      links: %{"GitHub" => "https://github.com/mlainez/nx_arm"}
    ]
  end
end
