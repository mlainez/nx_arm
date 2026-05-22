{application,cpu_governor,
             [{modules,['Elixir.CpuGovernor','Elixir.CpuGovernor.Performance',
                        'Elixir.CpuGovernor.Topology']},
              {optional_applications,[]},
              {applications,[kernel,stdlib,elixir,logger]},
              {description,"CPU governor scoping + big.LITTLE topology detection for Nerves devices"},
              {registered,[]},
              {vsn,"0.1.0"}]}.
