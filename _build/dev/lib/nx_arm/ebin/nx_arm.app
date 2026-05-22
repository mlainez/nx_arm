{application,nx_arm,
             [{modules,['Elixir.NxArm','Elixir.NxArm.Backend',
                        'Elixir.NxArm.Compiler']},
              {optional_applications,[rustler]},
              {applications,[kernel,stdlib,elixir,logger,nx,arm_ai,rustler,
                             rustler_precompiled]},
              {description,"Nx backend + Nx-tensor model wrappers for ARM CPUs, built on the arm_ai NEON inference NIF"},
              {registered,[]},
              {vsn,"0.2.0"}]}.
