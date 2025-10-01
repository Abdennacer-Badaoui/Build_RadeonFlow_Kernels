{
  description = "Flake for GEMM kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/rocm-per-source-arches";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:

    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
