module SparseMatFac

using SparseArrays, CUDA, LinearAlgebra, ScikitLearnBase,
      Zygote, ChainRules, ChainRulesCore, HDF5

include("typedefs.jl")
include("model.jl")
include("noise_models.jl")
include("model_core.jl")
include("fit.jl")
include("io.jl")

end # module
