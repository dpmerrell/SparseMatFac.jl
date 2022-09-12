module SparseMatFac

using SparseArrays, LinearAlgebra, ScikitLearnBase,
      Flux, Zygote, ChainRules, ChainRulesCore, BSON, CUDA

include("util.jl")
include("model.jl")
include("noise_models.jl")
include("model_core.jl")
include("viewable.jl")
include("fit.jl")
include("impute.jl")

end # module
