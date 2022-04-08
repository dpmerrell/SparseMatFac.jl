
export SparseMatFacModel

mutable struct SparseMatFacModel

    X::AbstractMatrix
    Y::AbstractMatrix

    X_b::AbstractVector
    Y_b::AbstractVector

    X_reg::Vector{<:AbstractMatrix}
    Y_reg::Vector{<:AbstractMatrix}

    X_b_reg::AbstractMatrix
    Y_b_reg::AbstractMatrix

    loss::String
end


function SparseMatFacModel(X_reg::Vector{<:AbstractMatrix}, 
                           Y_reg::Vector{<:AbstractMatrix},
                           X_b_reg::AbstractMatrix,
                           Y_b_reg::AbstractMatrix;
                           loss::String="normal")

    K = length(X_reg)
    @assert K == length(Y_reg)

    M = size(X_reg[1], 1)
    N = size(Y_reg[1], 1)

    X = CUDA.randn(Float32,K,M) ./ Float32(sqrt(K)) ./ 1000.0f0
    X_b = CUDA.zeros(Float32,M)
    Y = CUDA.randn(Float32,K,N) ./ Float32(sqrt(K)) ./ 1000.f0
    Y_b = CUDA.zeros(Float32,N)

    return SparseMatFacModel(X, Y, X_b, Y_b, 
                             X_reg, Y_reg, X_b_reg, Y_b_reg,
                             loss)
end


function SparseMatFacModel(X_reg::AbstractMatrix, 
                           Y_reg::AbstractMatrix, 
                           K::Integer; 
                           loss::String="normal")

    X_reg_v = fill(deepcopy(X_reg), K)
    Y_reg_v = fill(deepcopy(Y_reg), K)

    X_b_reg = deepcopy(X_reg)
    Y_b_reg = deepcopy(Y_reg)

    return SparseMatFacModel(X_reg_v, Y_reg_v, X_b_reg, Y_b_reg;
                             loss=loss)
end



