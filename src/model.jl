
export SparseMatFacModel, save_model, load_model

mutable struct SparseMatFacModel

    X::AbstractMatrix
    Y::AbstractMatrix

    row_transform::Any  # These transforms may be functions
    col_transform::Any  # or callable structs (with trainable parameters)
    noise_model::String

    X_reg::Any                   # These regularizers may be functions
    Y_reg::Any                   # or callable structs (with trainable
    row_transform_reg::Any       # parameters)
    col_transform_reg::Any

    lambda_X::Number             # Regularizer weights
    lambda_Y::Number
    lambda_row::Number
    lambda_col::Number
end


"""
    SparseMatFacModel(M, N, K; row_transform=identity,
                               col_transform=identity,
                               noise_model::String="normal",
                               X_reg=x->0.0, 
                               Y_reg=x->0.0,
                               row_transform_reg=x->0.0,
                               col_transform_reg=x->0.0,
                               lambda_X=1.0,
                               lambda_Y=1.0,
                               lambda_row=1.0,
                               lambda_col=1.0)

    Build a SparseMatFacModel for an M x N dataset, with K-dimensional factors.
    Supply row- and column-transforms, noise model, regularizers, and 
    regularizer weights as keyword arguments.
"""
function SparseMatFacModel(M::Integer, N::Integer, K::Integer;
                           row_transform=identity,
                           col_transform=identity,
                           noise_model::String="normal",
                           X_reg=x->0.0, 
                           Y_reg=x->0.0,
                           row_transform_reg=x->0.0,
                           col_transform_reg=x->0.0,
                           lambda_X=1.0,
                           lambda_Y=1.0,
                           lambda_row=1.0,
                           lambda_col=1.0)

    X = randn(K,M) ./ sqrt(K) / 10.0
    Y = randn(K,N) ./ sqrt(K) / 10.0

    row_transform = make_viewable(row_transform)
    col_transform = make_viewable(col_transform)

    return SparseMatFacModel(X, Y, row_transform, col_transform, noise_model,
                                   X_reg, Y_reg, 
                                   row_transform_reg, col_transform_reg,
                                   lambda_X, lambda_Y, lambda_row, lambda_col)
end



################################################
# Model file I/O

"""
    save_model(model, filename)

Save `model` to a BSON file located at `filename`.
"""
function save_model(model, filename)
    BSON.@save filename model
end

"""
    load_model(filename)

load a model from the BSON located at `filename`.
"""
function load_model(filename)
    d = BSON.load(filename, @__MODULE__)
    return d[:model]
end


