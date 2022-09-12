
export impute

"""
    impute(model::SparseMatFacModel, I, J)
 
    use `model` to impute the value at indices given by I, J.
"""
function impute(model::SparseMatFacModel, I::AbstractVector{Int}, J::AbstractVector{Int})

    X_view = view(model.X, :, I)
    Y_view = view(model.Y, :, J)

    row_trans_view = view(model.row_transform, I)
    col_trans_view = view(model.col_transform, I)

    invlink_fn = INVLINK_FUNCTION_MAP[model.noise_model]

    Z = forward(X_view, Y_view, row_trans_view, col_trans_view)

    return invlink_fn(Z)
end


