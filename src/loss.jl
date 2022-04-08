

function forward(X_view, Y_view, X_b_view, Y_b_view, link_fn)

    Z = vec(sum(X_view .* Y_view , dims=1)) .+ X_b_view .+ Y_b_view
    return link_fn(Z)
end


function log_likelihood(X_view, Y_view, X_b_view, Y_b_view, 
                        D_vec, link_fn, loss_fn)

    A = forward(X_view, Y_view, X_b_view, Y_b_view, link_fn)
    return sum(loss_fn(A, D_vec))
end


function regularization(X::AbstractMatrix, X_reg::Vector{<:AbstractMatrix})

    loss = 0
    for k=1:size(X,1)
        x = X[k,:]
        loss += 0.5*dot(x, dot(X_reg[k], x))
    end

    return loss
end

function regularization(x::AbstractVector, x_reg::AbstractMatrix)
    return 0.5*dot(x, dot(X_reg, x))
end

