

function forward(X_view, X_b_view, Y_view, Y_b_view)
    return vec(sum(X_view .* Y_view; dims=1)) .+ X_b_view .+ Y_b_view
end


function neg_log_likelihood(X_view, X_b_view, 
                            Y_view, Y_b_view, 
                            V, link_fn, loss_fn)
    
    Z_values = link_fn(forward(X_view, X_b_view, 
                               Y_view, Y_b_view))
    
    loss = sum(loss_fn(Z_values, V))

    return loss
end


function mat_reg(X::AbstractMatrix, X_reg::AbstractVector)

    loss = 0.0f0
    for k=1:length(X_reg)
        loss += 0.5f0*dot(X[k,:], X_reg[k]*X[k,:])
    end
    return loss
end

function ChainRules.rrule(::typeof(mat_reg), X, X_reg)
    
    loss = 0.0f0
    X_grad = zero(X)
    for i=1:length(X_reg)
        X_grad[i,:] .= X_reg[i]*X[i,:]
        loss += 0.5f0*dot(X[i,:], X_grad[i,:])
    end

    function mat_reg_pullback(loss_bar)
        return ChainRules.NoTangent(), loss_bar.*X_grad, ChainRulesCore.ZeroTangent()
    end

    return loss, mat_reg_pullback

end

function vec_reg(x::AbstractVector, x_reg::AbstractMatrix)
    return 0.5f0*dot(x, x_reg*x)
end

