

function forward(X_view, Y_view, row_transform_view, col_transform_view)
    return col_transform_view( row_transform_view( vec(sum(X_view .* Y_view; dims=1)) ))
end


function neg_log_likelihood(X_view, Y_view, 
                            row_transform_view, col_transform_view, 
                            V, invlink_fn, loss_fn)
    
    Z_values = invlink_fn(forward(X_view, Y_view,
                                  row_transform_view, col_transform_view)) 
        
    loss = sum(loss_fn(Z_values, V))

    return loss
end



