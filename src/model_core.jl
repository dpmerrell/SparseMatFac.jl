

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


#"""
#    Whenever you implement a custom row- or column- transformation
#    for a SparseMatFacModel, you must implement a `collect_view_gradients`
#    function for the transformation.
#
#    The function must (1) receive the gradient for the _view_ of a layer
#    and (2) correctly map the _view's_ gradient to a gradient for the 
#    original layer.
#"""
#function collect_view_gradients(layer_view, view_grads)
#    error(string("collect_view_gradients(...) not implemented for ", typeof(layer_view)))
#end

