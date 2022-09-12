

import ScikitLearnBase: fit!


function fit!(model::SparseMatFacModel, 
              I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector{<:Number};
              lr=0.01, opt=nothing, max_iter=1000, abs_tol=1e-12, rel_tol=1e-9, max_tol_iter=3,
              verbosity=1)

    K = size(model.X, 1)
    M = size(model.X, 2)
    N = size(model.Y, 2)
    nnz = length(V)

    # Construct some useful views of the model parameters
    X_view = view(model.X, :, I)
    Y_view = view(model.Y, :, J)
    row_transform_view = view(model.row_transform, I)
    col_transform_view = view(model.col_transform, J)

    # These matrices allow us to efficiently
    # combine gradients belonging to the same
    # columns of X or Y.
    nz_to_i = sparse(I, 1:nnz, ones(nnz), M, nnz)
    nz_to_j = sparse(J, 1:nnz, ones(nnz), N, nnz)
    if typeof(model.X) <: CuArray
        nz_to_i = gpu(nz_to_i)
        nz_to_j = gpu(nz_to_j)
    end

    invlink_fn = INVLINK_FUNCTION_MAP[model.noise_model]
    loss_fn = LOSS_FUNCTION_MAP[model.noise_model]
    
    if opt == nothing
        opt = Flux.Optimise.AdaGrad(lr)
    end

    # Preallocate arrays to store gradients
    X_grad = zero(model.X)
    Y_grad = zero(model.Y)

    # Curry some functions for autograd
    curried_likelihood = (X_view, Y_view, 
                          row_trans_view, 
                          col_trans_view) -> neg_log_likelihood(X_view, Y_view, 
                                                                row_trans_view,
                                                                col_trans_view,
                                                                V, invlink_fn, loss_fn)

    curried_X_prior = X -> model.X_reg(X)
    curried_Y_prior = Y -> model.Y_reg(Y)
    curried_rowtrans_prior = rt -> model.row_transform_reg(rt)
    curried_coltrans_prior = ct -> model.col_transform_reg(ct)

    prev_loss = Inf
    tol_iter = 0
    start_time = time()
    for iter=1:max_iter
        # Compute log-likelihood gradients w.r.t. observation-wise
        # views of X and Y
        likelihood_loss, 
        (X_v_grad, Y_v_grad,
         row_trans_v_grad, col_trans_v_grad) = withgradient(curried_likelihood, 
                                                            X_view, Y_view,
                                                            row_transform_view, 
                                                            col_transform_view)

        # Map the observation-wise gradients on to X and Y
        X_grad .= transpose(nz_to_i * transpose(X_v_grad))
        Y_grad .= transpose(nz_to_j * transpose(Y_v_grad))
        row_trans_grad = collect_view_gradients(model.row_transform, row_trans_v_grad)
        col_trans_grad = collect_view_gradients(model.col_transform, col_trans_v_grad)

        # Compute regularization gradients w.r.t. X and Y
        X_reg_loss, (X_reg_grad,) = withgradient(curried_X_prior, model.X)
        Y_reg_loss, (Y_reg_grad,) = withgradient(curried_Y_prior, model.Y)
        row_trans_reg_loss, (row_trans_reg_grad,) = withgradient(curried_rowtrans_prior, model.row_transform)
        col_trans_reg_loss, (col_trans_reg_grad,) = withgradient(curried_coltrans_prior, model.col_transform)

        # Update gradients and sum-squared gradients
        binop!(+, X_grad, X_reg_grad)
        binop!(+, Y_grad, Y_reg_grad)
        binop!(+, row_trans_grad, row_trans_reg_grad)
        binop!(+, col_trans_grad, col_trans_reg_grad)

        # Adagrad updates
        update!(opt, model.X, X_grad)
        update!(opt, model.Y, Y_grad)
        update!(opt, model.row_transform, row_trans_grad)
        update!(opt, model.col_transform, col_trans_grad)

        # Compute total loss
        loss = (likelihood_loss + X_reg_loss + Y_reg_loss
                                + row_trans_reg_loss + col_trans_reg_loss)
        cur_time = time()
        verbose_print("(", iter, ")\tLOSS: ", round(loss, digits=6), "\tElapsed time: ", round(Int, cur_time - start_time), "s\n"; verbosity=verbosity)

        loss_delta = loss - prev_loss
        if (loss_delta > -abs_tol)
            tol_iter += 1
            verbose_print(tol_iter,"/", max_tol_iter, "\tabs_tol=", abs_tol, "\n"; verbosity=verbosity)
        elseif (abs(loss_delta/loss) < rel_tol)
            tol_iter += 1
            verbose_print(tol_iter,"/", max_tol_iter, " rel_tol=", rel_tol, "\n"; verbosity=verbosity) 
        else
            tol_iter = 0
        end
        
        prev_loss = loss
        if tol_iter >= max_tol_iter
            verbose_print("Reached max termination counter (", max_tol_iter,"). Terminating.\n"; verbosity=verbosity, level=0)
            break    
        end
        if iter >= max_iter
            verbose_print("Reached max_iter (", max_iter, "). Terminating.\n"; verbosity=verbosity, level=0)
        end
    end

    return model
end


