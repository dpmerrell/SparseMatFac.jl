

import ScikitLearnBase: fit!


function fit!(model::SparseMatFacModel, 
              I::Vector{Int}, J::Vector{Int},
              V::Vector{<:Number};
              lr=0.01, max_iter=1000, 
              abs_tol=1e-12, rel_tol=1e-9)

    V_view = CUDA.CuVector(V)
    K = size(model.X, 1)
    M = size(model.X_reg[1], 1)
    N = size(model.Y_reg[1], 1)
    nnz = length(V)

    # Construct some observation-wise views of X and Y
    X_view = view(model.X, :, I)
    X_b_view = view(model.X_b, I)
    Y_view = view(model.Y, :, J)
    Y_b_view = view(model.Y_b, J)

    # These matrices allow us to efficiently
    # combine gradients belonging to the same
    # columns of X or Y.
    nz_to_i = CuSparseMatrixCSC{Float32}(sparse(I, 1:nnz, ones(nnz), M, nnz))
    nz_to_j = CuSparseMatrixCSC{Float32}(sparse(J, 1:nnz, ones(nnz), N, nnz))

    link_fn = LINK_FUNCTION_MAP[model.loss]
    loss_fn = LOSS_FUNCTION_MAP[model.loss]

    # Preallocate arrays to store gradients
    adagrad_epsilon = 1e-9
    X_grad = zero(model.X)
    X_grad_sq = zero(model.X) .+ adagrad_epsilon 
    X_b_grad = zero(model.X_b)
    X_b_grad_sq = zero(model.X_b) .+ adagrad_epsilon
    Y_grad = zero(model.Y)
    Y_grad_sq = zero(model.Y) .+ adagrad_epsilon
    Y_b_grad = zero(model.Y_b)
    Y_b_grad_sq = zero(model.Y_b) .+ adagrad_epsilon

    # Curry some functions for autograd
    curried_likelihood = (X_view, Y_view, 
                          X_b_view, Y_b_view) -> neg_log_likelihood(X_view, X_b_view,
                                                                    Y_view, Y_b_view,
                                                                    V_view, 
                                                                    link_fn, loss_fn)
    curried_X_prior = X -> mat_reg(X, model.X_reg)
    curried_X_b_prior = X_b -> vec_reg(X_b, model.X_b_reg)
    curried_Y_prior = Y -> mat_reg(Y, model.Y_reg)
    curried_Y_b_prior = Y_b -> vec_reg(Y_b, model.Y_b_reg)

    prev_loss = Inf

    for iter=1:max_iter
        # Compute log-likelihood gradients w.r.t. observation-wise
        # views of X and Y
        likelihood_loss, 
        (X_ll_grad, Y_ll_grad,
         X_b_ll_grad, Y_b_ll_grad) = withgradient(curried_likelihood, 
                                                  X_view, Y_view,
                                                  X_b_view, Y_b_view)

        # Map the observation-wise gradients on to X and Y
        #X_grad .= transpose(transpose(nz_to_i) * transpose(X_ll_grad)) 
        #X_b_grad .= transpose(transpose(nz_to_i) * X_b_ll_grad) 
        #Y_grad .= transpose(transpose(nz_to_j) * transpose(Y_ll_grad))
        #Y_b_grad .= transpose(transpose(nz_to_j) * transpose(Y_b_ll_grad))
        X_grad .= transpose(nz_to_i * transpose(X_ll_grad))
        X_b_grad .= nz_to_i * X_b_ll_grad
        Y_grad .= transpose(nz_to_j * transpose(Y_ll_grad))
        Y_b_grad .= nz_to_j * Y_b_ll_grad

        # Compute regularization gradients w.r.t. X and Y
        X_reg_loss, (X_reg_grad,) = withgradient(curried_X_prior, model.X)
        X_b_reg_loss, (X_b_reg_grad,) = withgradient(curried_X_b_prior, model.X_b)
        Y_reg_loss, (Y_reg_grad,) = withgradient(curried_Y_prior, model.Y)
        Y_b_reg_loss, (Y_b_reg_grad,) = withgradient(curried_Y_b_prior, model.Y_b)

        # Update gradients and sum-squared gradients
        X_grad .+= X_reg_grad
        X_b_grad .+= X_b_reg_grad
        Y_grad .+= Y_reg_grad
        Y_b_grad .+= Y_b_reg_grad

        X_grad_sq .+= (X_grad.*X_grad)
        X_b_grad_sq .+= (X_b_grad.*X_b_grad)
        Y_grad_sq .+= (Y_grad.*Y_grad)
        Y_b_grad_sq .+= (Y_b_grad.*Y_b_grad)

        # Adagrad update
        model.X .-= lr.*X_grad ./ sqrt.(X_grad_sq)
        model.X_b .-= lr.*X_b_grad ./ sqrt.(X_b_grad_sq)
        model.Y .-= lr.*Y_grad ./ sqrt.(Y_grad_sq)
        model.Y_b .-= lr.*Y_b_grad ./ sqrt.(Y_b_grad_sq)

        # Compute total loss
        loss = (likelihood_loss + X_reg_loss + Y_reg_loss
                                + X_b_reg_loss + Y_b_reg_loss)
        println(string("(", iter, ")\tLOSS: ", loss))

        loss_delta = loss - prev_loss
        if (loss_delta > -abs_tol)
            println(string("Reached abs_tol=", abs_tol, "; terminating"))
            break
        elseif (abs(loss_delta/loss) < rel_tol)
            println(string("Reached rel_tol=", rel_tol, "; terminating")) 
            break
        end
        prev_loss = loss
    end

end


