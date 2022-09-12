

####################################
# INVERSE LINK FUNCTIONS
####################################

function normal_invlink(Z::AbstractArray)
    return Z
end


function bernoulli_invlink(Z::AbstractArray)
    # We shrink the value slightly toward 0.5
    # in order to prevent NaNs.
    return 0.5f0 .+ (0.9999f0 .*(1 ./ (1 .+ exp.(-Z)) .- 0.5))
end


function ChainRules.rrule(::typeof(bernoulli_invlink), Z)
    A = bernoulli_invlink(Z)

    function bernoulli_invlink_pullback(A_bar)
        return ChainRules.NoTangent(), A_bar .* (A .* (1 .- A))
    end
    return A, bernoulli_invlink_pullback
end


function poisson_invlink(Z::AbstractArray)
    return exp.(Z)
end



INVLINK_FUNCTION_MAP = Dict("normal"=>normal_invlink,
                            "bernoulli"=>bernoulli_invlink,
                            "poisson"=>poisson_invlink,
                           )


####################################
# LOSS FUNCTIONS
####################################

function normal_loss(A::AbstractArray, D::AbstractArray) 
    return 0.5f0.*(A .- D).^2
end

function ChainRules.rrule(::typeof(normal_loss), A, D) 
   
    diff = A .- D

    function normal_loss_pullback(loss_bar)
        return ChainRules.NoTangent(), loss_bar.*diff, ChainRules.NoTangent()
    end

    return 0.5f0.*(diff.^2), normal_loss_pullback 
end


function bernoulli_loss(A::AbstractArray, D::AbstractArray)
    loss = -D .* log.(A) .- (1 .- D) .* log.( 1 .- A)
    return loss
end


function ChainRules.rrule(::typeof(bernoulli_loss), A, D)
    
    loss = bernoulli_loss(A,D)

    function bernoulli_loss_pullback(loss_bar)
        A_bar = loss_bar .* (-D./A .+ (1 .- D)./(1 .- A))
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent()
    end
    return loss, bernoulli_loss_pullback 
end


function poisson_loss(A::AbstractArray, D::AbstractArray) 
    return A .- D.*log.(A)
end


function ChainRules.rrule(::typeof(poisson_loss), A, D)
    
    loss = poisson_loss(A, D)

    function poisson_loss_pullback(loss_bar)
        A_bar = loss_bar .* (1 .- D ./ A)
        return ChainRules.NoTangent(), A_bar, ChainRules.NoTangent()
    end

    return loss, poisson_loss_pullback 
end


LOSS_FUNCTION_MAP = Dict("normal"=>normal_loss,
                         "bernoulli"=>bernoulli_loss,
                         "poisson"=>poisson_loss,
                         )

