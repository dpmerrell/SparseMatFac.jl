
using SparseMatFac, Test, Flux, ScikitLearnBase

SMF = SparseMatFac

function model_assembly_tests()

    M = 10
    N = 20
    K = 4

    @testset "Model assembly tests" begin

        model = SparseMatFacModel(M,N,K; noise_model="bernoulli")
        @test size(model.X) == (K,M)
        @test size(model.Y) == (K,N)
        @test model.lambda_X == 1.0
        @test model.lambda_Y == 1.0
        @test model.noise_model == "bernoulli"
    end
 
end


function model_core_tests()

    M = 10
    N = 20
    K = 4
    noise = "bernoulli"
    model = SparseMatFacModel(M,N,K; noise_model=noise)
    
    I = rand(1:M, N)
    J = collect(1:N)
    V = rand([0.0, 1.0], N)
 
    # Code organization is kind of awkward; 
    # have to construct these explicitly
    X_view = view(model.X, :, I)
    Y_view = view(model.Y, :, J)
    row_transform_view = view(model.row_transform, I)
    col_transform_view = view(model.col_transform, J)

    invlink_fn = SMF.INVLINK_FUNCTION_MAP[noise] 
    loss_fn = SMF.LOSS_FUNCTION_MAP[noise]

    @testset "Model core tests" begin
        forward_values = SMF.forward(X_view, Y_view,
                                    row_transform_view,
                                    col_transform_view)
        @test size(forward_values) == size(V)

        nll = SMF.neg_log_likelihood(X_view, Y_view,
                                     row_transform_view, col_transform_view,
                                     V, invlink_fn, loss_fn)

        @test nll > 0.0
    end
end


function model_fit_tests()

    M = 10
    N = 20
    K = 1
    
    I = rand(1:M, N)
    J = collect(1:N)
    V = rand([0.0, 1.0], N)


    @testset "CPU model fit tests" begin
        model = SparseMatFacModel(M, N, K; noise_model="bernoulli", 
                                           Y_reg=y->0.1*sum(y.*y))
        X_start = copy(model.X)
        Y_start = copy(model.Y)    

        fit!(model, I, J, V; max_iter=5000, lr=0.3, verbosity=-1)
        @test !isapprox(model.X, X_start)
        @test !isapprox(model.Y, Y_start) 
    end
    
    @testset "GPU model fit tests" begin
        model = SparseMatFacModel(M, N, K; noise_model="poisson", 
                                           Y_reg=y->0.1*sum(y.*y))
        X_start = copy(model.X)
        Y_start = copy(model.Y)    
        
        model_gpu = gpu(model)
        I_gpu = gpu(I)
        J_gpu = gpu(J)
        V_gpu = gpu(V)
        
        fit!(model_gpu, I_gpu, J_gpu, V_gpu; max_iter=5000, lr=0.3, verbosity=-1)
        model = cpu(model_gpu)
        @test !isapprox(model.X, X_start)
        @test !isapprox(model.Y, Y_start) 
    end

end


function model_io_tests()
    
    M = 10
    N = 20
    K = 1

    @testset "Model IO tests" begin
        model = SparseMatFacModel(M, N, K; noise_model="bernoulli", 
                                           Y_reg=y->0.1*sum(y.*y))
        save_model(model, "test.hdf")
        recovered_model = load_model("test.hdf")

        @test isapprox(recovered_model.X, model.X)
        @test isapprox(recovered_model.Y, model.Y) 
    end
end


function main()

    model_assembly_tests()
    model_core_tests()
    model_fit_tests()
    model_io_tests()

end

main()


