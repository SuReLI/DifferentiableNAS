include("CIFAR10.jl")

@testset "CIFAR Train/val test" begin
    train, val = get_processed_data(0.5f0, 8)
    
end

@testset "CIFAR Test test" begin
    unbatched = get_test_data()

    batched = get_test_data(1f0, 8)
end
