using DifferentiableNAS
using Distributions
using Flux


@testset "Perturb test" begin
    full_αs = [[2e-3*(rand(8).-0.5) |> f32 |> gpu  for _ in 1:14] for _ in 1:2]
    rn, row, inds, perturbs = DifferentiableNAS.perturb(full_αs)
    @test length(inds) == 8

    partial_αs = [[rand([-Inf32,0],8) |> f32 |> gpu  for _ in 1:14] for _ in 1:2]
    rn, row, inds, perturbs = DifferentiableNAS.perturb(partial_αs)

    half_αs = [[2e-3*(ones(8).-0.5) |> f32 |> gpu  for _ in 1:14], [-Inf32.*ones(8) |> f32 |> gpu  for _ in 1:14]]
    for i in 1:14
        half_αs[2][i][i%8+1] = 1
    end
    rn, row, inds, perturbs = DifferentiableNAS.perturb(half_αs)
    @test rn == 1

    almostdone_αs = [[-Inf32.*ones(8) |> f32 |> gpu  for _ in 1:14] for _ in 1:2]
    for i in 1:14
        almostdone_αs[1][i][i%8+1] = -1
        almostdone_αs[2][i][i%8+1] = 1
    end
    almostdone_αs[2][1][5] = 1
    rn, row, inds, perturbs = DifferentiableNAS.perturb(almostdone_αs)
    @test rn == 2
    @test row == 1
    @test inds[1] == 2
    @test inds[2] == 5

    done_αs = [[-Inf32.*ones(8) |> f32 |> gpu  for _ in 1:14] for _ in 1:2]
    for i in 1:14
        done_αs[1][i][i%8+1] = -1
        done_αs[2][i][i%8+1] = 1
    end
    rn, row, inds, perturbs = DifferentiableNAS.perturb(done_αs)
    @test rn = -1
end
