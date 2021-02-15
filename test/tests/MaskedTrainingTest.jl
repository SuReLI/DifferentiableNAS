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
    @test rn == -1
end

@testset "ADMM test" begin
    steps = 4
    k = floor(Int, steps^2/2+3*steps/2)
    num_ops = length(PRIMITIVES)-1
    masked_αs = [
        (
            atanh.(
                ones(Float32, num_ops) ./ num_ops +
                2f-3 * (rand(Float32, num_ops) .- 0.5f0),
            )
        ) .* rand(Bernoulli(), num_ops) .* rand(Bernoulli(), num_ops) .*
        rand(Bernoulli(), num_ops) |> f32 for _ = 1:k
    ]
    m = DARTSModelBN()
    m.normal_αs = masked_αs
    display(vcat([tanh.(relu.(n)) for n in m.normal_αs], [tanh.(relu.(n)) for n in m.reduce_αs]))
    for αs in [m.normal_αs, m.reduce_αs]
        for a in αs
            a .= a .* rand(Bernoulli(), num_ops) .* rand(Bernoulli(), num_ops) .*
            rand(Bernoulli(), num_ops)
            display(a)
            a[findall(<=(0),a)] .= -Inf32
            if count(<=(0),a) == length(a)-1
                a[findall(>(0),a)] .= Inf32
            end
        end
    end
    display(vcat([tanh.(relu.(n)) for n in m.normal_αs], [tanh.(relu.(n)) for n in m.reduce_αs]))
end
