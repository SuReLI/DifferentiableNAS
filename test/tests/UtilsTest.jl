using DifferentiableNAS
using Distributions
using Flux

@testset "Flip test" begin
    batch = rand(Float32,32,32,3,8)
    orig = copy(batch)
    flips = flip_batch!(batch)
    for i=1:size(batch,4)
        if flips[i]
            @test orig[:,:,:,i]==batch[:,end:-1:1,:,i]
        else
            @test orig[:,:,:,i]==batch[:,:,:,i]
        end
    end
end

@testset "Shift test" begin
    batch = rand(Float32,32,32,3,8)
    orig = copy(batch)
    shifts = shift_batch!(batch)
    for i=1:size(batch,4)
        shiftx = shifts[i,1]
        shifty = shifts[i,2]
        @test batch[5-shiftx:28-shiftx,5-shifty:28-shifty,:,i]==orig[5:28,5:28,:,i]
        #@test batch[:,:,:,i]
    end
end

@testset "Cutout test" begin
    batch = rand(Float32,32,32,3,8)
    orig = copy(batch)
    cutouts = cutout_batch!(batch, 16)
end
