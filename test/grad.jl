using DifferentiableNAS
using Flux
using Zygote
#using CUDA

function gshow(x)
         @show typeof(x) size(x)
         x
       end
Zygote.@adjoint function gshow(x)
       gshow(x), dx -> begin
         @show typeof(dx) size(dx)
         tuple(dx)
       end
       end

mo = MixedOp(4,1,1)  |> gpu
data = rand(Float32,8,8,4,2)  |> gpu
a = rand(Float32, 8)  |> gpu
g = gradient((x,α) -> sum(mo(x, a)), data, a)
#stacktrace(backtrace())
#ga = gradient(params(a)) do
#        sum(gshow(mo(data,a)))
#    end
