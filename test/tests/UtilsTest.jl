using DifferentiableNAS
using Distributions
using Flux
using CUDA


@testset "Depthwise test" begin
    layer = DepthwiseConv((3,3),3=>3,stride=1,pad=1,bias=false) |> gpu
    test_image = rand(Float32, 32, 32, 3, 1) |> gpu
    params = Flux.params(layer)
    grads = gradient(params) do
       sum(layer(test_image))
    end
    @show [grads.grads[param] for param in params]
end
conv_answer_dict = Dict(
    # Known-good answers for 1d convolution operations
    1 => Dict(
        "y_pad"  => [1, 4,  7, 10, 13, 10.],
        "y_dil"  => [5, 8, 11.],
        "y_flip" => [5, 8, 11, 14.],

        "dx"        => [ 8, 18, 27, 36, 13.],
        "dx_stride" => [ 8,  4, 20, 10,  0.],
        "dx_pad"    => [ 9, 18, 27, 36, 33.],
        "dx_dil"    => [10, 16, 27,  8, 11.],
        "dx_flip"   => [ 5, 18, 27, 36, 28.],

        "dw"        => [134, 100.],
        "dw_stride" => [ 48,  34.],
        "dw_pad"    => [135, 150.],
        "dw_dil"    => [102,  54.],
        "dw_flip"   => [110, 148.],
    ),

    # Known-good answers for 2d convolution operations
    2 => Dict(
        "y_pad" => [
            1  9  29  49  48;
            4 29  79 129 115;
            7 39  89 139 122;
            10 49  99 149 129;
            13 59 109 159 136;
            10 40  70 100  80.
        ],
        "y_dil" => [
            48   98;
            58  108;
            68  118.
        ],
        "y_flip" => [
            51  101  151;
            61  111  161;
            71  121  171;
            81  131  181.
        ],

        "dx" => [
            116  374   674  258;
            243  700  1200  407;
            313  800  1300  437;
            383  900  1400  467;
            177  386   586  159.
        ],
        "dx_stride" => [
            116  58  516  258;
            87  29  387  129;
            196  98  596  298;
            147  49  447  149;
            0   0    0    0.
        ],
        "dx_pad" => [
            152  470   850   911;
            261  700  1200  1240;
            340  800  1300  1319;
            419  900  1400  1398;
            370  746  1126  1087.
        ],
        "dx_dil" => [
            192  392   96  196;
            232  432  116  216;
            416  766  184  334;
            174  324   58  108;
            204  354   68  118.
        ],
        "dx_flip" => [
            51  254   454   453;
            163  700  1200  1087;
            193  800  1300  1157;
            223  900  1400  1227;
            162  586   886   724.
        ],

        "dw" => [
            17378  11738;
            16250  10610.
        ],
        "dw_stride" => [
            5668  3888;
            5312  3532.
        ],
        "dw_pad" => [
            18670  22550;
            19850  23430.
        ],
        "dw_dil" => [
            8632  3652;
            7636  2656.
        ],
        "dw_flip" => [
            12590  19550;
            13982  20942.
        ],
    ),

    # Known-good answers for 3d convolution operations (these are getting rather large)
    3 => Dict(
        "y_pad"  => reshape([
            1, 4, 7, 10, 13, 10, 9, 29, 39, 49, 59, 40, 29, 79, 89, 99, 109, 70, 49, 129,
            139, 149, 159, 100, 48, 115, 122, 129, 136, 80, 26, 80, 94, 108, 122, 80, 126,
            322, 358, 394, 430, 260, 206, 502, 538, 574, 610, 360, 286, 682, 718, 754, 790,
            460, 220, 502, 524, 546, 568, 320, 146, 360, 374, 388, 402, 240, 446, 1042, 1078,
            1114, 1150, 660, 526, 1222, 1258, 1294, 1330, 760, 606, 1402, 1438, 1474, 1510,
            860, 420, 942, 964, 986, 1008, 560, 205, 456, 467, 478, 489, 270, 517, 1133, 1159,
            1185, 1211, 660, 577, 1263, 1289, 1315, 1341, 730, 637, 1393, 1419, 1445, 1471,
            800, 392, 847, 862, 877, 892, 480.
        ], (6,5,4)),
        "y_dil"  => reshape([608, 644, 680, 788, 824, 860.], (3,2,1)),
        "y_flip" => reshape([
            686, 722, 758, 794, 866, 902, 938, 974, 1046, 1082, 1118, 1154, 1406, 1442,
            1478, 1514, 1586, 1622, 1658, 1694, 1766, 1802, 1838, 1874.
        ], (4,3,2)),

        "dx"        => reshape([
            2576, 5118, 5658, 6198, 3010, 5948, 11576, 12512, 13448, 6420, 8468, 16256,
            17192, 18128, 8580, 4092, 7718, 8114, 8510, 3950, 9624, 18316, 19108, 19900,
            9340, 18680, 34992, 36288, 37584, 17320, 22280, 41472, 42768, 44064, 20200,
            9776, 17756, 18260, 18764, 8340, 4168, 7438, 7690, 7942, 3450, 6972, 11896,
            12256, 12616, 5140, 8052, 13696, 14056, 14416, 5860, 2804, 4278, 4386, 4494,
            1510.
        ], (5,4,3)),
        "dx_stride" => reshape([
            2576, 2254, 3152, 2758, 0, 1932, 1610, 2364, 1970, 0, 5456, 4774, 6032,
            5278, 0, 4092, 3410, 4524, 3770, 0, 1288, 966, 1576, 1182, 0, 644, 322,
            788, 394, 0, 2728, 2046, 3016, 2262, 0, 1364, 682, 1508, 754, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.
        ], (5,4,3)),
        "dx_pad"    => reshape([
            4220, 6343, 7116, 7889, 6550, 8490, 12276, 13312, 14348, 11606, 12350,
            17456, 18492, 19528, 15546, 11989, 16664, 17469, 18274, 14333, 16200,
            22628, 23616, 24604, 19392, 25336, 34992, 36288, 37584, 29320, 30216,
            41472, 42768, 44064, 34200, 26236, 35664, 36652, 37640, 28940, 22816,
            30831, 31636, 32441, 24794, 32522, 43668, 44704, 45740, 34742, 36462,
            48848, 49884, 50920, 38602, 29501, 39264, 40037, 40810, 30733.
        ], (5,4,3)),
        "dx_dil"    => reshape([
            4864, 5152, 9696, 4508, 4760, 6304, 6592, 12396, 5768, 6020, 3648,
            3864, 7120, 3220, 3400, 4728, 4944, 9100, 4120, 4300, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2432, 2576, 4544, 1932, 2040,
            3152, 3296, 5804, 2472, 2580, 1216, 1288, 1968, 644, 680, 1576, 1648,
            2508, 824, 860.
        ], (5,4,3)),
        "dx_flip"   => reshape([
            686, 2094, 2202, 2310, 1588, 2924, 7544, 7904, 8264, 5124, 3644, 9344,
            9704, 10064, 6204, 3138, 7430, 7682, 7934, 4616, 4836, 11980, 12484,
            12988, 7792, 14936, 34992, 36288, 37584, 21640, 17816, 41472, 42768,
            44064, 25240, 12620, 28412, 29204, 29996, 16728, 7030, 15646, 16042,
            16438, 9084, 17772, 38968, 39904, 40840, 22276, 19932, 43648, 44584,
            45520, 24796, 12362, 26742, 27282, 27822, 14992.
        ], (5,4,3)),

        "dw"        => reshape([1.058184e6, 1.0362e6,    948264,    926280,
                                    618504,   596520,    508584,    486600], (2,2,2)),
        "dw_stride" => reshape([    74760,     72608,     64000,     61848,
                                    31720,     29568,     20960,     18808.], (2,2,2)),
        "dw_pad"    => reshape([1.26055e6, 1.30805e6, 1.40327e6, 1.44923e6,
                                1.73731e6, 1.77589e6, 1.83259e6, 1.86731e6], (2,2,2)),
        "dw_dil"    => reshape([   250320,    241512,    206280,    197472,
                                    74160,     65352,     30120,     21312.], (2,2,2)),
        "dw_flip"   => reshape([    639480,   670200,    793080,    823800,
                                    1.25388e6, 1.2846e6, 1.40748e6,  1.4382e6], (2,2,2)),
    ),
)

@testset "Depthwise Convolution" begin
    # Start with some easy-to-debug cases that we have worked through and _know_ work
    for rank in (2,3)
        @testset "depthwiseconv$(rank)d" begin
            # Pull out known-good answers for y = depthwiseconv(x, w)
            y_pad = conv_answer_dict[rank]["y_pad"]  |> gpu
            y_dil = conv_answer_dict[rank]["y_dil"]  |> gpu
            y_flip = conv_answer_dict[rank]["y_flip"]  |> gpu

            # We can always derive y_plain and y_stride from the other answers.
            y_plain = y_pad[((2:(size(y_pad,idx)-1)) for idx in 1:rank)...]  |> gpu
            y_stride = y_pad[((2:2:(size(y_pad,idx)-1)) for idx in 1:rank)...]  |> gpu

            # Same for dx and dw:
            dx = conv_answer_dict[rank]["dx"]  |> gpu
            dx_stride = conv_answer_dict[rank]["dx_stride"]  |> gpu
            dx_pad = conv_answer_dict[rank]["dx_pad"]  |> gpu
            dx_dil = conv_answer_dict[rank]["dx_dil"]  |> gpu
            dx_flip = conv_answer_dict[rank]["dx_flip"]  |> gpu

            dw = conv_answer_dict[rank]["dw"] |> gpu
            dw_stride = conv_answer_dict[rank]["dw_stride"]  |> gpu
            dw_pad = conv_answer_dict[rank]["dw_pad"]  |> gpu
            dw_dil = conv_answer_dict[rank]["dw_dil"]  |> gpu
            dw_flip = conv_answer_dict[rank]["dw_flip"]  |> gpu

            # We generate x and w from the shapes we know they must be
            x = reshape(Float64[1:prod(size(dx));], size(dx)..., 1, 1)  |> gpu
            w = reshape(Float64[1:prod(size(dw));], size(dw)..., 1, 1)  |> gpu

            # A "drop channels and batch dimension" helper
            ddims(x) = dropdims(x, dims=(rank+1, rank+2))

            for conv in (depthwiseconv,)
                @testset "$(conv)" begin
                    # First, your basic convolution with no parameters
                    cdims = DepthwiseConvDims(x, w)
                    @test ddims(conv(x, w, cdims)) == y_plain

                    # Next, test convolution on views and alternate datatypes:
                    @test isapprox(ddims(conv(Float32.(x), Float32.(w), cdims)), Float32.(y_plain), rtol = 1.0e-7)

                    # Next, introduce stride:
                    cdims = DepthwiseConvDims(x, w; stride=2)
                    @test isapprox(ddims(conv(x, w, cdims)), y_stride, rtol = 1.0e-7)

                    # Next, introduce dilation:
                    cdims = DepthwiseConvDims(x, w; dilation=2)
                    @test isapprox(ddims(conv(x, w, cdims)), y_dil, rtol = 1.0e-7)

                    # Next, introduce padding:
                    cdims = DepthwiseConvDims(x, w; padding=1)
                    @test isapprox(ddims(conv(x, w, cdims)), y_pad, rtol = 1.0e-7)
                end
            end

            # Test all implementations/interfaces
            for (∇conv_filter, ∇conv_data) in (
                    (∇depthwiseconv_filter,        ∇depthwiseconv_data),
                )
                @testset "$(∇conv_filter)/$(∇conv_data)" begin
                    # First, your basic convolution with no parameters
                    cdims = DepthwiseConvDims(x, w)
                    dy = depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx

                    # Next, test convolution on views and alternate datatypes:
                    @test ddims(∇conv_filter(x, view(dy, repeat([:], ndims(dy))...), cdims)) == dw
                    @test ddims(∇conv_data(view(dy, repeat([:], ndims(dy))...), w,   cdims)) == dx

                    @test ddims(∇conv_filter(Float32.(x), Float32.(dy), cdims)) == dw
                    @test ddims(∇conv_data(Float32.(dy),  Float32.(w),  cdims)) == dx

                    # Next, introduce stride:
                    cdims = DepthwiseConvDims(x, w; stride=2)
                    dy = depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_stride
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_stride

                    # Next, introduce dilation:
                    cdims = DepthwiseConvDims(x, w; dilation=2)
                    dy = depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_dil
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_dil

                    # Next, introduce padding:
                    cdims = DepthwiseConvDims(x, w; padding=1)
                    dy = depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_pad
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_pad
                end
            end
        end
    end
end

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
