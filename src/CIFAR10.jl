#source: https://github.com/FluxML/model-zoo/blob/master/vision/cifar10/cifar10.jl
export get_processed_data, get_test_data
using Metalhead, Images, Flux
using Flux: onehotbatch
using Base.Iterators: partition

# Function to convert the RGB image to Float64 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_processed_data(args)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:40000]
    #onehot encode labels of batch

    labels = onehotbatch([X[i].ground_truth.class for i in 1:40000],1:10)
    train_pop = Int((1-args.splitr_)* 40000)
    #train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, args.batchsize)])
    train = ([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, args.batchsize)])
    #valset = collect(train_pop+1:40000)
    #valX = cat(imgs[valset]..., dims = 4) # |> gpu
    #valY = labels[:, valset] #|> gpu
    #val = (valX,valY)
    val = ([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(train_pop+1:40000, args.batchsize)])
    return train, val
end

function get_test_data()
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:1000]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:1000], 1:10)# |> gpu
    testX = cat(testimgs..., dims = 4)# |> gpu

    test = (testX,testY)
    return test
end
