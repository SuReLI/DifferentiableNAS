#source: https://github.com/FluxML/model-zoo/blob/master/vision/cifar10/cifar10.jl
using Metalhead, Images, Flux
using Flux: onehotbatch
using Base.Iterators: partition

# Function to convert the RGB image to Float64 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_test_data()
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)
    testimgs = [getarray(test[i].img) for i in 1:1000]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:1000], 1:10)# |> gpu
    testX = cat(testimgs..., dims = 4)# |> gpu
    test = (testX,testY)
    return test
end


function get_processed_data(splitr = 0.5, batchsize = 64)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:40000]
    labels = onehotbatch([X[i].ground_truth.class for i in 1:40000],1:10)
    train_pop = Int((1-splitr)* 40000)
    #train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)])
    train = ([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)])
    val = ([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(train_pop+1:40000, batchsize)])
    return train, val
end

export get_processed_data, get_test_data
