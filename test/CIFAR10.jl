using Metalhead, Images, Flux
using Flux: onehotbatch
using Zygote: @nograd
using Base.Iterators: partition
using Random
using CUDA

@nograd onehotbatch
# Function to convert the RGB image to Float64 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_test_data(get_proportion = 1.0)
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)
    if get_proportion < 1.0
        test = test[shuffle(1:length(test))[1:Int64(length(test)*get_proportion)]]
    end
    testimgs = [getarray(test[i].img) for i in 1:length(test)]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:length(test)], 1:10)
    testX = cat(testimgs..., dims = 4)
    test = (testX,testY)
    return test
end


function get_processed_data(splitr = 0.5, batchsize = 64)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:40000]
    labels = onehotbatch([X[i].ground_truth.class for i in 1:40000],1:10)
    train_pop = Int((1-splitr)* 40000)
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)]
    if train_pop < 40000
        val = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(train_pop+1:40000, batchsize)]
    else
        val = []
    end
    return train, val
end

export get_processed_data, get_test_data
