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

function get_test_data(get_proportion = 1.0, batchsize = 0)
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)
    if get_proportion < 1.0
        test = test[shuffle(1:length(test))[1:Int64(length(test)*get_proportion)]]
    end
    testimgs = [getarray(test[i].img) for i in 1:length(test)]
	testY = Matrix(onehotbatch([test[i].ground_truth.class for i in 1:length(test)], 1:10))
	if batchsize  == 0
	    testX = cat(testimgs..., dims = 4)
	    test = (testX,testY)
	else
		test = [(cat(testimgs[i]..., dims = 4), testY[:,i]) for i in partition(1:length(test), batchsize)]
	end
    return test
end


function get_processed_data(splitr = 0.5, batchsize = 64, mini = 1.0, val_batchsize = 0)
    # Fetching the train and validation data and getting them into proper shape
    total_img = Int(floor(40000*mini))
    X = shuffle(trainimgs(CIFAR10))
    imgs = [getarray(X[i].img) for i in 1:total_img]
    labels = Matrix(onehotbatch([X[i].ground_truth.class for i in 1:total_img],1:10))
    train_pop = Int(floor((1-splitr)* total_img))
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)]
    if train_pop < total_img
        if val_batchsize == 0
	    	val_batchsize = batchsize
        end
        val = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(train_pop+1:total_img, val_batchsize)]
    else
        val = []
    end
    return train, val
end

export get_processed_data, get_test_data
