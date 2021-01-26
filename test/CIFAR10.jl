export get_processed_data, get_test_data

using Images, Flux, MLDatasets
using Flux: onehotbatch
using Zygote: @nograd
using Base.Iterators: partition
using Random
using Statistics

function get_test_data(get_proportion = 1.0, batchsize = 0)
    test_x, test_y = MLDatasets.CIFAR10.testdata(Float32)
	total_img = Int(floor(length(test_y)*get_proportion))
	order = 1:total_img
	X = test_x[:,:,:,order]
	y = test_y[order]
	imgs = [X[:,:,:,i] for i in 1:total_img]
	labels = Matrix(onehotbatch([y[i] for i in 1:total_img],0:9))
	if batchsize == 0
	    test = (imgs,[labels[:,i] for i in 1:total_img])
	else
		test = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:total_img, batchsize)]
	end
    return test
end


function get_processed_data(splitr = 0.5, batchsize = 64, mini = 1.0, val_batchsize = 0)
	train_x, train_y = MLDatasets.CIFAR10.traindata(Float32)
    total_img = Int(floor(length(train_y)*mini))
	order = shuffle(1:total_img)
    X = train_x[:,:,:,order]
	y = train_y[order]
	imgs = [X[:,:,:,i] for i in 1:total_img]
    labels = Matrix(onehotbatch([y[i] for i in 1:total_img],0:9))
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
