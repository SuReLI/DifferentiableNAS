using Images, Flux, MLDatasets
using Flux: onehotbatch
using Zygote: @nograd
using Base.Iterators: partition
using Random
using Statistics

CIFAR_MEAN = [0.47095686, 0.46275508, 0.42994085]
CIFAR_STD = [0.24525039, 0.24154642, 0.25916523]

function get_test_data(get_proportion = 1.0, batchsize = 0)
    # Fetch the test data from Metalhead and get it into proper shape.
    test_x, test_y = MLDatasets.CIFAR10.testdata(Float32)
	mean_im = repeat(reshape(CIFAR_MEAN, (1,1,3)), outer = [32,32,1])
	std_im = repeat(reshape(CIFAR_STD, (1,1,3)), outer = [32,32,1])
	total_img = Int(floor(length(test_y)*get_proportion))
	order = shuffle(1:total_img)
    test = [(test_x[:,:,:,i], test_y[i]) for i in order]
    testimgs = [(test[i][1].-mean_im)./std_im for i in 1:length(test)]
	testY = Matrix(onehotbatch([test[i][2] for i in 1:length(test)], 0:9))
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
	train_x, train_y = MLDatasets.CIFAR10.traindata(Float32)
	mean_im = repeat(reshape(CIFAR_MEAN, (1,1,3)), outer = [32,32,1])
	std_im = repeat(reshape(CIFAR_STD, (1,1,3)), outer = [32,32,1])
    total_img = Int(floor(length(train_y)*mini))
	order = shuffle(1:total_img)
    X = train_x[:,:,:,order]
	y = train_y[order]
	@show (mean(X[:,:,:,1], dims = (1,2)), std(X[:,:,:,1], dims = (1,2)))
    imgs = [(X[:,:,:,i].-mean_im)./std_im for i in 1:total_img]
	@show (mean(imgs[1], dims = (1,2)), std(imgs[1], dims = (1,2)))
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

export get_processed_data, get_test_data
