export get_processed_data, get_test_data


using Images, Flux, MLDatasets
using Flux: onehotbatch
using Zygote: @nograd
using Base.Iterators: partition
using Random
using Statistics

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] |> f32
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] |> f32

function get_test_data(get_proportion = 1.0, batchsize = 0)
    # Fetch the test data from Metalhead and get it into proper shape.
    test_x, test_y = MLDatasets.CIFAR10.testdata(Float32)
	#pic = test_x[:,:,:,1]
	#imshow(pic)
	#PyPlot.savefig("test1.png")
	mean_im = repeat(reshape(CIFAR_MEAN, (1,1,3)), outer = [32,32,1])
	std_im = repeat(reshape(CIFAR_STD, (1,1,3)), outer = [32,32,1])
	total_img = Int(floor(length(test_y)*get_proportion))
	#order = shuffle(1:total_img)
	order = 1:total_img
    test_xy = [(test_x[:,:,:,i], test_y[i]) for i in order]
    #testimgs = [(test[i][1].-mean_im)./std_im for i in 1:length(test)]
	#pic = testimgs[1]
	#pic = pic .- minimum(pic)
	#pic = pic ./ maximum(pic)
	#imshow(pic)
	#PyPlot.savefig("test2.png")
	testY = Matrix(onehotbatch([test_xy[i][2] for i in 1:length(test_xy)], 0:9))
	if batchsize  == 0
	    testX = cat([t[1] for t in test_xy]..., dims = 4)
	    test = (testX,testY)
	else
		test = [(cat([t[1] for t in test_xy[i]]..., dims = 4), testY[:,i]) for i in partition(1:length(test_xy), batchsize)]
	end
    return test
end


function get_processed_data(splitr = 0.5, batchsize = 64, mini = 1.0, val_batchsize = 0)
    # Fetching the train and validation data and getting them into proper shape
	train_x, train_y = MLDatasets.CIFAR10.traindata(Float32)
	#pic = train_x[:,:,:,1]
	#imshow(pic)
	#PyPlot.savefig("tv1.png")
	mean_im = repeat(reshape(CIFAR_MEAN, (1,1,3)), outer = [32,32,1])
	std_im = repeat(reshape(CIFAR_STD, (1,1,3)), outer = [32,32,1])
    total_img = Int(floor(length(train_y)*mini))
	#order = shuffle(1:total_img)
	order = 1:total_img
    X = train_x[:,:,:,order]
	y = train_y[order]
	#@show (mean(X[:,:,:,1], dims = (1,2)), std(X[:,:,:,1], dims = (1,2)))
    #imgs = [(X[:,:,:,i].-mean_im)./std_im for i in 1:total_img]
	imgs = [X[:,:,:,i] for i in 1:total_img]
	#pic = imgs[1]
	#pic = pic .- minimum(pic)
	#pic = pic ./ maximum(pic)
	#imshow(pic)
	#PyPlot.savefig("tv2.png")
	#@show (mean(imgs[1], dims = (1,2)), std(imgs[1], dims = (1,2)))
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
