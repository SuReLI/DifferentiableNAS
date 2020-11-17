export get_CIFAR10_train, get_CIFAR10_test

using Metalhead, Images, Flux
using Flux: onehotbatch
using Base.Iterators: partition

# Function to convert the RGB image to Float64 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_processed_data(Args)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    println("test")
    imgs = [getarray(X[i].img) for i in 1:40000]
    #onehot encode labels of batch

    labels = onehotbatch([X[i].ground_truth.class for i in 1:40000],1:10)

    train_pop = Int((1-args.splitr_)* 40000)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, args.batchsize)])
    valset = collect(train_pop+1:40000)
    valX = cat(imgs[valset]..., dims = 4) # |> gpu
    valY = labels[:, valset] #|> gpu

    val = (valX,valY)
    return train, val
end

function get_test_data()
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:1000]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:1000], 1:10) |> gpu
    testX = cat(testimgs..., dims = 4) |> gpu

    test = (testX,testY)
    return test
end


getarray(X) = float.(permutedims(channelview(X), (2, 3, 1)))

function get_CIFAR10_train(batch_size::Int64, train_portion = 0.5)
  Metalhead.download(CIFAR10)
  X = trainimgs(CIFAR10)
  dataset_size = length(X)
  labels = onehotbatch([X[i].ground_truth.class for i in 1:dataset_size],1:10)
  imgs = [getarray(X[i].img) for i in 1:dataset_size]
  train_size = Int64(train_portion * length(X))
  train = ([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_size, batch_size)]) # |> gpu
  val_set = train_size+1:dataset_size
  valX = cat(imgs[val_set]..., dims = 4) #|> gpu
  valY = labels[:, val_set] #|> gpu
  train, valX, valY
end

function get_CIFAR10_test(batch_size::Int)
  Metalhead.download(CIFAR10)
  test_set = valimgs(CIFAR10)
  test_size = length(valset)
  test_img = [getarray(test_set[i].img) for i in 1:test_size]
  labels = onehotbatch([test_set[i].ground_truth.class for i in 1:test_size],1:10)
  #test = gpu.([(cat(valimg[i]..., dims = 4), labels[:,i]) for i in partition(1:10000, 1000)])
  test = ([(cat(test_img[i]..., dims = 4), labels[:,i]) for i in partition(1:test_size, batch_size)])
  test
end
