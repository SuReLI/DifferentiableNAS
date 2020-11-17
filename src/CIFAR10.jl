export get_CIFAR10_train, get_CIFAR10_test

using Metalhead, Images, Flux
using Flux: onehotbatch
using Base.Iterators: partition

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
