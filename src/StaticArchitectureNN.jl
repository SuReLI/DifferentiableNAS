using Metalhead, Images
using Metalhead: trainimgs
using Images.ImageCore
using Flux
using Flux.Optimise
using Base.Iterators: partition
using StatsBase:mean
# using CUDA

function run_SANN()

  # The image will give us an idea of what we are dealing with.
  # ![title](https://pytorch.org/tutorials/_images/cifar10.png)

  Metalhead.download(CIFAR10)
  X = trainimgs(CIFAR10)
  labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)

  # Let's take a look at a random image from the dataset

  image(x) = x.img # handy for use later
  ground_truth(x) = x.ground_truth
  image.(X[rand(1:end, 10)])

  # The images are simply 32 X 32 matrices of numbers in 3 channels (R,G,B). We can now
  # arrange them in batches of say, 1000 and keep a validation set to track our progress.
  # This process is called minibatch learning, which is a popular method of training
  # large neural networks. Rather that sending the entire dataset at once, we break it
  # down into smaller chunks (called minibatches) that are typically chosen at random,
  # and train only on them. It is shown to help with escaping
  # [saddle points](https://en.wikipedia.org/wiki/Saddle_point).

  # Defining a `getarray` function would help in converting the matrices to `Float` type.

  getarray(X) = float.(permutedims(channelview(X), (2, 3, 1)))
  imgs = [getarray(X[i].img) for i in 1:50000]

  # The first 49k images (in batches of 1000) will be our training set, and the rest is
  # for validation. `partition` handily breaks down the set we give it in consecutive parts
  # (1000 in this case). `cat` is a shorthand for concatenating multi-dimensional arrays along
  # any dimension.

  train = ([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 1000)]) # |> gpu
  valset = 49001:50000
  valX = cat(imgs[valset]..., dims = 4) #|> gpu
  valY = labels[:, valset] #|> gpu

  # ## Defining the Classifier
  # --------------------------
  # Now we can define our Convolutional Neural Network (CNN).

  # A convolutional neural network is one which defines a kernel and slides it across a matrix
  # to create an intermediate representation to extract features from. It creates higher order
  # features as it goes into deeper layers, making it suitable for images, where the strucure of
  # the subject is what will help us determine which class it belongs to.

  m = Chain(
    Conv((5,5), 3=>16, relu),
    MaxPool((2,2)),
    Conv((5,5), 16=>8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(200, 120),
    Dense(120, 84),
    Dense(84, 10),
    softmax) #|> gpu

  #-
  # We will use a crossentropy loss and an Momentum optimiser here. Crossentropy will be a
  # good option when it comes to working with mulitple independent classes. Momentum gradually
  # lowers the learning rate as we proceed with the training. It helps maintain a bit of
  # adaptivity in our optimisation, preventing us from over shooting from our desired destination.
  #-


  loss(x, y) = sum(crossentropy(m(x), y))
  opt = Momentum(0.01)

  # We can start writing our train loop where we will keep track of some basic accuracy
  # numbers about our model. We can define an `accuracy` function for it like so.

  accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))

  # ## Training
  # -----------

  # Training is where we do a bunch of the interesting operations we defined earlier,
  # and see what our net is capable of. We will loop over the dataset 10 times and
  # feed the inputs to the neural network and optimise.

  epochs = 10

  for epoch = 1:epochs
    for d in train
      gs = gradient(params(m)) do
        l = loss(d...)
      end
      update!(opt, params(m), gs)
    end
    @show accuracy(valX, valY)
  end

  # Seeing our training routine unfold gives us an idea of how the network learnt the
  # This is not bad for a small hand-written network, trained for a limited time.

  # Training on a GPU
  # -----------------

  # The `gpu` functions you see sprinkled through this bit of the code tell Flux to move
  # these entities to an available GPU, and subsequently train on it. No extra faffing
  # about required! The same bit of code would work on any hardware with some small
  # annotations like you saw here.

  # ## Testing the Network
  # ----------------------

  # We have trained the network for 100 passes over the training dataset. But we need to
  # check if the network has learnt anything at all.

  # We will check this by predicting the class label that the neural network outputs, and
  # checking it against the ground-truth. If the prediction is correct, we add the sample
  # to the list of correct predictions. This will be done on a yet unseen section of data.

  # Okay, first step. Let us perform the exact same preprocessing on this set, as we did
  # on our training set.

  valset = valimgs(CIFAR10)
  valimg = [getarray(valset[i].img) for i in 1:10000]
  labels = onehotbatch([valset[i].ground_truth.class for i in 1:10000],1:10)
  #test = gpu.([(cat(valimg[i]..., dims = 4), labels[:,i]) for i in partition(1:10000, 1000)])
  test = ([(cat(valimg[i]..., dims = 4), labels[:,i]) for i in partition(1:10000, 1000)])

  # Next, display some of the images from the test set.


  # The outputs are energies for the 10 classes. Higher the energy for a class, the more the
  # network thinks that the image is of the particular class. Every column corresponds to the
  # output of one image, with the 10 floats in the column being the energies.

  # Let's see how the model fared.

  rand_test = getarray.(image.(valset[ids]))
  rand_test = cat(rand_test..., dims = 4)# |> gpu
  rand_truth = ground_truth.(valset[ids])
  m(rand_test)

  # This looks similar to how we would expect the results to be. At this point, it's a good
  # idea to see how our net actually performs on new data, that we have prepared.

  accuracy(test[1]...)

  # This is much better than random chance set at 10% (since we only have 10 classes), and
  # not bad at all for a small hand written network like ours.

  # Let's take a look at how the net performed on all the classes performed individually.

  class_correct = zeros(10)
  class_total = zeros(10)
  for i in 1:10
    preds = m(test[i][1])
    lab = test[i][2]
    for j = 1:1000
      pred_class = findmax(preds[:, j])[2]
      actual_class = findmax(lab[:, j])[2]
      if pred_class == actual_class
        class_correct[pred_class] += 1
      end
      class_total[actual_class] += 1
    end
  end

  @show class_correct ./ class_total
end
