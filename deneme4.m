%% %81,88
%% alexnet mimarsinini oluþturma
net = alexnet;

%% Verilerin oldugu bölüm
matlabpath = "C:\Users\koray\Documents\MATLAB"
data = fullfile(matlabpath,'DataSet')
train = imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');

%% egitimde kullanýlacak veri miktarýnýn belirler
[imdsTrain,imdsValidation] = splitEachLabel(train,0.7); %% %70 eðitim için %30 test için

%% Load network alexnet mimarisi katmanlarý
layers = [
    imageInputLayer([227 227 3],"Name","data")
    
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
    batchNormalizationLayer("Name","batchnorm1")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    
    groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","relu2")
    batchNormalizationLayer("Name","batchnorm2")
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    
    convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu3")
    
    groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu4")
    
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu5")
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    batchNormalizationLayer("Name","batchnorm6")
    
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2)
    reluLayer("Name","relu7")
    dropoutLayer(0.1,"Name","drop7")
    
    fullyConnectedLayer(2,"Name","fc8","BiasLearnRateFactor",2)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];

%% training
options = trainingOptions('sgdm', ...
    'MiniBatchSize',3, ...
    'MaxEpochs',15, ...
    'ExecutionEnvironment', ...
    'auto', ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

[net,tr] = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YTest = imdsValidation.Labels;
accuracy = mean(YPred == YTest)

confusionchart(YTest,YPred);
lay='fc';