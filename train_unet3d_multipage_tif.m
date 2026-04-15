% 使用多页 TIFF 数据训练 3D U-Net（trainnet）
% 数据目录要求：
% C:\Users\chezhu\Desktop\data\image\image1.tif,image2.tif,image3.tif
% C:\Users\chezhu\Desktop\data\label\label1.tif,label2.tif,label3.tif

clear;
clc;

dataDir = "C:\Users\chezhu\Desktop\data";
imageDir = fullfile(dataDir,"image");
labelDir = fullfile(dataDir,"label");

imageFiles = fullfile(imageDir,["image1.tif","image2.tif","image3.tif"]);
labelFiles = fullfile(labelDir,["label1.tif","label2.tif","label3.tif"]);

for i = 1:numel(imageFiles)
    if ~isfile(imageFiles(i))
        error("找不到图像文件: %s", imageFiles(i));
    end
    if ~isfile(labelFiles(i))
        error("找不到标签文件: %s", labelFiles(i));
    end
end

numSamples = numel(imageFiles);
XTrain = cell(numSamples,1);
rawLabels = cell(numSamples,1);
classIDs = [];

for i = 1:numSamples
    imgVol = readMultipageTiffVolume(imageFiles(i));
    lblVol = readMultipageTiffVolume(labelFiles(i));

    if ~isequal(size(imgVol),size(lblVol))
        error("图像与标签尺寸不一致: %s 与 %s", imageFiles(i), labelFiles(i));
    end

    imgVol = single(imgVol);
    imgVol = normalizeVolume(imgVol);
    XTrain{i} = reshape(imgVol,[size(imgVol,1),size(imgVol,2),size(imgVol,3),1]);

    lblVol = uint16(lblVol);
    rawLabels{i} = lblVol;
    classIDs = union(classIDs,unique(lblVol(:))');
end

classIDs = sort(classIDs);
classNames = "class" + string(classIDs);
backgroundIdx = find(classIDs == 0,1);
if ~isempty(backgroundIdx)
    classNames(backgroundIdx) = "background";
end

TTrain = cell(numSamples,1);
for i = 1:numSamples
    TTrain{i} = labelToCategorical(rawLabels{i},classIDs,classNames);
end

inputSize = size(XTrain{1});
inputSize = inputSize(1:4);
numClasses = numel(classNames);
encoderDepth = chooseEncoderDepth(inputSize(1:3));

net = unet3d(inputSize,numClasses,EncoderDepth=encoderDepth);
if ~isa(net,"dlnetwork")
    net = dlnetwork(net);
end

dsX = arrayDatastore(XTrain,IterationDimension=1);
dsT = arrayDatastore(TTrain,IterationDimension=1);
dsTrain = combine(dsX,dsT);

options = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=100, ...
    MiniBatchSize=1, ...
    Shuffle="every-epoch", ...
    Verbose=true, ...
    Plots="training-progress");

trainedNet = trainnet(dsTrain,net,"crossentropy",options);

save("trainedUnet3d.mat","trainedNet","classNames","classIDs");
disp("训练完成，模型已保存为 trainedUnet3d.mat");

function vol = readMultipageTiffVolume(filePath)
info = imfinfo(filePath);
numSlices = numel(info);
firstSlice = imread(filePath,1);

vol = zeros(size(firstSlice,1),size(firstSlice,2),numSlices,"like",firstSlice);
vol(:,:,1) = firstSlice;
for k = 2:numSlices
    vol(:,:,k) = imread(filePath,k);
end
end

function out = normalizeVolume(vol)
vmin = min(vol(:));
vmax = max(vol(:));
if vmax > vmin
    out = (vol - vmin) ./ (vmax - vmin);
else
    out = zeros(size(vol),"like",vol);
end
end

function catLabel = labelToCategorical(lblVol,classIDs,classNames)
indexVol = zeros(size(lblVol),"uint16");
for c = 1:numel(classIDs)
    indexVol(lblVol == classIDs(c)) = c;
end
catLabel = categorical(indexVol,1:numel(classNames),cellstr(classNames));
end

function depth = chooseEncoderDepth(spatialSize)
maxAllowed = floor(log2(double(min(spatialSize)))) - 1;
depth = max(2,min(4,maxAllowed));
end
