--
-- Xiyang Dai
-- 2012.11
--
--


torch.setdefaulttensortype('torch.DoubleTensor')

require 'nn'
require 'nnx'
require 'image'
require 'optim'
require 'lfw_pairloader' 
   
torch.setnumthreads(8)


-----------------------------------------------------------------------------------------
-- Build two same parallel CNN
-- {1,8,5,5,46,46}
-----------------------------------------------------------------------------------------
local ch = 1   
   

network = nn.Sequential()
local filters = nn.ParallelTable()
network:add(filters)

-- first one
local filter1 = nn.Sequential()
filters:add(filter1)
filter1:add(nn.SpatialContrastiveNormalization(ch, image.gaussian1D(7)))
--filter1:add(nn.SpatialPadding(-datap.lWin, -datap.tWin, -datap.rWin, -datap.bWin))
local filters_elem = {}

table.insert(filters_elem, nn.SpatialConvolution(ch,4,5,5))
table.insert(filters_elem, nn.Tanh())
table.insert(filters_elem, nn.SpatialSubSampling(4,2,2,2,2))
table.insert(filters_elem, nn.SpatialConvolution(4,16,7,7))
table.insert(filters_elem, nn.Tanh())

-- second one (clone first one)
local filter2 = nn.Sequential()
filters:add(filter2)
filter2:add(filter1.modules[1]:clone())

for i = 1,#filters_elem do
   filter1:add(filters_elem[i])
   filter2:add(filters_elem[i]:clone('weight', 'bias', 'gradWeight', 'gradBias'))
end

network:add(nn.SpatialMatching(5,5))
network:add(nn.Reshape(17,17,5*5))
network:add(nn.Min(3))
--network:add(nn.Reshape(1,23,23))
--network:add(nn.SpatialSubSampling(1,5,5,5,5))
--network:add(nn.SpatialLPPooling(1,2,18,18))
network:add(nn.Reshape(17*17))
network:add(nn.Linear(17*17,2))
network:add(nn.LogSoftMax())

function network:getWeights()
   local weights = {}
   weights.layer1 = self.modules[1].modules[1].modules[2].weight
   return weights
end

-----------------------------------------------------------------------------------------
-- Load Data
-----------------------------------------------------------------------------------------

local trainData = LFWDataset{logfile='/Users/xiyangdai/Desktop/peopleDevTrain.txt',
                            dataSetFolder='/Users/xiyangdai/Downloads/lfw/',
                            sampleSize={ch,50,50},
                            channels=ch,
                            unrestricted=true,
                            nbSamplesRequired=10000}

trainData.match:shuffle()
trainData.dismatch:shuffle()
trainData:displayBoth(24)

-- local testData = LFWDataset{logfile='/Users/xiyangdai/Desktop/pairs.txt',
--                            dataSetFolder='/Users/xiyangdai/Downloads/lfw/',
--                            sampleSize={ch,50,50},
--                            channels=ch}

-- trainData = testData

matchData1, dismatchData1 = trainData:popSubset{overall=false,ratio=0.2}
--matchData2, dismatchData2 = testData:popSubset{overall=false,ratio=0.2}





local trains = nn.DataList()
trains:appendDataSet(trainData:MatchSet(),'Match')
trains:appendDataSet(trainData:DismatchSet(),'Dismatch')
--trains:appendDataSet(testData:MatchSet(),'Match')
--trains:appendDataSet(testData:DismatchSet(),'Dismatch')
trains:shuffle()

-- testing set
local tests = nn.DataList()
tests:appendDataSet(matchData1,'Match')
tests:appendDataSet(dismatchData1,'Dismatch')
--tests:appendDataSet(matchData2,'Match')
--tests:appendDataSet(dismatchData2,'Dismatch')
tests:shuffle()

-- local trains = nn.DataList()
-- trains:appendDataSet(trainData:MatchSet(),'Match')
-- trains:appendDataSet(trainData:DismatchSet(),'Dismatch')
-- trains:shuffle()

-- -- testing set
-- local tests = nn.DataList()
-- tests:appendDataSet(testData:MatchSet(),'Match')
-- tests:appendDataSet(testData:DismatchSet(),'Dismatch')
-- tests:shuffle()

--local trains = trainData
--local tests = testData
-- this matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix{'Match','Dismatch'}

-- log results to files
local dname,fname = sys.fpath()

local labelLogger = optim.Logger(paths.concat(dname, 'label.log'))
local scoreLogger = optim.Logger(paths.concat(dname, 'score.log'))

date = os.date('%y%m%d_%H%M')
-----------------------------------------------------------------------------------------
-- Train!!!
-----------------------------------------------------------------------------------------
local learningp = {
   nEpochs = 1000,
   rate = 5e-4,
   --weightDecay = 0.01,
   rateDecay = 1e-3,
}

local iEpoch=1


parameters, gradParameters = network:getParameters()
parameters:copy(torch.load('./face_verf/face_vertify_121209_0047.param'))



-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()
   
   local roc_data = {}


   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      --xlua.progress(t, dataset:size())

      -- get new sample
      local sample = dataset[t]
      local input = sample[1]
      local target = sample[2]

      if target[1]==-1 then target=2
      else target=1
      end  

      -- test sample
      local criterion = nn.ClassNLLCriterion()
      local result = network:forward(input)
      local err = criterion:forward(result, target)
      local _, index = torch.max(result,1)
      err = result[1]/result[2]
      print(err..'|'..index[1])
      table.insert(roc_data,{score = err, label = target})
      scoreLogger:add{['% Scores (test set)'] = err}
      labelLogger:add{['% Labels (test set)'] = target}  
      confusion:add(result, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   confusion:updateValids()
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end


test(tests)