--
-- Xiyang Dai
-- 2014.05
-- Test
--


--torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'nnx'
require 'image'
require 'optim'
require 'lfw_pairloader' 



-----------------------------------------------------------------------------------------
-- Load Data
-----------------------------------------------------------------------------------------
local testData = LFWDataset{logfile='/Users/xiyangdai/Desktop/pairs.txt',
                           dataSetFolder='/Users/xiyangdai/Downloads/lfw/',
                           sampleSize={1,50,50},
                           channels=1}

--testData:shuffle()
testData:display(100,'test')



-----------------------------------------------------------------------------------------
-- Test!!!
-----------------------------------------------------------------------------------------
network = torch.load('face_verf/face_vertify.net')



-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local sample = dataset[t]
      local input = sample[1]
      local target = sample[2]

      -- test sample
      confusion:add(network:forward(input), target)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end

local iEpoch = 1

while iEpoch<=10 do
   -- train/test
   local matchData1, dismatchData1 = testData:popSubset{overall=false,nElement=300}

   -- testing set
   local tests = nn.DataList()
   tests:appendDataSet(matchData1,'Match')
   tests:appendDataSet(dismatchData1,'Dismatch')
   tests:shuffle()


   confusion = optim.ConfusionMatrix{'Match','Dismatch'}
   -- log results to files
   local dname,fname = sys.fpath()
   testLogger = optim.Logger(paths.concat(dname, 'test.log'))

   -- test
   test(tests)

   -- plot errors
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   
   iEpoch = iEpoch + 1
end