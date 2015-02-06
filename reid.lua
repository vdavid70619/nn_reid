--
-- Xiyang Dai
-- 2015.1
--
--
require("mobdebug").start()

require 'torch'

torch.setdefaulttensortype('torch.DoubleTensor')

require 'paths'
require 'nn'
require 'nnx'
require 'image'
require 'optim'
require 'dataloader' 
   
torch.setnumthreads(8)


-----------------------------------------------------------------------------------------
-- Build two same parallel CNN
-- {1,8,5,5,46,46}
-----------------------------------------------------------------------------------------
local ch = 3
   

network = nn.Sequential()
local filters = nn.ParallelTable()
network:add(filters)

-- first one
local filter1 = nn.Sequential()
filters:add(filter1)
filter1:add(nn.SpatialContrastiveNormalization(ch, image.gaussian1D(7)))
--filter1:add(nn.SpatialPadding(-datap.lWin, -datap.tWin, -datap.rWin, -datap.bWin))
local filters_elem = {}

table.insert(filters_elem, nn.SpatialConvolution(ch,8,5,5))
table.insert(filters_elem, nn.ReLU())
table.insert(filters_elem, nn.SpatialSubSampling(8,2,2,2,2))
table.insert(filters_elem, nn.SpatialConvolution(8,16,7,7))
table.insert(filters_elem, nn.ReLU())

-- second one (clone first one)
local filter2 = nn.Sequential()
filters:add(filter2)
filter2:add(filter1.modules[1]:clone())

for i = 1,#filters_elem do
   filter1:add(filters_elem[i])
   filter2:add(filters_elem[i]:clone('weight', 'bias', 'gradWeight', 'gradBias'))
end

network:add(nn.SpatialMatching(1,1))
network:add(nn.Reshape(17,17,1))
network:add(nn.Min(3))
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

local dl = Dataloader()
dl:load_from_folder('../../MATLAB/DATA/i-LIDS-VID/sequences/mix', '.png', 30)
--dl:rgb2grey()
dl:resize(50,50)
dl:shuffle()
trains, tests = dl:train_test_split(0.2, 'inter')
ptrains = dl.generate_pairs(trains)
ptests = dl.generate_pairs(tests)

-- this matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix{'Match','Dismatch'}

-- log results to files
local dname,fname = sys.fpath()
local trainLogger = optim.Logger(paths.concat(dname, 'train.log'))
local testLogger = optim.Logger(paths.concat(dname, 'test.log'))

date = os.date('%y%m%d_%H%M')
-----------------------------------------------------------------------------------------
-- Train!!!
-----------------------------------------------------------------------------------------
local learningp = {
   nEpochs = 1000,
   rate = 5e-4,
   weightDecay = 0.01,
   rateDecay = 1e-3,
   nPecentage = 100
}

local iEpoch=1


parameters, gradParameters = network:getParameters()
--parameters:copy(torch.load('./face_verf/face_vertify_121208_1830.param.old'))

function train(dataset)

   local time = sys.clock()
   local tic = sys.clock()
   local nper = learningp.nPecentage

   local  criterion = nn.ClassNLLCriterion()
   --local criterion = nn.MSECriterion()
   
   local config = {learningRate = learningp.rate,
      weightDecay = learningp.weightDecay,
      momentum = 0,
      learningRateDecay = learningp.rateDecay}
   
   print("Epoch " .. iEpoch .. " of " .. learningp.nEpochs)

   win = image.display{image=network:getWeights().layer1, padding=2, zoom=4, win=win}


   for t = 1,dataset.size do
   	  if (nper*t)%dataset.size==0 then
   	  	toc = sys.clock()
   	  	print(nper*t/dataset.size .. '/' .. nper .. ' - ' .. string.format("%.4f",toc-tic) .. 's') 
   	  	tic = sys.clock()   	  	
      	collectgarbage()        	  	
   	  end
      --xlua.progress(t, dataset.size)

      local input = {dataset[t][1], dataset[t][2]}
      local target = dataset[t][3]
      
      -- stupid criterion def
      if target==0 then target=2
      else target=1
      end 

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
            -- get new parameters
            if x ~= parameters then
               parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0
            
            -- estimate f
            local output = network:forward(input)
            local err = criterion:forward(output, target)
            f = f + err
            --print('|'..output:squeeze()..'-'..targetCrit..'|^2 = '..f)
            
            -- update confusion
            confusion:add(output, target)

            -- estimate df/dW
            local df_do = criterion:backward(output, target)
            network:backward(input, df_do)
                         
            return f, gradParameters
      end

      optim.sgd(feval, parameters, config)
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset.size
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = fname:gsub('.lua','') .. '/face_vertify_'..date..'.param'
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving parameters to '..filename)
   torch.save(filename, parameters)

   iEpoch = iEpoch + 1

end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()
   local tic = sys.clock()
   local nper = learningp.nPecentage

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset.size do
      -- disp progress
   	  if (nper*t)%dataset.size==0 then
   	  	toc = sys.clock()
   	  	print(nper*t/dataset.size .. '/' .. nper .. ' - ' .. string.format("%.4f",toc-tic) .. 's') 
   	  	tic = sys.clock()   	  	
      	collectgarbage()        	  	
   	  end  
      --xlua.progress(t, dataset.size)

      -- get new sample
      local sample = dataset[t]
      local input = {sample[1], sample[2]}
      local target = sample[3]

      -- stupid criterion def
      if target==0 then target=2
      else target=1
      end 

      -- test sample
      local result = network:forward(input)
      confusion:add(result, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset.size
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


while iEpoch<learningp.nEpochs do
   -- train/test
   train(ptrains)
   test(ptests)

   -- plot errors
   --trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   --trainLogger:plot()
   --testLogger:plot()
end