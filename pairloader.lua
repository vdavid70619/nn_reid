--
-- Xiyang Dai
-- 2014.05
--
--


require 'nnx'
require 'image'
require 'xlua'


local pairwise_dataset = torch.class('PairData')
--pairwise_dataset = {}
--pairwise_dataset.match=nn.DataSet()
--pairwise_dataset.dismatch=nn.DataSet()

function pairwise_dataset:__init(...)
   self.match=nn.DataSet()
   self.dismatch=nn.DataSet()
   self.nbSamples=0
   if select('#',...) > 0 then
       self:load(...)
   end
end

function pairwise_dataset:size()
   return self.nbSamples
end

function pairwise_dataset:MatchSet()
   return self.match
end

function pairwise_dataset:DismatchSet()
   return self.dismatch
end

function pairwise_dataset:__tostring__()
   str = 'DataSet:\n'
   if self.nbSamples then
      str = str .. ' + nb samples : '..self.nbSamples
   else
      str = str .. ' + empty set...'
   end
   return str
end

function pairwise_dataset:write(file)
   --file:writeBool(self.resized)
   file:writeInt(self.nbSamples)
   -- write all the samples
   for i = 1,self.nbSamples do
      file:writeObject(self[i])
   end
end

function pairwise_dataset:read(file)
   self.resized = file:readBool()
   self.nbSamples = file:readInt()
   -- read all the samples
   for i = 1,self.nbSamples do
      self[i] = file:readObject()
   end
end

local function split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
         table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end     

local function format_num(num)
   local out=''
   if num<10 then
      out = '000'..num
   elseif num>=10 and num<100 then
      out = '00'..num
   elseif num>=100 then
      out = '0'..num
   end
   return out
end

function pairwise_dataset:load(...)
   local args, logfile, data_dir, channels, nbSamplesRequired, sampleSize, padding, dataset
      = xlua.unpack(
                    {...},
                    'pairwise_dataset:load', 'load required lfw pair to the dataset object',
                    {arg='logfile', type='string', help='path to pairfile', req=true},
                    {arg='dataSetFolder', type='string', help='path to dataset', req=true},
                    {arg='channels', type='number', help='number of channels for the image to load', default=3},
                    {arg='nbSamplesRequired', type='number', help='max number of samples to load'},
                    {arg='sampleSize', type='table', help='resize all sample: {c,w,h}'},
                    {arg='padding', type='boolean', help='do we padd all the inputs in w,h'},
                    {arg='dataset', type='string', help='generate data from which dataset (e.g. VIPeR)', default='VIPeR'},
                    {arg='verbose', type='boolean', help='visulize loading process', default=false}
                 )
   
   local i=0
   
   nbSamplesRequired = nbSamplesRequired or 10000
   local image_list=nn.DataSet()

   while true do
      i=i+1

      local dataSetFolder = data_dir
      local files = sys.dir(dataSetFolder)

      for k,file in pairs(files) do
         local input, dinput

         if (string.find(file,'.bmp')) then
            -- load the PPM into a new Tensor
            pathToPpm = sys.concat(dataSetFolder, file)
            input = image.load(pathToPpm,channels)

            -- parse the file name and set the ouput from it
            rawOutput = string.gsub(file, ".bmp", "")
            personid = rawOutput:sub(1,4)
            if verbose then print('File: '..rawOutput..' Person: '..personid) end
         end

         -- if image loaded then add into the set
         if (input and rawOutput) then
         
            -- put input in 3D tensor
            input:resize(channels, input:size(2), input:size(3))

            -- rescale ?
            if sampleSize then
               dinput = torch.Tensor(channels, sampleSize[2], sampleSize[3])
               if padding then
                  offw = math.floor((sampleSize[2] - input[2])*0.5)
                  offh = math.floor((sampleSize[3] - input[3])*0.5)
                  if offw >= 0 and offh >= 0 then
                     dinput:narrow(2,offw,input[2]):narrow(3,offh,input[3]):copy(input)
                  else
                     print('reverse crop not implemented w,h must be larger than all data points')
                  end
               else
                  image.scale(input, dinput, 'bilinear')
               end
            else
               dinput = input
            end
            -- insert to image list
            image_list:add({input=dinput, output=personid})
         end
      end
      
   end
   
   --generate image pairs from image_list
   print('Generating...')
   local image_maped={}
   
   image_list:shuffle()
   while true do
      local i = torch.ceil(torch.uniform(1e-12, #image_list/2-1))
      local j = torch.ceil(torch.uniform(1e-12, #image_list/2) + #image_list/2)
      --local j = i+1
      xlua.progress(self.nbSamples, nbSamplesRequired)
      
      if self.nbSamples>= nbSamplesRequired then return end

      if not image_maped[i..j] and i~=j then
         image_maped[i..j] = true
         local new_data={}

         if image_list[i][2] == image_list[j][2] and self.match:size() < torch.floor(nbSamplesRequired/2) then
            new_data.input = {image_list[i][1], image_list[j][1]}
            new_data.output = 1
            self.match:add(new_data)
             table.insert(self,{new_data.input, new_data.output})
            self.nbSamples = self.nbSamples + 1
         elseif image_list[i][2] ~= image_list[j][2] and self.dismatch:size() < torch.ceil(nbSamplesRequired/2) then
            new_data.input = {image_list[i][1], image_list[j][1]}
            new_data.output = 2
            self.dismatch:add(new_data)
            table.insert(self,{new_data.input, new_data.output})
            self.nbSamples = self.nbSamples + 1
         end
      end
   end
   
   -- cleanup for memory
   --collectgarbage()
end


function pairwise_dataset:shuffle()
   if (self.nbSamples == 0) then
      print('Warning, trying to shuffle empty Dataset, no effect...')
      return
   end
   local n = self.nbSamples

   while n > 2 do
      local k = math.random(n)
      -- swap elements
      self[n], self[k] = self[k], self[n]
      n = n - 1
   end
end

function pairwise_dataset:popSubset(...)
   local args = xlua.unpack(
                    {...},
                    'pairwise_dataset:popSubset', 'pop out subset for validation',
                    {arg='overall', type='boolean', help='Pop subset on overall dataset or match/dismatch subdataset?', req=true},
                    {arg='nElement', type='number', help='how many elements', default=0},
                    {arg='ratio', type='number', help='what ratio to pop', default=0.1})
   local nElement = args.nElement
   local ratio = args.ratio
   local overall = args.overall
   local subset = LFWDataset()

   -- Stupidly implement two seperate parts of pop operarion, if have better solutions let me know
   if overall then
      -- get nb of samples to pop
      local start_index
      if (nElement ~= 0) then
         start_index = self:size() - nElement + 1
      else
         start_index = math.floor((1-ratio)*self:size()) + 1
      end

      -- info
      print('<DataSet> Popping ' .. self:size() - start_index + 1 .. ' samples dataset')

      -- extract samples
      for i = self:size(), start_index, -1 do
         subset.nbSamples = subset.nbSamples + 1
         table.insert(subset,self[i])
         local popdata={}
         if self[i][2]==1 then
            popdata.input={self[i][1][1]:clone(), self[i][1][2]:clone()}
            popdata.output=self[i][2]
            subset.match:add(popdata)
         else
            popdata.input={self[i][1][1]:clone(), self[i][1][2]:clone()}
            popdata.output=self[i][2]
            subset.dismatch:add(popdata)
         end
         self[i] = nil
         self.nbSamples = self.nbSamples - 1
      end
      -- return network
      return subset
   else
      -- get nb of samples to pop
      local start_index1, start_index2
      if (nElement ~= 0) then
         start_index1 = self.match:size() - nElement + 1
         start_index2 = self.dismatch:size() - nElement + 1
      else
         start_index1 = math.floor((1-ratio)*self.match:size()) + 1
         start_index2 = math.floor((1-ratio)*self.dismatch:size()) + 1
      end

      -- info
      print('<DataSet> Popping ' .. self.match:size() - start_index1 + 1 .. '|' .. self.dismatch:size() - start_index2 + 1 .. ' samples dataset')
      
       -- extract samples
      for i = self.match:size(), start_index1, -1 do
         local popdata={}
         popdata.input={self.match[i][1][1]:clone(), self.match[i][1][2]:clone()}
         popdata.output=self.match[i][2]
         subset.match:add(popdata)
         self.match[i] = nil
         self.match.nbSamples = self.match.nbSamples - 1
      end

      for i = self.dismatch:size(), start_index2, -1 do
         local popdata={}
         popdata.input={self.dismatch[i][1][1]:clone(), self.dismatch[i][1][2]:clone()}
         popdata.output=self.dismatch[i][2]
         subset.dismatch:add(popdata)
         self.dismatch[i] = nil
         self.dismatch.nbSamples = self.dismatch.nbSamples - 1
      end
      -- return network
      return subset.match, subset.dismatch
   end

end

function pairwise_dataset:display(nSamples,legend)
   local samplesToShow = {}
   for i = 1,nSamples do
      table.insert(samplesToShow, image.toDisplayTensor(self[i][1]))
   end
   image.display{image=samplesToShow,gui=false,legend=legend}
end

function pairwise_dataset:displayBoth(nSamples)
   local samplesToShow1 = {}
   local samplesToShow2 = {}
   for i = 1,nSamples do
      table.insert(samplesToShow1, image.toDisplayTensor(self.match[i][1]))
      table.insert(samplesToShow2, image.toDisplayTensor(self.dismatch[i][1]))
   end
   image.display{image=samplesToShow1,gui=false,legend='Match'}
   image.display{image=samplesToShow2,gui=false,legend='Mismatch'}
end