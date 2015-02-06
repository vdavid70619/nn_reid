--[[
	Re-id dataloader for torch
	Xiyang Dai @ 2015.1
]]--


require 'torch'
require 'image'
require 'paths'


local dataloader = torch.class('Dataloader')

function dataloader:__init(...)
	-- constructor
	self.data = {}
end

function dataloader:insert_new_view(id, im)
	if self.data[id] then
		table.insert(self.data[id], im)
	else
		self.data[id] = {im}
	end
end

function dataloader:name2id(name, ...)
	local id
	-- default name id parsing rule from directory 
	id = name

	return id
end

function dataloader:load_from_folder(dname, nimage)
	math.inf = 1/0
	nimage = nimage or math.inf
	local dirs = paths.dir(dname)

	for _, dir_path in pairs(dirs) do
		print(dir_path)
		if(paths.dirp(dname..'/'..dir_path) and dir_path~='.' and dir_path~='..') then 
			local images = paths.dir(dname..'/'..dir_path)
			local nt = nimage
			if nt<math.inf then
		   		local n = #images

			   	while n > 2 do
			      	local k = math.random(n)
			      	-- swap elements
			      	images[n], images[k] = images[k], images[n]
			      	n = n - 1
			   	end				
			end

			for _, im_path in pairs(images) do
				if(string.find(im_path, '.jpg') or string.find(im_path, '.png')) then
					if nt<=0 then break end

					print(im_path)
					id = self:name2id(dir_path)
					im = image.load(dname..'/'..dir_path..'/'..im_path)
					self:insert_new_view(id, im)

					nt = nt - 1
				end
			end
		end
	end
end

function dataloader:__tostring__()
   	str = 'Classes\n'
   	for class, images in pairs(self.data) do
   		str = str .. class .. ': ' .. #images .. '\n'
   	end
   	return str
end


function dataloader:shuffle()
   	for classes, images in pairs(self.data) do
   		local n = #images

	   	while n > 2 do
	      	local k = math.random(n)
	      	-- swap elements
	      	images[n], images[k] = images[k], images[n]
	      	n = n - 1
	   	end
	end
end

function dataloader:resize(width, height)
   	for classes, images in pairs(self.data) do
   		--print(#images)
   		for i=1,#images do
   			images[i] = image.scale(images[i], width, height)
   		end
	end
end

function dataloader:rgb2grey()
   	for classes, images in pairs(self.data) do
   		--print(#images)
   		for i=1,#images do
   			if images[i]:size(1) == 3 then
   				images[i] = image.rgb2y(images[i])
   			end
   		end
	end
end 

function dataloader:train_test_split(ratio, opt)
	opt = opt or 'inner'

	self:shuffle()
	train = {}
	test = {}

	if opt == 'inner' then
		for class, images in pairs(self.data) do
			n = #images
			if ratio < 1 then
				nt = math.floor(n*ratio)
			else 
				nt = ratio
			end

			train[class] = {unpack(images, 1, nt)}
			test[class] = {unpack(images, nt+1, n)} 
		end
	elseif opt == 'inter' then
		classes = {}
		for class, _ in pairs(self.data) do
			table.insert(classes, class)
		end

		local n = #classes
	   	while n > 2 do
	      	local k = math.random(n)
	      	-- swap elements
	      	classes[n], classes[k] = classes[k], classes[n]
	      	n = n - 1
	   	end		

		local n = #classes
		if ratio < 1 then
			nt = math.floor(n*ratio)
		else 
			nt = ratio
		end

		for class, images in pairs(self.data) do
			if nt>0 then 
				train[class] = images
			else
				test[class] = images
			end
			nt = nt - 1
		end
	end
	return train, test
end

function dataloader.generate_pairs(data)
	match_list = {}
	dismatch_list = {}

	pre_nimage = 0;
	pre_class = '';
	for class, images in pairs(data) do
		nimage = #images
		print(nimage)
  		for i=1,nimage do
			for j=i+1,nimage do
				table.insert(match_list,{{class,i},{class,j}})
			end

			if pre_nimage~=0 and pre_class~='' then
				for j=1,pre_nimage do
					table.insert(dismatch_list,{{pre_class,j},{class,i}})
				end
			end
		end

		pre_class = class
		pre_nimage = nimage
	end

	pairdata = {__raw = data, __match = match_list, __dismatch = dismatch_list, size = 2*math.max(#match_list, #dismatch_list)}
   	setmetatable(pairdata, {__index = function(self, key)
   										ind = math.ceil(key/2)
   										if math.fmod(key,2) == 1 then
   											-- match list
   											ind = math.fmod(key, #self.__match)
                                       		if ind==0 then ind=#self.__match end
                                       		image1 = self.__raw[self.__match[ind][1][1]][self.__match[ind][1][2]]
                                       		image2 = self.__raw[self.__match[ind][2][1]][self.__match[ind][2][2]]
                                       		return {image1,image2,1}
                                       	else
   											-- dismatch list
   											ind = math.fmod(key, #self.__dismatch)
                                       		if ind==0 then ind=#self.__dismatch end
                                       		image1 = self.__raw[self.__dismatch[ind][1][1]][self.__dismatch[ind][1][2]]
                                       		image2 = self.__raw[self.__dismatch[ind][2][1]][self.__dismatch[ind][2][2]]
                                       		return {image1,image2,0}
                                       	end
                                    end})
   	return pairdata	
end




