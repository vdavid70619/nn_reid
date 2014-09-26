--[[
	Re-id dataloader for torch
	Xiyang Dai @ 2014.8
]]--


require 'torch'
require 'image'
require 'paths'


local dataloader = torch.class('Dataloader')


function dataloader:__init(...)
	-- constructor
	self.raw = {}
end

function dataloader:insert_new_view(id, im)
	if self.raw[id] then
		table.insert(self.raw[id], im)
	else
		self.raw[id] = {im}
	end
end

function dataloader:name2id(name, ...)
	local id;
	-- default VIPeR id parsing rule
	id = name:sub(1,3)

	return id
end

function dataloader:load_from_folder(dname)
	local files = paths.dir(dname)

	for _, file in pairs(files) do
		--print(file)
		if(string.find(file, '.jpg')) then
			print(id)
			id = self:name2id(file)
			im = image.load(dname..file)
			self:insert_new_view(id, im)
		end
	end

end