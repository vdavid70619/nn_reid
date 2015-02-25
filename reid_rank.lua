--
-- Xiyang Dai
-- 2012.11
--
--


torch.setdefaulttensortype('torch.DoubleTensor')

require 'nn'
require 'nnx'
require 'image'
   
torch.setnumthreads(8)


-----------------------------------------------------------------------------------------
-- Load Model
-----------------------------------------------------------------------------------------

network = torch.load('./model.net')
parameters, gradParameters = network:getParameters()
parameters:copy(torch.load('./reid/face_vertify_121209_0047.param'))


-----------------------------------------------------------------------------------------
-- Load Data
-----------------------------------------------------------------------------------------
local dl = Dataloader()
dl:load_from_folder('../../MATLAB/DATA/i-LIDS-VID/images/mix', '.png')
--dl:rgb2grey()
dl:resize(50,50)
fold_conf = torch.load('')
trains, tests = dl:fold(fold_conf)







function similarity_score(im1, im2)


end
