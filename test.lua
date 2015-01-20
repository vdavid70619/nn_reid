require('mobdebug').start()


require 'dataloader'

dl = Dataloader()
dl:load_from_folder('/Users/xiyangdai/Documents/DATA/VIPeR/cam_a_jpg/')
dl:resize(20,20)
dl:shuffle()
train, test = dl:train_test_split()
pdata = dl.generate

debug=1