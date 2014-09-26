require('mobdebug').start()


require 'dataloader'

dl = Dataloader()
dl:load_from_folder('/Users/xiyangdai/Documents/DATA/VIPeR/cam_a_jpg/')
debug=1