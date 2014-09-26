require('parallel')

function worker()
   require('torch')
   --parallel.print('me')
   
   while true do
      local m = parallel.yield()
      if m == 'break' then break end
      --parallel.print('working')

      num = torch.uniform(0,1)
      --parallel.print(num)
      if num<1e-2 then 
         --parallel.print('i have one! ' .. num)
         parallel.parent:send(num) 
      else
         parallel.parent:send(nil)
      end
      
   end
end

function parent()
   
   local list = {}

   parallel.nfork(100)
   parallel.children:exec(worker)
   
   parallel.print('who!')
   while true do
      parallel.children:join()

      --parallel.print('waiting')
      product = parallel.children:receive()
      --parallel.print('get')
      
      for i = 1,parallel.nchildren do
         if product[i]~=nil then
            parallel.print(product[i])
            table.insert(list,product[i])

            parallel.print(#list)   
         end
      end 


  
      if #list >= 1000 then break end
   end
   
   parallel.children:join('break')
   parallel.close()
   --print(list)
   

end

ok,err = pcall(parent)
if not ok then 
   print(err) 
   parallel.children:join('break')
   parallel.close() 
end   
      