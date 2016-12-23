local THNN = require 'nn.THNN'
local myCriterion, parent = torch.class('nn.myCriterion', 'nn.MyCriterion')

function myCriterion:__init(sizeAverage)
	parent.__init(self)
	if sizeAverage ~= nil then
		self.sizeAverage = sizeAverage
	else
		self.sizeAverage = true
	end

	self.target = torch.zeros(1):long()
end

function myCriterion:updateOutput(input, target, rewards)
	if type(target) == 'number' then
		if input:type() ~= 'torch.CudaTensor' then
			self.target = self.target:long()
		end
		self.target[1] = target
	elseif target:type() == 'torch.CudaTensor' then
		self.target = target
	else
		self.target = target:long()
	end

	self.rewards = rewards --[batch_size, vocab_size]
	--print(self.rewards:size())
	--print(input:size())
	--os.exit()

	local n_dims = input:dim()
	if target:dim() > 1 then
		assert("multi-target not supported")
	end
	
	if n_dims > 2 then
		assert("input tensor should be 1D or 2D")
	end
	
	self.output = 0.0

	local batch_size = input:size(1)
	
	for i = 1, batch_size do
		local policy_func_value = input[i]:double()
		if policy_func_value ~= 0 then
			local t_entropy_cost = -torch.dot(policy_func_value, torch.log(policy_func_value+opt.eps*1e-4))
			--self.output = self.output - torch.log(policy_func_value+opt.eps*1e-4) * self.rewards[i] + opt.beta * t_entropy_cost
			self.output = self.output - torch.dot(policy_func_value, self.rewards[i]:double()) + opt.beta * t_entropy_cost
		end
	end

	if self.sizeAverage then
		self.output = self.output / batch_size
	end

	return self.output
end

function myCriterion:updateGradInput(input, target, rewards)
	if type(target) == 'number' then
		if input:type() ~= 'torch.CudaTensor' then
			self.target = self.target:long()
		end
		self.target[1] = target
	elseif target:type() == 'torch.CudaTensor' then
		self.target = target
	else
		self.target = target:long()
	end

	self.rewards = rewards --[batch_size, vocab_size]
	self.gradInput:resizeAs(input:double()):zero()
	local n_dims = input:dim()
	
	if not self.gradInput:isContiguous() then	
		assert("gradInput must be contiguous")
	end
	
	if self.target:dim() > 1 then
		assert("multi-target not supported")
	end

	if n_dims > 2 then
		assert("input tensor should be 1D or 2D")
	end

	local batch_size = input:size(1)
	assert(self.target:size(1) == batch_size)

	for i = 1, batch_size do
		--local policy_func_value = input[i]:double()
		local cp_rewards = torch.Tensor(opt.vocab_size):copy(self.rewards[i]:double())
		local cp_policy_func_value = torch.Tensor(opt.vocab_size):copy(input[i]:double())
		if self.sizeAverage then
			self.gradInput[i] = (-cp_rewards:cdiv(cp_policy_func_value+opt.eps)-opt.beta*(torch.log(cp_policy_func_value+opt.eps*1e-4)+1)) / batch_size
			--self.gradInput[i] = (-self.rewards[i] - opt.beta*(torch.log(policy_func_value+opt.eps*1e-4)+1)) / batch_size
		else
			self.gradInput[i] = -cp_rewards:cdiv(cp_policy_func_value+opt.eps)-opt.beta*(torch.log(cp_policy_func_value+opt.eps*1e-4)+1)
			--self.gradInput[i] = -self.rewards[i] - opt.beta*(torch.log(policy_func_value+opt.eps*1e-4)+1)
		end
	
		--[[
		-- the desigh of the loss function have some problem, with the first part of the gradient is too big	
		if torch.abs(self.gradInput[i][1]) > 3.0 then	
			print(-self.rewards[i])
			print(policy_func_value+opt.eps*1e-4)	
			print(-self.rewards[i]/(policy_func_value+opt.eps*1e-4))
			print(opt.beta*(torch.log(policy_func_value+opt.eps*1e-4)))
			print(policy_func_value/(policy_func_value+opt.eps*1e-4))
			os.exit()
		end
		]]--
	end

	return self.gradInput
end
	
