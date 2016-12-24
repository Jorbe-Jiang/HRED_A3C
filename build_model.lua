----------------------------------------------------
--model structure
----------------------------------------------------
require('myCriterionBase')
require('myCriterion')
require('myMaskZeroCriterion')
require('mySequencerCriterion')

local emb = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
if opt.load_embeddings == 1 then
	print('Loading pre-trained word vectors from: '..opt.load_embeddings_file)
	local trained_word_vecs = npy4th.loadnpy(opt.load_embeddings_file)
	emb.weight = trained_word_vecs
end

function RNN_elem(recurrence)
	local utterance_rnn = nn.Sequential()
	local hred_enc_embeddings = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
	hred_enc_embeddings.weight = emb.weight:clone()
	utterance_rnn:add(hred_enc_embeddings)

	local rnn = recurrence(opt.word_dim, opt.enc_hidden_size)
	utterance_rnn:add(nn.Sequencer(rnn:maskZero(1)))

    	--batch norm
    	--utterance_rnn:add(nn.Sequencer(nn.BatchNormalization(opt.enc_hidden_size)))

	if opt.drop_rate > 0 then
		utterance_rnn:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end

	utterance_rnn:add(nn.Select(1, -1))

	return utterance_rnn
end

--build hred encoder(utterance encoder and context encoder)
function build_hred_encoder(recurrence)
	local hred_enc = nn.Sequential()
	local hred_enc_rnn
	local par = nn.ParallelTable()
	
	--build parallel utterance rnns
	local rnns = {}
	for i = 1, 2 do
		table.insert(rnns, RNN_elem(recurrence))
	end

	rnns[2] = rnns[1]:clone('weight', 'bias', 'gradWeight', 'gradBias') --utterance 2 rnn share the weight with utterance 1 rnn

	for i = 1, 2 do
		par:add(rnns[i])
	end

	hred_enc:add(par)
	
	hred_enc:add(nn.JoinTable(1, 2))
	hred_enc:add(nn.Reshape(2, opt.batch_size, opt.enc_hidden_size))

	--build context layer
	local context_layer = nn.Sequential()
	hred_enc_rnn = recurrence(opt.enc_hidden_size, opt.context_hidden_size)
	context_layer:add(nn.Sequencer(hred_enc_rnn:maskZero(1)))

    	--batch norm
    	--context_layer:add(nn.Sequencer(nn.BatchNormalization(opt.context_hidden_size)))
    
	if opt.drop_rate > 0 then
		context_layer:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end

	context_layer:add(nn.Select(1, -1))
	hred_enc:add(context_layer)

	return hred_enc, hred_enc_rnn
end


--build decoder and value net
function build_dec_and_val_net(recurrence)
	local dec_and_val_net = nn.Sequential()
	local dec_embeddings = nn.LookupTableMaskZero(opt.vocab_size, opt.word_dim):setMaxNorm(2)
	dec_embeddings.weight = emb.weight:clone()
	dec_and_val_net:add(dec_embeddings)
	
	local dec_rnn = recurrence(opt.word_dim, opt.dec_hidden_size)
	dec_and_val_net:add(nn.Sequencer(dec_rnn:maskZero(1)))
	if opt.drop_rate > 0 then
		dec_and_val_net:add(nn.Sequencer(nn.Dropout(opt.drop_rate)))
	end
	
	local con_table = nn.ConcatTable()
	local dec_linear = nn.Linear(opt.dec_hidden_size, opt.vocab_size)
	local val_net_linear = nn.Linear(opt.dec_hidden_size, 1)
	local dec_tmp = nn.Sequential()
	dec_tmp:add(nn.Sequencer(nn.MaskZero(dec_linear, 1)))
	dec_tmp:add(nn.Sequencer(nn.MaskZero(nn.SoftMax(), 1)))
	con_table:add(dec_tmp)
	con_table:add(nn.Sequencer(nn.MaskZero(val_net_linear, 1)))
	dec_and_val_net:add(con_table)
	
	return dec_and_val_net, dec_rnn
end

--build model 
function build()
	local recurrence = nn[opt.cell]
	print('Building model...')
	print('Layer type: '..opt.cell)
	print('Vocab size: '..opt.vocab_size)
	print('Embedding size: '..opt.word_dim)
	print('Encoder layer hidden size: '..opt.enc_hidden_size)
	print('Context layer hidden size: '..opt.context_hidden_size)
	print('Decoder layer hidden size: '..opt.dec_hidden_size)
	print('Top k actions: '..opt.top_k_actions)
	
	--my criterion
	local my_criterion = nn.MySequencerCriterion(nn.myCriterion()) 
	local value_criterion = nn.MSECriterion()

	local hred_enc, hred_enc_rnn, dec_and_val_net, dec_rnn
	
	-- whether to load pre-trained model from load_model_file
	if opt.load_model == 0 then
		hred_enc, hred_enc_rnn = build_hred_encoder(recurrence)
		dec_and_val_net, dec_rnn = build_dec_and_val_net(recurrence)
		
	else
		--load the trained model
		assert(path.exists(opt.load_model_file), 'check the model file path')
		print('Loading model from: '..opt.load_model_file..'...')
		local model_and_opts = torch.load(opt.load_model_file)
		local model, model_opt = model_and_opts[1], model_and_opts[2]
		--opt = model_opt
		
		--load the model components
		hred_enc = model[1]:double()
		dec_and_val_net = model[2]:double()
		hred_enc_rnn = model[3]:double()
		dec_rnn = model[4]:double()
		--if batch_size is changed
		hred_enc:remove(3)
		hred_enc:insert(nn.Reshape(2, opt.batch_size, opt.enc_hidden_size), 3)
	end
	
	local layers = {hred_enc, dec_and_val_net}
	
	--run on GPU
	if opt.gpu_id >= 0 then
		for i = 1, #layers do
			layers[i]:cuda()
		end
		my_criterion:cuda()
		value_criterion:cuda()
	end
	local Model = nn.Sequential()
	Model:add(hred_enc)
	Model:add(dec_and_val_net)
	local params, grad_params = Model:getParameters()

	if opt.gpu_id >= 0 then
		params:cuda()
		grad_params:cuda()
	end

	--package model for training
	local model = {
		hred_enc,
		hred_enc_rnn,
		dec_and_val_net,
		dec_rnn,
		params,
		grad_params
	}
	
	print('Building model successfully...')
	return model, my_criterion, value_criterion
end
