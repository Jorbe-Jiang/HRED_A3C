--------------------------------------------------
--some extra functions
--------------------------------------------------
function idx_2_word()
	local vocab_file = './datas/VOCAB.dat'
	local idx_2_word = {}
	for line in io.lines(vocab_file) do
		table.insert(idx_2_word, line)
	end
	return idx_2_word
end

function word_2_idx()
	local vocab_file = './datas/VOCAB.dat'
	local word_2_idx = {}
	local word_idx = 1
	for line in io.lines(vocab_file) do
		word_2_idx[line] = word_idx
		word_idx = word_idx + 1
	end
	return word_2_idx
end

function forward_connect(hred_enc_rnn, dec_rnn)
	dec_rnn.userPrevOutput = nn.rnn.recursiveCopy(dec_rnn.userPrevOutput, hred_enc_rnn.outputs[2])
	if opt.cell ~= 'GRU' then
		dec_rnn.userPrevCell = nn.rnn.recursiveCopy(dec_rnn.userPrevCell, hred_enc_rnn.cells[2])
	end
end

function translate_to_word(U3_words_idx_table)
	local U3_words_table = {}
	local n_words = #U3_words_idx_table
	local n = 1
	while (U3_words_idx_table[n] ~= 1 or U3_words_idx_table[n] ~= 0) and n <= n_words do
		local word = idx_2_word_table[U3_words_idx_table[n]]
		table.insert(U3_words_table, word)
		n = n+1
	end
	return U3_words_table
end

function argmax(pred_output)
	local max_value, max_idx = torch.max(pred_output, 1)
	return max_idx
end

function padding_zero(Uxs, Uxs_maxlen)
	local Uxs_len = table.getn(Uxs)
	for i = 1, Uxs_len do
		for j = table.getn(Uxs[i])+1, Uxs_maxlen do
			Uxs[i][j] = 0
		end
	end
	return Uxs
end

function test(model, batches_test_data)
	print('Begin to test...')
	local idx_2_word_table = idx_2_word()
	local date = os.date("%m_%d") --today's date
	opt.batch_size = 1 --when test only one sample process once time
	local hred_enc, dec_and_val_net, hred_enc_rnn, dec_rnn, dec_embeddings
	hred_enc = model[1]:double()
	dec_and_val_net = model[2]:double()
	hred_enc_rnn = model[3]:double()
	dec_rnn = model[4]:double()
	--if batch size is changed
	hred_enc:remove(3)
    	hred_enc:insert(nn.Reshape(2, opt.batch_size, opt.enc_hidden_size), 3) 
	dec_embeddings = dec_and_val_net:get(1).weight

	local U1s_batches_enc, U2s_batches_enc, U2s_batches_dec, U3s_batches_dec, U2s_batches_tar, U3s_batches_tar, nbatches
	U1s_batches_enc = batches_test_data[1]
	U2s_batches_enc = batches_test_data[2]
	U2s_batches_dec = batches_test_data[3]
	U3s_batches_dec = batches_test_data[4]
	U2s_batches_tar = batches_test_data[5]
	U3s_batches_tar = batches_test_data[6]
	nbatches = batches_test_data[7]

	local hred_enc_inputs
	local hred_enc_outputs 
	local U3s_pred = {}
	local U3s_true = {}
	local max_U3s_pred_len = 0
	local max_U3s_true_len = 0
	
	local start_emb = dec_embeddings[2] --<srt> embedding
	print(dec_embeddings[1]) --zero embedding for test
	
	local max_len = 100
	
	local linear = nn.Linear(opt.dec_hidden_size, opt.vocab_size):clone(dec_and_val_net:get(4):get(1):get(1), 'weight', 'bias')

	for i = 1, nbatches do	
		for j = 1, U1s_batches_enc[i]:size(2) do
			hred_enc:evaluate()
			dec_and_val_net:evaluate()

			local U3_true = torch.totable(U3s_batches_tar[i]:select(2, j))
			hred_enc_inputs = {U1s_batches_enc[i]:narrow(2, j, 1), U2s_batches_enc[i]:narrow(2, j, 1)}
			--[[
			if opt.gpu_id >= 0 then
				for i = 1, 2 do
					hred_enc_inputs[i] = hred_enc_inputs[i]:int():cuda()
				end
			end
			]]--
			hred_enc_outputs = hred_enc:forward(hred_enc_inputs)
			collectgarbage()
			forward_connect(hred_enc_rnn, dec_rnn)
			local U3_pred = {}
			local dec_pred_idx = 1
			local n = 1
			while dec_pred_idx ~= 2 and n < max_len do
				if n == 1 then
					dec_rnn:forward(start_emb)
				else
					dec_rnn:forward(dec_embeddings[dec_pred_idx + 1])
				end
				collectgarbage()
				dec_pred_output = nn.SoftMax():forward(linear:forward(dec_rnn.outputs[n]))
				dec_pred_idx = argmax(dec_pred_output)
				table.insert(U3_pred, dec_pred_idx)
				n = n+1
			end
			
			local print_U3_pred_words_table = translate_to_word(U3_pred)
			local print_U3_true_words_table = translate_to_word(U3_true)
			print('U3 pred:------------------------------------>')
			--print(print_U3_pred_words_table)
			--print(print_U3_true_words_table)
			--os.exit()
			--print(U3s_pred_idx:narrow(1, 1, 3))
			print(table.concat(print_U3_pred_words_table, ' '))
			print('U3 true:------------------------------------>')
			--print(U3s_true_idx:narrow(1, 1, 3))
			print(table.concat(print_U3_true_words_table, ' '))

			table.insert(U3s_pred, U3_pred)
			table.insert(U3s_true, U3_true)
			if table.getn(U3_pred) > max_U3s_pred_len then
				max_U3s_pred_len = table.getn(U3_pred)
			end
			
			if table.getn(U3_true) > max_U3s_true_len then
				max_U3s_true_len = table.getn(U3_true)
			end
		end
	end
	--saving the results
	U3s_pred = padding_zero(U3s_pred, max_U3s_pred_len)
	U3s_true = padding_zero(U3s_true, max_U3s_true_len)
	local pred_result_file = string.format("./results/%s_%d_%d_test_pred_result.npy", date, opt.num_epochs, opt.batch_size)
	local true_result_file = string.format("./results/%s_%d_%d_test_true_result.npy", date, opt.num_epochs, opt.batch_size)
	print('Saving results to: '..pred_result_file..' and '..true_result_file)
	npy4th.savenpy(pred_result_file, torch.Tensor(U3s_pred))
	npy4th.savenpy(true_result_file, torch.Tensor(U3s_true))
end
