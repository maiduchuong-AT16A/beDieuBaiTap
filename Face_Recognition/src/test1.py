text_encoder = RNN_ENCODER(self.n_words,nhidden = cfg.TEXT.EBEDDING_DIM)
#self.n_words:5450  nhidden:256

class RNN_ENCODER(nn.Module):
    def __init__( self, ntoken, ninput=300, drop_prob=0.0, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # kích thước từ điển
        self.ninput = ninput  # kích thước mỗi vector nhúng
        self.drop_prob = drop_prob  # xác suất của mỗi phần tử bằng 0
        self.nlayers = nlayers  # số lớp tái phát
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()
    def forward(self, captions, cap_lens, hidden, mask=None):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        if self.rnn_type == "LSTM":
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb
