import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_size=512

def get_positional_encodings(sequence):
  #sequence: list of strings
  seq_length=len(sequence)
  indices=seq_length//2
  rem=seq_length%2
  pos_encoding=torch.zeros(seq_length).to(device)
  #fill indexes
  for idx in range(indices):
    even_idx=2*idx
    odd_idx=2*idx+1
    pos_encoding[even_idx]=np.sin(even_idx/10000**(2*idx/model_size))
    if odd_idx<seq_length:
      pos_encoding[odd_idx]=np.cos(odd_idx/10000*(2*idx/model_size))
  return pos_encoding

class PositionalFF(nn.Module):
  def __init__(self, ff_hidden_size=2048):
    #positional feed forward network after sublayers of encoder & decoder
    super(PositionalFF, self).__init__()
    self.linear1=nn.Linear(model_size, ff_hidden_size).to(device)
    self.linear2=nn.Linear(ff_hidden_size, model_size).to(device)
    self.element_init()
  
  def element_init(self):
    for element in self.children():
      if isinstance(element, nn.Linear):
        nn.init.xavier_uniform_(element.weight)
        nn.init.zeros_(element.bias)
  
  def forward(self, x):
    gelu=nn.GELU()
    layers=nn.Sequential(self.linear1, gelu, self.linear2)
    output=layers(x)
    return output

class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, mask=True):
    super(MultiHeadAttention, self).__init__()
    self.n_head=n_head
    self.mask=mask
    self.q_projection=nn.Linear(model_size, model_size).to(device)
    self.k_projection=nn.Linear(model_size, model_size).to(device)
    self.v_projection=nn.Linear(model_size, model_size).to(device)
    self.output_projection=nn.Linear(model_size, model_size).to(device)
    #dropouts
    self.output_dropout=nn.Dropout(p=0.1)
    self.attention_dropout=nn.Dropout(p=0.1)
    #initialize linear layers
    self.element_init()
  
  def element_init(self):
    for element in self.children():
      if isinstance(element, nn.Linear):
        nn.init.xavier_uniform_(element.weight)
        nn.init.zeros_(element.bias)
  
  def get_mask_tensor(self, batch_size, n_head, seq_length):
    #create mask matrix
    base=torch.ones(batch_size, n_head, seq_length, seq_length).to(device)
    mask_tensor=torch.triu(base, diagonal=1).bool()
    return mask_tensor
  
  def forward(self, K, Q, V):
    softmax=nn.Softmax(dim=-1)
    #input: shape of batch_size*seq_length*model_size => batch of padded sequences
    batch_size, seq_length, kd=K.size()
    key_dim=kd//self.n_head
    value_dim=V.size(2)//self.n_head
    queries=self.q_projection(Q)
    keys=self.k_projection(K)
    values=self.v_projection(V)
    #use multi-head => into shape of (batch_size, n_head, seq_length, key_dim)
    queries=queries.reshape(batch_size, seq_length, self.n_head, key_dim).permute(0,2,1,3)
    keys=keys.reshape(batch_size, seq_length, self.n_head, key_dim).permute(0,2,1,3)
    values=values.reshape(batch_size, seq_length, self.n_head, value_dim).permute(0,2,1,3)
    #use scaled dot product attention.
    affinity_scores=torch.div(queries.matmul(keys.permute(0,1,3,2)), np.sqrt(model_size))
    if self.mask:
      #add positional affinity score masks.
      mask_tensor=self.get_mask_tensor(batch_size, self.n_head, seq_length)
      affinity_scores.masked_fill_(mask_tensor, float('-inf'))
    attention_weight=self.attention_dropout(softmax(affinity_scores))
    #get values as weighted sum: shape=(batch_size, n_head, seq_length, key_dim)
    attention_output=attention_weight.matmul(values)
    #flatten values => project
    attention_output=attention_output.permute(0,2,1,3)
    attention_output=attention_output.reshape(batch_size, seq_length, -1)
    output_tensor=self.output_dropout(self.output_projection(attention_output))
    return output_tensor

class EncoderBlock(nn.Module):
  def __init__(self, n_head):
    super(EncoderBlock, self).__init__()
    self.mha=MultiHeadAttention(n_head=n_head, mask=False)
    self.positional_ff=PositionalFF()
    self.layer_norm=nn.LayerNorm(model_size).to(device)
  
  def forward(self, input):
    #input of size (batch_size, seq_length, model_dim(=embedding size))
    mha_output=self.mha(K=input, Q=input, V=input)
    mha_post_ln=self.layer_norm(input+mha_output)
    ff_output=self.positional_ff(mha_post_ln)
    encoder_output=self.layer_norm(mha_post_ln+ff_output)
    return encoder_output

class DecoderBlock(nn.Module):
  def __init__(self, n_head):
    super(DecoderBlock, self).__init__()
    self.masked_mha=MultiHeadAttention(n_head=n_head, mask=True)
    self.cross_mha=MultiHeadAttention(n_head=n_head, mask=False)
    self.positional_ff=PositionalFF()
    self.layer_norm=nn.LayerNorm(model_size).to(device)

  def forward(self, input, final_encoder_output):
    #input of size (batch_size, seq_length, model_dim(=embedding size))
    mha_output=self.masked_mha(K=input, Q=input, V=input)
    mha_post_ln=self.layer_norm(input+mha_output)
    cross_mha_output=self.cross_mha(K=final_encoder_output, Q=mha_post_ln, V=final_encoder_output)
    cross_mha_post_ln=self.layer_norm(mha_post_ln+cross_mha_output)
    ff_output=self.positional_ff(cross_mha_output)
    decoder_output=self.layer_norm(cross_mha_post_ln+ff_output)
    return decoder_output

class Transformer(nn.Module):
  def __init__(self, n_head, vocab_size):
    super(Transformer, self).__init__()
    self.n_head=n_head
    self.vocab_size=vocab_size
    #encoder blocks
    self.encoder1=EncoderBlock(self.n_head)
    self.encoder2=EncoderBlock(self.n_head)
    self.encoder3=EncoderBlock(self.n_head)
    self.encoder4=EncoderBlock(self.n_head)
    self.encoder5=EncoderBlock(self.n_head)
    self.encoder6=EncoderBlock(self.n_head)
    #decoder blocks
    self.decoder1=DecoderBlock(self.n_head)
    self.decoder2=DecoderBlock(self.n_head)
    self.decoder3=DecoderBlock(self.n_head)
    self.decoder4=DecoderBlock(self.n_head)
    self.decoder5=DecoderBlock(self.n_head)
    self.decoder6=DecoderBlock(self.n_head)

    self.output_linear=nn.Linear(model_size, self.vocab_size).to(device)

  def forward(self, input_sequence, output_sequence):
    softmax=nn.Softmax(dim=-1)
    #input, output sequence: embedding+pos. encoding
    encoders=nn.Sequential(self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5, self.encoder6)
    final_encoder_output=encoders(input_sequence)
    #decoder phase
    do1=self.decoder1(output_sequence, final_encoder_output)
    do2=self.decoder2(do1, final_encoder_output)
    do3=self.decoder3(do2, final_encoder_output)
    do4=self.decoder4(do3, final_encoder_output)
    do5=self.decoder5(do4, final_encoder_output)
    final_decoder_output=self.decoder6(do5, final_encoder_output)
    probabilities=softmax(self.output_linear(final_decoder_output))
    return probabilities