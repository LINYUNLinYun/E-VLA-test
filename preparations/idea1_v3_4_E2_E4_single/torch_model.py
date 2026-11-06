"""Autograd-capable functional pi0 prefix and complete AE from released fused weights.

Fixed vision embeddings/normalization/rotary tables are supplied by the original
preprocessor. No Triton calls occur in this model. FP32 mechanism branch; weights
are frozen, and native BF16 arithmetic equality is deliberately not asserted.
"""
import math
import torch
import torch.nn.functional as F

def rms(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)

def rope(x,r):
    shape=x.shape
    a=x.reshape(shape[0],-1,128,2); rr=r.reshape(shape[0],1,128,2)
    return torch.stack((a[...,0]*rr[...,0]-a[...,1]*rr[...,1],a[...,1]*rr[...,0]+a[...,0]*rr[...,1]),-1).reshape(shape)

def attend(q,k,v,mask=None):
    # Token-major eight query heads, a single shared KV head.
    qq=q.reshape(len(q),8,256).permute(1,0,2)[None]
    kk=k[None,None].expand(1,8,-1,-1);vv=v[None,None].expand(1,8,-1,-1)
    y=F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mask,dropout_p=0.,scale=256**-.5)
    return y[0].permute(1,0,2).reshape(len(q),2048)

def hadamard(x):
    shape=x.shape;y=x
    size=1
    while size<shape[-1]:
        y=y.reshape(*shape[:-1],-1,2,size)
        a,b=y.unbind(-2);y=torch.stack((a+b,a-b),-2).reshape(shape);size*=2
    return y/math.sqrt(shape[-1])

class Model:
    def __init__(self,path,device='cuda'):
        data=torch.load(path,map_location='cpu',weights_only=False)
        self.w={k:v.to(device=device,dtype=torch.float32) for k,v in data['weights'].items()}
        self.biases=[v.to(device=device,dtype=torch.float32) for v in data['biases']]
        self.outw=[v.to(device=device,dtype=torch.float32) for v in data['outw']]
        self.outb=[v.to(device=device,dtype=torch.float32) for v in data['outb']]
        self.device=device

    def item(self,path):
        return {k:(v.to(device=self.device,dtype=torch.float32) if torch.is_tensor(v) else v) for k,v in torch.load(path,map_location='cpu',weights_only=False).items()}

    def block(self,x,i,r,update=True):
        w=self.w;z=rms(x)@w['encoder_attn_qkv_w'][i]
        q,k,v=z.split((2048,256,256),-1);k=rope(k,r)
        if not update:return x,k,v
        q=rope(q,r);a=x+attend(q,k,v)@w['encoder_attn_o_w'][i]
        n=rms(a);f=F.gelu(n@w['encoder_ffn_gate_w'][i],approximate='tanh')*(n@w['encoder_ffn_up_w'][i])
        y=a+f@w['encoder_ffn_down_w'][i]
        return y,k,v

    def encode(self,item,gates=None,skip=(),patch=None,trace=False,start=0,prefix=None):
        x=item['x'];r=item['encoder_rope'];ks=[];vs=[];tr=[]
        if prefix is not None:
            x=prefix['inputs'][start];ks=list(prefix['K'][:start]);vs=list(prefix['V'][:start])
        for i in range(start,18):
            inp=x
            update=(i<17 and i+1 not in skip)
            y,k,v=self.block(x,i,r,update=update)
            if gates is not None and i<17:x=gates[i]*y+(1-gates[i])*x
            else:x=y
            if patch is not None and patch[0]==i+1:x=patch[1](inp)
            ks.append(k);vs.append(v)
            if trace:tr.append((inp,x))
        result=dict(K=ks,V=vs)
        if trace:result.update(inputs=[a for a,b in tr],outputs=[b for a,b in tr])
        return result

    def sample(self,item,c):
        w=self.w;state=item['state']@w['decoder_state_in_proj_w']+w['decoder_state_in_proj_b'];z=item['noise']
        seq=len(item['x']);r=item['decoder_rope']
        mask=torch.ones(51,seq+51,dtype=torch.bool,device=z.device);mask[0,seq+1:]=False
        for t in range(2):
            a=F.silu(z@w['decoder_action_fused_in_proj_w']+self.biases[t])
            x=torch.cat((state.reshape(1,-1),a@w['decoder_action_mlp_w']+w['decoder_action_mlp_b']),0)
            for i in range(18):
                proj=rms(x)@w['decoder_attn_qkv_w'][i];q,k,v=proj.split((2048,256,256),-1)
                q=rope(q,r);k=rope(k,r)
                x=x+attend(q,torch.cat((c['K'][i],k)),torch.cat((c['V'][i],v)),mask)@w['decoder_attn_o_w'][i]
                n=rms(x);h=F.gelu(n@w['decoder_ffn_gate_w'][i],approximate='tanh')*(n@w['decoder_ffn_up_w'][i])
                x=x+h@w['decoder_ffn_down_w'][i]
            z=z+rms(x[1:])@self.outw[t]+self.outb[t]
        return z

    def forward(self,item,gates=None,skip=(),patch=None,prefix=None,start=0):
        return self.sample(item,self.encode(item,gates=gates,skip=skip,patch=patch,prefix=prefix,start=start))

    @staticmethod
    def decode(item,raw):
        # Affine physical action denormalization and current-state delta offset
        # are exported from the actual native output transform, not re-guessed.
        return (raw[:,:7]@item['decode_matrix']+item['decode_offset']).detach().cpu().numpy()
