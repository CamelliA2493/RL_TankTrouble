"""403维原版适配编码与GRU策略。层序和Beta参数化为本项目实现。"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta,Categorical
from torch.nn import functional as F


OBS_DIM=403
WALL_SLICE=slice(108,333)
HIDDEN_SIZE=128
GRU_SIZE=32
GROUPS=dict(self_state=[0,12],enemies=[12,60],projectiles=[60,76],items=[76,108],
            local_walls=[108,333],radar=[333,373],global_state=[373,385],
            weapon_state=[385,397],context=[397,403])


def encode_observation(obs,context=None):
    """公开原版状态转403维。空间以格为单位，墙图是半格分辨率的15×15窗口。"""
    tanks=np.asarray(obs["tanks"]).reshape(2,32)
    own,enemy=tanks
    entities=np.asarray(obs["entities"]).reshape(-1,16)
    width=int(round(float(obs["global"][1])*12))
    height=int(round(float(obs["global"][2])*10))
    grid=np.asarray(obs["map"]).reshape(3,11,13)
    result=np.zeros(OBS_DIM,dtype=np.float32)
    result[:12]=own[[0,1,2,3,4,5,6,14,15,13,16,17]]
    result[12:18]=enemy[:6]  # 其余7个敌方槽为空，alive字段同时提供有效性。
    positions=entities[:,2:4]*[12,10]
    delta=positions-own[1:3]*[12,10]
    distance2=np.square(delta).sum(1)
    order=np.argsort(distance2,kind="stable")
    projectile_ids=[i for i in order if entities[i,0]>.5 and entities[i,1]<9/13-1e-5]
    for slot,index in enumerate(projectile_ids[:2]):
        e=entities[index];dx,dy=delta[index];vx,vy=e[6:8]*7
        result[60+slot*8:68+slot*8]=[1,dx/4,dy/4,(vx*own[4]-vy*own[3])/.1,
            (vx*own[3]+vy*own[4])/.1,e[13],e[1],e[8]]
    item_ids=[i for i in order if entities[i,0]>.5 and abs(entities[i,1]-9/13)<1e-5]
    for slot,index in enumerate(item_ids[:4]):
        e=entities[index]
        result[76+slot*8:84+slot*8]=[1,*list(delta[index]/4),e[4],e[5],e[14],e[15],e[8]]
    raster=np.zeros((2*height+1,2*width+1),dtype=np.float32)
    raster[::2,1::2]=grid[1,:height+1,:width]
    raster[1::2,::2]=grid[2,:height,:width+1]
    for y in range(0,2*height+1,2):
        for x in range(0,2*width+1,2):
            raster[y,x]=max(raster[max(y-1,0):min(y+2,2*height+1),x].max(),
                            raster[y,max(x-1,0):min(x+2,2*width+1)].max())
    cx,cy=np.floor(own[1:3]*[24,20]).astype(int)
    patch=np.ones((15,15),dtype=np.float32)
    for py in range(15):
        for px in range(15):
            x,y=cx+px-7,cy+py-7
            if 0<=x<raster.shape[1] and 0<=y<raster.shape[0]:
                patch[py,px]=raster[y,x]
    result[WALL_SLICE]=patch.ravel()
    radar=np.zeros((8,5),dtype=np.float32)
    for index in projectile_ids:
        dx,dy=delta[index];vx,vy=entities[index,6:8]*7
        forward=dx*own[4]-dy*own[3];side=dx*own[3]+dy*own[4]
        sector=int(np.floor((np.arctan2(side,forward)+np.pi)/(2*np.pi)*8))%8
        if radar[sector,0]:
            continue
        speed2=vx*vx+vy*vy
        time=float(np.clip(-(dx*vx+dy*vy)/max(speed2,1e-12),0,32))
        closest=float(np.hypot(dx+vx*time,dy+vy*time))
        radar[sector]=[1,np.sqrt(distance2[index])/8,time/32,closest/4,entities[index,13]]
    result[333:373]=radar.ravel()
    result[373:385]=np.asarray(obs["global"])[:12]
    result[385:397]=np.r_[own[18:28],own[7],own[6]]
    if context is not None:
        assert np.asarray(context).shape==(6,)
        result[397:403]=context
    assert np.isfinite(result).all()
    return np.clip(result,-5,5)


class HybridPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),
            nn.Conv2d(8,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,3)),nn.Flatten())
        self.state_encoder=nn.Sequential(nn.Linear(178,64),nn.Tanh())
        self.fusion=nn.Sequential(nn.Linear(72+64,HIDDEN_SIZE),nn.Tanh())
        self.gru=nn.GRU(HIDDEN_SIZE,GRU_SIZE,batch_first=True)
        self.output=nn.Linear(GRU_SIZE,17)
        self.log_concentration=nn.Parameter(torch.tensor(1.))

    def forward(self,observations,state=None,episode_starts=None):
        # (B,T,403) -> (B,T,17)，每个高层决策推进一次GRU。
        batch,length,dimension=observations.shape
        assert dimension==OBS_DIM
        flat=observations.reshape(batch*length,dimension)
        maps=self.map_encoder(flat[:,WALL_SLICE].reshape(-1,1,15,15))
        scalars=torch.cat((flat[:,:108],flat[:,333:]),dim=-1)
        features=self.fusion(torch.cat((maps,self.state_encoder(scalars)),dim=-1)).reshape(batch,length,-1)
        if state is None:
            state=observations.new_zeros(1,batch,GRU_SIZE)
        outputs=[]
        for step in range(length):
            if episode_starts is not None:
                state=state*(~episode_starts[:,step].bool()).to(state.dtype).reshape(1,batch,1)
            output,state=self.gru(features[:,step:step+1],state)
            outputs.append(output)
        return self.output(torch.cat(outputs,dim=1)),state

    def distributions(self,outputs):
        template=Categorical(logits=outputs[...,:7])
        # 单Laika只有目标槽0有效，固定掩码同时用于采样和PPO重算。
        target_logits=torch.full_like(outputs[...,7:15],-torch.inf)
        target_logits[...,0]=outputs[...,7]
        target=Categorical(logits=target_logits)
        mean=outputs[...,15].sigmoid()
        concentration=F.softplus(self.log_concentration)+2
        aggression=Beta(1+mean*concentration,1+(1-mean)*concentration)
        return template,target,aggression

    def actions(self,outputs,deterministic=False):
        template,target,aggression=self.distributions(outputs)
        if deterministic:
            actions=torch.stack((template.logits.argmax(-1).float(),target.logits.argmax(-1).float(),aggression.mean),dim=-1)
        else:
            actions=torch.stack((template.sample().float(),target.sample().float(),aggression.sample()),dim=-1)
        log_prob,entropy=self.evaluate_actions(outputs,actions)
        return actions,log_prob,entropy,outputs[...,16]

    def evaluate_actions(self,outputs,actions):
        template,target,aggression=self.distributions(outputs)
        log_prob=(template.log_prob(actions[...,0].long())+target.log_prob(actions[...,1].long())
                  +aggression.log_prob(actions[...,2].clamp(1e-6,1-1e-6)))
        entropy=template.entropy()+target.entropy()+aggression.entropy()
        return log_prob,entropy
