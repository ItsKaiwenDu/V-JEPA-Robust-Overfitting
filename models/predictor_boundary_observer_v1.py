"""Bounded native output/cotangent copies; hooks return None and change no tensor."""
import os
from pathlib import Path
import platform
import sys
import threading


class BoundaryObserver:
    def __init__(self,model,banks,folder,*,tiny,publish):
        import torch
        if type(tiny) is not bool or (tiny and platform.node() in ('linux10603','linux10609')) or (not tiny and platform.node()!='linux10603'):
            raise ValueError('Exact observer domain required.')
        self.device='cpu' if tiny else 'cuda'
        self.features=128 if tiny else 1024
        if len(banks)!=8 or any(len(draw)!=2 for draw in banks):raise ValueError('Original eight-draw/two-family bank required.')
        if any(t.device.type!=self.device for t in (*model.parameters(),*model.buffers())):raise ValueError('Observer device changed.')
        self.model,self.folder,self.publish=model,Path(folder),publish
        self.pairs=[pair for draw in banks for pair in draw]
        self.rows=[];self.active=False;self.closed=False;self.handles=[];self.handle=None
        self.restored=False;self.lock=threading.RLock();self.creator=threading.get_ident();self.pid=os.getpid()
        if model.predictor._forward_hooks or model.predictor._forward_pre_hooks:raise ValueError('Refuse an already observed predictor.')

    def __enter__(self):
        if self.active or self.closed:raise ValueError('Observer is single use.')
        self.active=True
        try:self.handle=self.model.predictor.register_forward_hook(self._forward,with_kwargs=True)
        except BaseException:self.active=False;self.closed=True;raise
        return self

    def _forward(self,module,args,kwargs,output):
        import torch
        from models.predictor_eot8_capture_v1 import save_return
        with self.lock:
            i=len(self.rows)
            if not self.active or i>=16 or os.getpid()!=self.pid or threading.get_ident()!=self.creator or module is not self.model.predictor:raise ValueError('Unowned or extra predictor observation.')
            pair=self.pairs[i]
            if len(args)!=3 or kwargs!={'mask_index':0} or not torch.equal(args[1],pair.context) or not torch.equal(args[2],pair.target):raise ValueError('Native prediction order/masks changed.')
            shape=[1,pair.target.shape[1],self.features]
            name=f'draw-{i//2:02d}-family-{i%2}'
            item=save_return(self.folder/(name+'-prediction.pt'),output,'input',shape)
            row={'index':i,'draw':i//2,'family':i%2,'prediction':item,'cotangent':None,
                 'forward_thread':self.creator,'cotangent_thread':None,'pid':self.pid,'hook_removed':False}
            self.rows.append(row);self.publish()
            if output.device.type!=self.device or not output.requires_grad:raise ValueError('Native observation lost its autograd/device domain.')
            def incoming(gradient):
                with self.lock:
                    if not self.active or os.getpid()!=self.pid or row['cotangent'] is not None:raise ValueError('Repeated/unowned native cotangent.')
                    # Retain actual returned values before finite/numerical gates. No cast.
                    row['cotangent']=save_return(self.folder/(name+'-cotangent.pt'),gradient,'input',shape)
                    row['cotangent_thread']=threading.get_ident();self.publish()
                    if gradient.device.type!=self.device or gradient.requires_grad:raise ValueError('First-order cotangent device/autograd drift.')
                return None
            self.handles.append(output.register_hook(incoming))
        return None

    def __exit__(self,typ,error,tb):
        try:
            if self.handle is not None:self.handle.remove()
            for handle in self.handles:handle.remove()
            for row in self.rows:row['hook_removed']=True
        finally:
            self.active=False;self.closed=True
            self.restored=not self.model.predictor._forward_hooks and not self.model.predictor._forward_pre_hooks
        if typ is None:self.publish()
        else:
            try:self.publish()
            except BaseException as secondary:print('Secondary boundary publication failed: '+type(secondary).__name__,file=sys.stderr,flush=True)
        if typ is None and (not self.restored or len(self.rows)!=16 or any(r['cotangent'] is None for r in self.rows)):
            raise ValueError('All native predictions/cotangents and removed hooks required.')
        return False
