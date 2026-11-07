"""Exact whole-cotangent validation and explicit sparse VJP interventions."""
import os
import platform
import threading


class ControlledCotangent:
    def __init__(self,model,targets,banks,bundle,*,tiny,path,cotangent,publish):
        import torch
        from models.predictor_cotangent_attribution_v1 import validate_bundle,digest_array,mask_digest
        if type(tiny) is not bool or (tiny and platform.node() in ('linux10603','linux10609')) or (not tiny and platform.node()!='linux10603'):
            raise ValueError('Exact local/live controlled observer domain required.')
        validate_bundle(bundle);self.device='cpu' if tiny else 'cuda';self.features=128 if tiny else 1024
        if bundle['domain']!=('local-tiny-CPU' if tiny else 'target-saved-only') or targets.requires_grad or targets.device.type!=self.device:
            raise ValueError('Saved cotangent/clean-target domain changed.')
        if (list(targets.shape)!=[1,32 if tiny else 2048,self.features] or targets.dtype!=torch.float32
                or any(t.device.type!=self.device for t in (*model.parameters(),*model.buffers()))
                or type(path) is not int or type(cotangent) is not int or path not in (0,1) or cotangent not in (0,1)
                or len(banks)!=8 or any(len(d)!=2 for d in banks)):raise ValueError('Original controlled EOT8 geometry required.')
        masks=[[{'context':p.context.detach().cpu().tolist(),'target':p.target.detach().cpu().tolist()} for p in d] for d in banks]
        if mask_digest(masks)!=bundle['masks_sha256'] or digest_array(targets.detach().cpu().numpy())!=bundle['targets_sha256']:
            raise ValueError('Bound masks or targets changed.')
        if model.predictor._forward_hooks or model.predictor._forward_pre_hooks:raise ValueError('Refuse an already observed predictor.')
        self.model,self.bundle,self.path,self.cotangent,self.publish=model,bundle,path,cotangent,publish
        self.pairs=[p for d in banks for p in d];self.pid=os.getpid();self.creator=threading.get_ident()
        self.rows=[];self.handles=[];self.module_hook=None;self.active=self.closed=self.restored=False;self.lock=threading.RLock()

    def __enter__(self):
        if self.active or self.closed:raise ValueError('Controlled observer is single use.')
        self.active=True
        try:self.module_hook=self.model.predictor.register_forward_hook(self._forward,with_kwargs=True)
        except BaseException:self.active=False;self.closed=True;raise
        return self

    def _forward(self,module,args,kwargs,output):
        import torch
        from models.predictor_eot8_capture_v1 import tensor_info
        from models.predictor_cotangent_attribution_v1 import replay_pair,digest_array
        with self.lock:
            k=len(self.rows)
            if not self.active or k>=16 or module is not self.model.predictor or self.pid!=os.getpid() or self.creator!=threading.get_ident():raise ValueError('Unowned/extra controlled forward observation.')
            pair,expected=self.pairs[k],self.bundle['pairs'][k]
            if (len(args)!=3 or kwargs!={'mask_index':0} or not torch.equal(args[1],pair.context) or not torch.equal(args[2],pair.target)
                    or output.device.type!=self.device or output.dtype!=torch.float32 or not output.requires_grad or list(output.shape)!=expected['shape']):
                raise ValueError('Native predictor order/mask/device/geometry changed.')
            prediction=tensor_info(output)
            if prediction['sha256']!=expected['arms'][self.path]['prediction_sha256']:raise ValueError('Same-path prediction did not replay its saved bytes.')
            row={'index':k,'draw':k//2,'family':k%2,'pid':self.pid,'forward_thread':self.creator,'cotangent_thread':None,
                 'prediction':prediction,'received':None,'applied':None,'sparse_entries':[],'replacement':self.path!=self.cotangent,'hook_removed':False}
            self.rows.append(row);self.publish()
            def incoming(gradient):
                with self.lock:
                    if (not self.active or self.pid!=os.getpid() or row['received'] is not None or gradient.device.type!=self.device
                            or gradient.dtype!=torch.float32 or gradient.requires_grad):raise ValueError('Repeated/unowned native incoming cotangent.')
                    row['received']=tensor_info(gradient);row['cotangent_thread']=threading.get_ident();self.publish()
                    native=gradient.detach().cpu().contiguous().numpy()
                    patched=replay_pair(native,expected,self.path,self.cotangent)
                    # The copy is deliberate: the received native tensor is never modified.
                    applied=gradient if self.path==self.cotangent else torch.from_numpy(patched).to(device=gradient.device)
                    row['applied']=tensor_info(applied)
                    bits=native.view('uint32');after=applied.detach().cpu().contiguous().numpy().view('uint32')
                    for edit in expected['edits']:
                        loc=tuple(edit['coordinate'])
                        row['sparse_entries'].append({'coordinate':edit['coordinate'],'received_bits':f'{int(bits[loc]):08x}','applied_bits':f'{int(after[loc]):08x}'})
                    if (digest_array(gradient.detach().cpu().contiguous().numpy())!=expected['arms'][self.path]['cotangent_sha256']
                            or row['applied']['sha256']!=expected['arms'][self.cotangent]['cotangent_sha256']):raise ValueError('Native mutation or replayed cotangent changed.')
                    self.publish()
                    return None if self.path==self.cotangent else applied
            self.handles.append(output.register_hook(incoming))
        return None

    def __exit__(self,typ,error,tb):
        failures=[]
        for handle in ([self.module_hook] if self.module_hook is not None else [])+self.handles:
            try:handle.remove()
            except BaseException as exc:failures.append(exc)
        self.active=False;self.closed=True
        self.restored=not failures and not self.model.predictor._forward_hooks and not self.model.predictor._forward_pre_hooks
        for row in self.rows:row['hook_removed']=self.restored
        if typ is None:
            self.publish()
            if not self.restored or len(self.rows)!=16 or any(r['received'] is None or r['applied'] is None for r in self.rows):raise ValueError('Incomplete controlled observation/restoration.')
        else:
            try:self.publish()
            except BaseException:pass
        return False
