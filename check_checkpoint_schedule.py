"""Full BERT numerical/memory check before considering adaptive checkpointing.

Run alone on the GPU after a completed epoch/fold. This does not save trained
weights or accuracy results, and does not change the active experiment.
"""
import argparse
import gc
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from common import ROOT, ESC_DATA, batch, load_features, save_json
from adapted_model import ReproductionModel
from legacy_utils import setup_seed


def rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def set_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def same_rng(a,b):
    return torch.equal(a[0],b[0]) and all(torch.equal(x,y) for x,y in zip(a[1],b[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-limit',type=int,default=2048)
    parser.add_argument('--esc-checkpoint',type=Path)
    cli = parser.parse_args()
    torch.set_num_threads(4)
    setup_seed(209)
    device = torch.device('cuda')
    results = []
    for dataset in ['esc','maven']:
        args = SimpleNamespace(device=device,model_name=str(ROOT/'models/bert-base-uncased'),
            n_last=768,mlp_size=2304,mlp_drop=.4,no_of_classes=3,rate=0.,w=.6,
            max_iteration=9 if dataset=='esc' else 19,threshold=2,min_iteration=2,
            in_channels={'event':768},metadata=(['event'],[('event','intra','event'),('event','inter','event')]),
            num_heads=4,beta_intra=.3,beta_inter=.7,gamma=2.,loss_type='focal',
            checkpoint_mlp=dataset=='maven',resize_vocab_size=30524 if dataset=='maven' else None)
        model = ReproductionModel(args).to(device)
        model.CGE.incoming_orientation = True
        if dataset=='esc' and cli.esc_checkpoint:
            model.load_state_dict(torch.load(cli.esc_checkpoint,map_location='cpu'))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5)
        if dataset=='esc':
            pool = sum([load_features(ESC_DATA/('EventStoryLine_bert_intra_and_inter_%s_features.pkl'%s))
                        for s in ['train','dev']],[])
        else:
            pool = sum([load_features(ROOT/('data/maven/%s.pkl'%s)) for s in ['train','dev','test']],[])
        candidates = [f for f in pool if f.enc_input_ids.numel()<=cli.token_limit]
        selected = {str(f.doc_id):f for f in [max(candidates,key=lambda f:f.enc_input_ids.numel()),
                    max(candidates,key=lambda f:len(f.target)),min(candidates,key=lambda f:f.enc_input_ids.numel())]}
        for doc_id,feature in selected.items():
            inputs = batch(feature,device)
            model.zero_grad(set_to_none=True)
            initial_rng = rng_state()
            model.pretrained_model.gradient_checkpointing_enable()
            torch.cuda.synchronize()
            start = time.monotonic()
            loss_a,logits_a = model(**inputs)
            loss_a.backward()
            torch.cuda.synchronize()
            checkpoint_seconds = time.monotonic()-start
            rng_a = rng_state()
            gradients = {name:None if p.grad is None else p.grad.detach().cpu()
                         for name,p in model.named_parameters()}
            logits_cpu,loss_cpu = logits_a.detach().cpu(),loss_a.detach().cpu()
            del loss_a,logits_a
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            set_rng(initial_rng)
            model.pretrained_model.gradient_checkpointing_disable()
            torch.cuda.synchronize()
            start = time.monotonic()
            loss_b,logits_b = model(**inputs)
            loss_b.backward()
            torch.cuda.synchronize()
            direct_seconds = time.monotonic()-start
            equal_forward = torch.equal(logits_cpu,logits_b.detach().cpu()) and torch.equal(loss_cpu,loss_b.detach().cpu())
            equal_gradients,max_difference = True,0.
            for name,p in model.named_parameters():
                expected = gradients[name]
                if expected is None or p.grad is None:
                    equal_gradients &= expected is None and p.grad is None
                else:
                    actual = p.grad.detach().cpu()
                    equal_gradients &= torch.equal(expected,actual)
                    max_difference = max(max_difference,float((expected-actual).abs().max()))
            equal_rng = same_rng(rng_a,rng_state())
            del gradients,loss_b,logits_b
            # Account for persistent Adam moments and existing gradient buffers.
            optimizer.step()
            model.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loss,logits = model(**inputs)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            record = dict(dataset=dataset,doc_id=doc_id,input_shape=list(feature.enc_input_ids.shape),
                pairs=len(feature.target),equal_forward=equal_forward,equal_gradients=equal_gradients,
                equal_rng=equal_rng,max_gradient_difference=max_difference,
                checkpoint_seconds=checkpoint_seconds,direct_seconds=direct_seconds,
                direct_peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            results.append(record)
            print(record,flush=True)
            del loss,logits,inputs
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        del model,optimizer
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(token_limit=cli.token_limit,bitwise_equal=all(
        r['equal_forward'] and r['equal_gradients'] and r['equal_rng'] for r in results),cases=results)
    save_json(ROOT/'artifacts/full_bert_checkpoint_schedule_check.json',report)
    print('BITWISE_EQUAL',report['bitwise_equal'],flush=True)


if __name__=='__main__':
    main()
