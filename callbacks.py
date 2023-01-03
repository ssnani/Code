import torch 
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing import Any, Optional

from metrics import eval_metrics_batch, eval_metrics_batch_v1, _mag_spec_mask, gettdsignal_mag_spec_mask

class Losscallbacks(Callback):
    def __init__(self):
        self.frame_sie = 320
        self.frame_shift = 160    #TODO

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        ) -> None:
        mix_ri_spec, tgt_ri_spec, doa = batch
        est_ri_spec = outputs['est_ri_spec']

        #adjusting shape, type for istft
        tgt_ri_spec = tgt_ri_spec.to(torch.float32)
        est_ri_spec = est_ri_spec.to(torch.float32)
        mix_ri_spec = mix_ri_spec.to(torch.float32)

        _mix_ri_spec = torch.permute(mix_ri_spec[:,:2],[0,3,2,1])
        mix_sig = torch.istft(_mix_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_mix_ri_spec))

        _tgt_ri_spec = torch.permute(tgt_ri_spec,[0,3,2,1])
        tgt_sig = torch.istft(_tgt_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_tgt_ri_spec))

        _est_ri_spec = torch.permute(est_ri_spec,[0,3,2,1])
        est_sig = torch.istft(_est_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_est_ri_spec))

        trainer.logger.experiment.add_audio(f'mix_{batch_idx}', mix_sig/torch.max(torch.abs(mix_sig)), sample_rate=16000)
        trainer.logger.experiment.add_audio(f'tgt_{batch_idx}', tgt_sig/torch.max(torch.abs(tgt_sig)), sample_rate=16000)
        trainer.logger.experiment.add_audio(f'est_{batch_idx}', est_sig/torch.max(torch.abs(est_sig)), sample_rate=16000)

        #torch.rad2deg(doa[0,:,1])

        #torchaudio.save(f'mix_{batch_idx}.wav', (mix_sig/torch.max(torch.abs(mix_sig))).cpu(), sample_rate=16000)
        #torchaudio.save(f'tgt_{batch_idx}.wav', (tgt_sig/torch.max(torch.abs(tgt_sig))).cpu(), sample_rate=16000)
        #torchaudio.save(f'est_{batch_idx}.wav', (est_sig/torch.max(torch.abs(est_sig))).cpu(), sample_rate=16000)

        mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
        self.log("MIX_SNR", mix_metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("MIX_STOI", mix_metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("MIX_ESTOI", mix_metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("MIX_PESQ_NB", mix_metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("MIX_PESQ_WB", mix_metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)


        _metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), est_sig.cpu().numpy())
        self.log("SNR", _metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("STOI", _metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ESTOI", _metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("PESQ_NB", _metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("PESQ_WB", _metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return


    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        ) -> None:
        mix_ri_spec, tgt_ri_spec, doa = batch
        est_ri_spec = outputs['est_ri_spec']

        #adjusting shape, type for istft
        tgt_ri_spec = tgt_ri_spec.to(torch.float32)
        est_ri_spec = est_ri_spec.to(torch.float32)
        mix_ri_spec = mix_ri_spec.to(torch.float32)

        _mix_ri_spec = torch.permute(mix_ri_spec[:,:2],[0,3,2,1])
        mix_sig = torch.istft(_mix_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_mix_ri_spec))

        _tgt_ri_spec = torch.permute(tgt_ri_spec,[0,3,2,1])
        tgt_sig = torch.istft(_tgt_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_tgt_ri_spec))

        _est_ri_spec = torch.permute(est_ri_spec,[0,3,2,1])
        est_sig = torch.istft(_est_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_est_ri_spec))
        
        #print(f"mix val batch idx: {batch_idx} \n")
        if 0 == pl_module.current_epoch:
            #breakpoint()
            mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
            self.log("MIX_SNR", mix_metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("MIX_STOI", mix_metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("MIX_ESTOI", mix_metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("MIX_PESQ_NB", mix_metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("MIX_PESQ_WB", mix_metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #print(f"est val batch idx: {batch_idx} \n")
        _metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), est_sig.cpu().numpy())
        self.log("VAL_SNR", _metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("VAL_STOI", _metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("VAL_ESTOI", _metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("VAL_PESQ_NB", _metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("VAL_PESQ_WB", _metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return