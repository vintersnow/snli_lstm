from torch import nn
from torchutils import get_logger, DEBUG, Runner
from torchutils.misc import OneLinePrint, Timer
from torchutils.utils import optimzier
from dataset import next_batch
import sklearn.metrics

logger = get_logger(__name__, DEBUG)


class SentRunner(Runner):
    def __init__(self, model, vocab, use_cuda):
        criterion = nn.NLLLoss()
        self.model = model
        self.vocab_size = vocab.size
        self.use_cuda = use_cuda
        super(SentRunner, self).__init__(criterion)

    def run(self, batch):
        keys = ('s1', 's1_len', 's1_mask', 's2', 's2_len')
        outputs = self.model(*self.getvars(batch, *keys))  # (B*2)
        return outputs

    def target(self, batch):
        tgt = self.getvars(batch, 'label').view(-1)
        return tgt


def train(model, vocab, train_loader, val_loader, hps):
    '''
    Args:
        model (torchutils.Model)
        vocab (Vocab)
        train_loader (torch.utils.data.dataloader)
        val_loader (torch.utils.data.dataloader)
    '''
    olp = OneLinePrint()
    timer = Timer()

    # remove parameters if requires_grad == False
    model_params = list(
        filter(lambda p: p.requires_grad, model.model.parameters()))
    model.addopt(optimzier(hps.opt, model_params, lr=hps.init_lr))

    if hps.restore:
        init_step, ckpt_name = model.restore(hps.restore)
        logger.info('Restored from %s' % ckpt_name)
    else:
        init_step = hps.start_step

    runner = SentRunner(model, vocab, hps.use_cuda)

    # for store summary
    if hps.store_summary:
        writer = model.make_writer()

    t_batcher = next_batch(train_loader)

    logger.info('----Start training: %s----' % model.name)
    timer.start()
    loss_sum = 0
    for step in range(init_step, hps.num_iters + 1):
        model.train()

        model.opt.zero_grad()
        batch = next(t_batcher)
        loss, _, _ = runner.step(batch)
        loss.backward()

        global_norm = nn.utils.clip_grad_norm(model_params, hps.clip)
        model.opt.step()
        loss_sum += loss.data[0]

        olp.write('step %s train loss: %f' % (step, loss.data[0]))

        # save checkpoint
        if step % hps.ckpt_steps == 0:
            model.save(step, loss.data[0])
            olp.write('save checkpoint (step=%d)\n' % step)
        olp.flush()

        # store summary
        if hps.store_summary and (step - 1) % hps.summary_steps == 0:
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('global_norm', global_norm, step)
            # average time
            if step - 1 != 0:
                lap_time, _ = timer.lap('summary')
                steps = hps.summary_steps
                writer.add_scalar('avg time/step', lap_time / steps, step)

        if step % hps.check_steps == 0:
            logger.info('\nstep:%d~%d avg loss: %f', step - hps.check_steps,
                        step, loss_sum / hps.check_steps)
            loss_sum = 0

            # validation
            model.eval()
            preds = []
            tgts = []
            for v_batch in val_loader:
                v_outputs = runner.run(v_batch)
                _, pred = v_outputs.max(1)
                pred = pred.cpu().data.tolist()
                preds.extend(pred)
                tgts.extend(v_batch['label'])
                if len(preds) > hps.val_num:
                    break

            assert len(preds) == len(tgts)
            f1 = sklearn.metrics.f1_score(tgts, preds, average='macro')
            precision = sklearn.metrics.precision_score(
                tgts, preds, average='macro')
            recall = sklearn.metrics.recall_score(tgts, preds, average='macro')

            if f1 is None:
                continue
            if hps.store_summary:
                writer.add_scalar('F1', f1, step)
                writer.add_scalar('Precision', precision, step)
                writer.add_scalar('Recall', recall, step)

            logger.info('F1: %.3f, P: %.3f, R: %.3f' % (f1, precision, recall))

    if hps.store_summary:
        writer.close()


def test(model, vocab, loader, hps):
    olp = OneLinePrint()
    model.eval()
    preds = []
    tgts = []
    runner = SentRunner(model, vocab, hps.use_cuda)

    if hps.restore:
        _, ckpt_name = model.restore(hps.restore)
        logger.info('Restored from %s' % ckpt_name)

    logger.info('----Start testing: %s----' % model.name)
    for batch in loader:
        outputs = runner.run(batch)
        _, pred = outputs.max(1)
        pred = pred.cpu().data.tolist()
        preds.extend(pred)
        tgts.extend(batch['label'])
        olp.write('Num: %s' % len(preds)).flush()

    assert len(preds) == len(tgts)
    f1 = sklearn.metrics.f1_score(tgts, preds, average='macro')
    precision = sklearn.metrics.precision_score(tgts, preds, average='macro')
    recall = sklearn.metrics.recall_score(tgts, preds, average='macro')

    logger.info('\nF1: %.3f, P: %.3f, R: %.3f' % (f1, precision, recall))
