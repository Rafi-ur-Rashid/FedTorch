# -*- coding: utf-8 -*-
import os
import time
import logging

from fedtorch.utils.op_files import write_txt


def record(content, path):
    write_txt(content + "\n", path, type="a")


log_path = None


def configure_log(cfg=None):
    global log_path

    if cfg is not None:
        log_path = os.path.join(
            cfg.checkpoint.checkpoint_dir, 'record' + str(cfg.graph.rank))
    else:
        log_path = os.path.join(os.getcwd(), "record")


def log(content, debug=True):
    """print the content while store the information to the path."""
    content = time.strftime("%Y:%m:%d %H:%M:%S") + "\t" + content
    if debug:
        print(content)
        write_txt(content + "\n", log_path, type="a")


def setup_logging(log_file='log.txt'):
    """Setup logging configuration."""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging


def log_cfgs(cfg, debug=True):
    log('parameters: ', debug=debug)
    for arg in vars(cfg):
        log(str(arg) + '\t' + str(getattr(cfg, arg)), debug=debug)
    for name in ['n_nodes', 'world', 'rank',
                 'ranks_with_blocks', 'blocks_with_ranks',
                 'device', 'device_type', 'get_neighborhood']:
        log('{}: {}'.format(name, getattr(cfg.graph, name)), debug=debug)


def logging_computing(tracker, loss, performance, _input, lr):
    # measure accuracy and record loss.
    tracker = update_performancec_tracker(tracker, loss, performance, _input.size(0))

    # measure elapsed time.
    tracker['computing_time'].update(time.time() - tracker['end_data_time'])
    tracker['start_sync_time'] = time.time()
    tracker['learning_rate'].update(lr)


def logging_sync_time(tracker):
    # measure elapsed time.
    tracker['sync_time'].update(time.time() - tracker['start_sync_time'])


def logging_load_time(tracker):
    # measure elapsed time.
    tracker['load_time'].update(time.time() - tracker['start_load_time'])


def logging_globally(tracker, start_global_time):
    tracker['global_time'].update(time.time() - start_global_time)


def logging_display_training(cfg, tracker):
    log_info = 'Epoch: {epoch:.3f}. Local index: {local_index}. Load: {load:.3f}s | Data: {data:.3f}s | Computing: {computing_time:.3f}s | Sync: {sync_time:.3f}s | Global: {global_time:.3f}s | Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f} | learning_rate: {lr:.4f} | rounds_comm: {rounds_comm}'.format(
        epoch=cfg.epoch_,
        local_index=cfg.local_index,
        load=tracker['load_time'].avg,
        data=tracker['data_time'].avg,
        computing_time=tracker['computing_time'].avg,
        sync_time=tracker['sync_time'].avg,
        global_time=tracker['global_time'].avg,
        loss=tracker['losses'].avg,
        top1=tracker['top1'].avg,
        top5=tracker['top5'].avg,
        lr=tracker['learning_rate'].val,
        rounds_comm=cfg.rounds_comm)
    log('Process {}: '.format(cfg.graph.rank) + log_info, debug=cfg.graph.debug)

def logging_display_val(cfg,performance, mode, personal=False):
    #TODO:improve this method
    if mode == 'test':
        if personal:
            log('Test at personal model at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
                cfg.local_index, cfg.epoch, cfg.graph.rank, performance[0], performance[1], performance[2], cfg.rounds_comm),
                debug=cfg.graph.debug)
        else:
            log('Test at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
                cfg.local_index, cfg.epoch, cfg.graph.rank, performance[0], performance[1], performance[2], cfg.rounds_comm),
                debug=cfg.graph.debug)
    else:
        pretext = []
        pretext.append('Personal' if personal else 'Global')
        pretext.append('validation' if mode=='validation' else 'train')

        log('{} performance for {} at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
            pretext[0], pretext[1], cfg.local_index, cfg.epoch, cfg.graph.rank, performance[0], performance[1], performance[2], cfg.rounds_comm),
            debug=cfg.graph.debug)
    
    if cfg.graph.rank == 0:
        model_mode = 'Local' if personal else 'Global'
        cfg.tb_writer.add_scalar('{}/{}/Loss'.format(model_mode, mode), performance[2], cfg.rounds_comm)
        cfg.tb_writer.add_scalar('{}/{}/Acc'.format(model_mode, mode), performance[0], cfg.rounds_comm)


def logging_display_test_summary(cfg,debug=True):
    log('best accuracy for rank {} at local index {} \
        (best epoch {:.3f}, current epoch {:.3f}): {}.'.format(
        cfg.graph.rank, cfg.local_index,
        cfg.best_epoch[-1] if len(cfg.best_epoch) != 0 else 0.0,
        cfg.epoch_, cfg.best_prec1), debug=debug)


def update_performancec_tracker(tracker, loss, performance, size):
    tracker['losses'].update(loss.item(), size)

    if len(performance) == 2:
        tracker['top5'].update(performance[1], size)
    tracker['top1'].update(performance[0], size)
    return tracker

def update_performance_per_class(tracker,acc,count,classes):
    for a,n,c in zip(acc, count, classes):
        tracker[c.item()].update(a.item(),n.item())
    return tracker