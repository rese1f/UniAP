
import os
import pytorch_lightning as pl
import torch
import warnings

from train.train_utils import configure_experiment, load_model



def run(config):
    # set monitor name and postfix
    if config.stage == 0:
        config.monitor = 'summary/mtrain_valid_pred'
    else:
        config.monitor = f'mtest_support/{config.task}_pred'
        if config.save_postfix == '':
            config.save_postfix = f'_task:{config.task}{config.save_postfix}'

    # load model
    model, ckpt_path = load_model(config, verbose=IS_RANK_ZERO)

    # environmental settings
    logger, log_dir, callbacks, precision, strategy, plugins = configure_experiment(config, model)
    if config.stage == 2:
        model.config.result_dir = log_dir

    # create pytorch lightning trainer.
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=log_dir,
        accelerator='gpu',
        # max_epochs=((config.n_steps // config.val_iter) if (not config.no_eval) and config.stage <= 1 else 1),
        max_epochs=10000,
        log_every_n_steps=-1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        benchmark=True,
        devices=config.n_gpus,
        # devices=[7],
        strategy=strategy,
        precision=precision,
        plugins=plugins,

    )

    # validation at start
    if config.stage == 1:
        trainer.validate(model, verbose=False)
    # start training or fine-tuning
    if config.stage <= 1:
        trainer.fit(model, ckpt_path=ckpt_path)
    # start evaluation
    else:
        trainer.test(model)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    from args import config # parse arguments

    run(config)

