# python3 cacp.py --arch=resnet20_cifar \
# /home/dataset/cifar \
# --resume=../../ssl/checkpoints/checkpoint_trained_dense.pth.tar \
# --cacp-protocol=mac-constrained \
# --cacp-action-range 0.05 1.0 \
# --cacp-target-density=0.5 \
# --etes=0.075 \
# --cacp-ft-epochs=0 \
# --cacp-prune-pattern=channels \
# --cacp-prune-method=fm-reconstruction \
# --cacp-agent-algo=DDPG \
# --cacp-cfg=auto_compression_channels.yaml \
# --cacp-rllib=hanlab \
# -j=1

import os
import logging
import traceback
from functools import partial
import utils
from environment import CACPWrapperEnvironment, Observation
import apputils as apputils
import apputils.image_classifier as classifier
from rewards import reward_factory


msglogger = logging.getLogger()


class AutoCompressionSampleApp(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)

    def train_auto_compressor(self):
        using_fm_reconstruction = self.args.cacp_prune_method == 'fm-reconstruction'
        fixed_subset, sequential = (using_fm_reconstruction, using_fm_reconstruction)
        msglogger.info("cacp: fixed_subset=%s\tsequential=%s" % (fixed_subset, sequential))
        train_loader, val_loader, test_loader = classifier.load_data(self.args, fixed_subset, sequential)

        self.args.display_confusion = False
        validate_fn = partial(classifier.test, test_loader=val_loader, criterion=self.criterion,
                              loggers=self.pylogger, args=self.args, activations_collectors=None)
        train_fn = partial(classifier.train, train_loader=train_loader, criterion=self.criterion,
                           loggers=self.pylogger, args=self.args)

        save_checkpoint_fn = partial(apputils.save_checkpoint, arch=self.args.arch, dir=msglogger.logdir)
        optimizer_data = {'lr': self.args.lr, 'momentum': self.args.momentum, 'weight_decay': self.args.weight_decay}
        return train_auto_compressor(self.model, self.args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn)


def main():
    import cacp_args
    # Parse arguments
    args = classifier.init_classifier_compression_arg_parser()
    args = cacp_args.add_automl_args(args).parse_args()
    app = AutoCompressionSampleApp(args, script_dir=os.path.dirname(__file__))
    return app.train_auto_compressor()

    
def train_auto_compressor(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn):
    dataset = args.dataset
    arch = args.arch
    num_ft_epochs = args.cacp_ft_epochs
    action_range = args.cacp_action_range
    conditional = args.conditional

    config_verbose(False)

    # Read the experiment configuration
    cacp_cfg_fname = args.cacp_cfg_file
    if not cacp_cfg_fname:
        raise ValueError("You must specify a valid configuration file path using --cacp-cfg")

    with open(cacp_cfg_fname, 'r') as cfg_file:
        compression_cfg = utils.yaml_ordered_load(cfg_file)

    if not args.cacp_rllib:
        raise ValueError("You must set --cacp-rllib to a valid value")

    #rl_lib = compression_cfg["rl_lib"]["name"]
    #msglogger.info("Executing cacp: RL agent - %s   RL library - %s", args.cacp_agent_algo, rl_lib)

    # Create a dictionary of parameters that Coach will handover to WrapperEnvironment
    # Once it creates it.
    services = utils.MutableNamedTuple({
            'validate_fn': validate_fn,
            'save_checkpoint_fn': save_checkpoint_fn,
            'train_fn': train_fn})

    app_args = utils.MutableNamedTuple({
            'dataset': dataset,
            'arch': arch,
            'optimizer_data': optimizer_data,
            'seed': args.seed})

    ddpg_cfg = utils.MutableNamedTuple({
            'heatup_noise': 0.5,
            'initial_training_noise': 0.5,
            'training_noise_decay': 0.95,
            'num_heatup_episodes': args.cacp_heatup_episodes,
            'num_training_episodes': args.cacp_training_episodes,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            "conditional":conditional})

    cacp_cfg = utils.MutableNamedTuple({
            'modules_dict': compression_cfg["network"],  # dict of modules, indexed by arch name
            'save_chkpts': args.cacp_save_chkpts,
            'protocol': args.cacp_protocol,
            'agent_algo': args.cacp_agent_algo,
            'num_ft_epochs': num_ft_epochs,
            'action_range': action_range,
            'reward_frequency': args.cacp_reward_frequency,
            'ft_frequency': args.cacp_ft_frequency,
            'pruning_pattern':  args.cacp_prune_pattern,
            'pruning_method': args.cacp_prune_method,
            'group_size': args.cacp_group_size,
            'n_points_per_fm': args.cacp_fm_reconstruction_n_pts,
            'ddpg_cfg': ddpg_cfg,
            'ranking_noise': args.cacp_ranking_noise,
            "conditional":conditional,
            "support_raito":args.support_ratio})

    cacp_cfg.reward_fn, cacp_cfg.action_constrain_fn = reward_factory(args.cacp_protocol)

    def create_environment():
        env = CACPWrapperEnvironment(model, app_args, cacp_cfg, services)
        env.cacp_cfg.ddpg_cfg.replay_buffer_size = cacp_cfg.ddpg_cfg.num_heatup_episodes * env.steps_per_episode
        return env

    env1 = create_environment()

    from lib.hanlab import hanlab_if
    rl = hanlab_if.RlLibInterface()
    args.observation_len = len(Observation._fields)
    rl.solve(env1, args)

def config_verbose(verbose, display_summaries=False):
    if verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
        logging.getLogger().setLevel(logging.WARNING)
    for module in ["examples.auto_compression.cacp",
                   "apputils.image_classifier",
                   "pruning.thinning",
                   "pruning.ranked_structures_pruner"]:
        logging.getLogger(module).setLevel(loglevel)

    # display training progress summaries
    summaries_lvl = logging.INFO if display_summaries else logging.WARNING
    logging.getLogger("examples.auto_compression.cacp.summaries").setLevel(summaries_lvl)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))