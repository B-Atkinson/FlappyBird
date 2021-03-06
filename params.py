# Copyright 2020, Prof. Marko Orescanin, NPS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  
# Created by marko.orescanin@nps.edu on 7/21/20
#
# Modified by Brian Atkinson on 1/5/21
# This module contains all parameter handling for the project, including
# all command line tunable parameter definitions, any preprocessing
# of those parameters, or generation/specification of other parameters.


import argparse
import os
# import datetime
import yaml


def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run training for FlappyBird reinforcement learning with human influence.')
    
    #render the screen
    # parser.add_argument('--render', type=str2bool, default=True,
    #                     help='use this to test in Wing IDE')
    parser.add_argument('--render', type=str2bool, default=False,
                        help='if True, the game will be displayed')    
    #network arguments
    parser.add_argument('--hidden', type=int, default=200,
                        help="the number of hidden nodes to use in the network")
    
    #hyperparameters
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--dropout', type=float, default=0,
                        help="percentage of hidden layer neurons to be dropped from the network each episode")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="specify the base learning rate for the model")
    parser.add_argument('--seed', type=int, default=88,
                        help="specify a number to seed the PRNGs with")
    parser.add_argument("--decay_rate", type=float, default=.99,
                        help="rate of decay for RMSprop")
    parser.add_argument('--batch_size', type=int, default=10,
                        help="number of episodes to conduct rmsprop parameter updates over")
    parser.add_argument('--normalize', type=str2bool, default=False,
                        help='if True, the network values get normalized each time they are recalculated')
    parser.add_argument('--pipe_reward', type=float, default=1.0)
    parser.add_argument('--loss_reward', type=float, default=-5.0)
    parser.add_argument('--flip_heuristic', type=str2bool, default=False,
                        help='if True, the actions recommended by the human heuristic are inverted where \
                                 y_agent>y_bottom_gap results in a NOOP recommendation')
    parser.add_argument('--bias', type=float, default=0,
                        help='value of bias neurons to be added to result of dot product of input with weights')
    parser.add_argument('--init', type=str, default='Xavier',\
                        help="weight initialization to use")
    parser.add_argument('--leaky', type=str2bool, default=False,\
                        help="if True, use Leaky ReLu activation, else use ReLU")
    
    #training arguments
    parser.add_argument('--percent_hybrid', type=float, default=1,
                        help="percentage of the previous frame to subtract from the current frame in the calculation of the hybrid.")
    parser.add_argument('--gap_size', type=float, default=1,
                        help="percent to increase or decrease standard gap size. training gap will be 100*gap_size")
    parser.add_argument('--human', type=str2bool, default=True,
                        help="determines if human influence is to be used in training the agent")
    parser.add_argument('--human_influence', type=float, default=.4,
                        help="determines if human influence is to be used in training the agent")
    parser.add_argument('--human_decay', type=float, default=0,
                        help="rate of exponential decay for human influence per episode")
    parser.add_argument('--num_episodes', type=int, default=20,
                        help="the number of episodes to train the agent on")
    parser.add_argument('--save_stats', type=int, default=100,
                        help="specifies the number of episodes to wait until saving network parameters, training summaries, and moves")
    parser.add_argument('--hidden_save_rate', type=int, default=200,
                        help='saves the hidden layer activations every X episodes')
    
    #load from checkpoint
    parser.add_argument('--continue_training', type=str2bool, default=False,
                        help="continue training from a checkpoint")
    parser.add_argument('--checkpoint_path', type=str,
                        help="path to the parent directory of the training session to resume")
    parser.add_argument('--train_start', type=int, default=100,
                        help="number of pickle file to load model weights from")
    
    #filepath arguments
    parser.add_argument('--output_dir', type=str,
                        help='a filepath to an existing directory to save to')
    
    # multi-gpu training arguments
    parser.add_argument('--mgpu_run', type=str2bool, default=False,
                        help="multi gpu run")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="number of gpu's on the machine, juno is 2")
    
    # multi-processing arguments
    parser.add_argument('--use_multiprocessing', type=str2bool, default=True,
                        help="specifys weather to use use_multiprocessing in .fit_genrator method ")
    parser.add_argument("--workers", type=int, default=6,
                        help="number of CPU's, for my machine 6 workers, for Juno 18")
    
    return parser.parse_args()


# you have to use str2bool
# because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_hparams():
    """any preprocessing, special handling of the hparams object"""

    parser = make_argparser()

    return parser


def save_hparams(hparams):
    path_ = os.path.join(hparams.model_dir, 'params.txt')
    hparams_ = vars(hparams)
    with open(path_, 'w') as f:
        for arg in hparams_:
            print(arg, ':', hparams_[arg])
            f.write(arg + ':' + str(hparams_[arg]) + '\n')

    path_ = os.path.join(hparams.model_dir, 'params.yml')
    with open(path_, 'w') as f:
        yaml.dump(hparams_, f,
                  default_flow_style=False)  # save hparams as a yaml, since that's easier to read and use
