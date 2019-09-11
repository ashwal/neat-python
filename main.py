"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import importlib  

import neat
import visualize


import gym
import gym_evolve
import pygame

import numpy as np



def eval_enviroment(genomes, config):
    env = gym.make('evolve-v0')
    for genome_id, genome in genomes:
        
        # genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        observation = env.reset()

        reward = 0
        investment = 0
        for _ in range(300):
            #env.render()
            output = net.activate(observation + [0])
            action = np.argmax(output)

            if action == 4:
                investment += 1

            # print("Current observation: " + str(observation))
            # print("Action chosen: " + str(action))

            observation, step_reward, done, info = env.step(action)
            reward += step_reward

        genome.fitness = reward
        genome.ornament = investment

        #print(genome.ornament)

    env.close()

def train(config, filename_prefix, max_generations=1):
    p = neat.Population(config)
   
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)


    #Are stats saved here?
    p.add_reporter(neat.Checkpointer(filename_prefix=filename_prefix,generation_interval=10, total_generations=max_generations))


    best_genome = p.run(eval_enviroment, max_generations)
    #print(stats.most_fit_genomes)
    stats.save(prefix_path=filename_prefix)

    #TODO: add pickeling for stats
    

    visualize_genome(config, best_genome, stats)
 
def visualize_genome(config, genome, stats):
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True) 
    visualize.draw_net(config, genome, True)

def run_genome(config, )    
def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None, step_max=None):
    
    env_done = False
    running = True

    video_size = (600, 400)
    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()
    env.reset()

    current_steps = 0

    while running:
        
        if step_max and current_steps == step_max:
            running = False

        # if env_done:
        #     env_done = False
        #     obs = env.reset()
        # else:
        #     action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
        #     prev_obs = obs
        #     obs, rew, env_done, info = env.step(action)
        #     if callback is not None:
        #         callback(prev_obs, obs, action, rew, env_done, info)
        # if obs is not None:
        #     if len(obs.shape) == 2:
        #         obs = obs[:, :, None]
        #     if obs.shape[2] == 1:
        #         obs = obs.repeat(3, axis=2)
        #     display_arr(screen, obs, transpose=transpose, video_size=video_size)

        # process pygame events
        key = None
        for event in pygame.event.get():
            # test events, set key states
            
            if event.type == pygame.KEYDOWN:
                print(event.key)
                if event.key == 273:
                    key = 3
                #left
                elif event.key == 276:
                    key = 1
                #Down
                elif event.key == 274:
                    key = 2
                #right    
                elif event.key == 275:
                    key = 0
                #escp
                elif event.key == 27:
                    pygame.quit()

            #     if event.key in relevant_keys:
            #         pressed_keys.append(event.key)
            #     elif event.key == 27:
            #         running = False
            # elif event.type == pygame.KEYUP:
            #     if event.key in relevant_keys:
            #         pressed_keys.remove(event.key)
            # elif event.type == pygame.QUIT:
            #     running = False
            # elif event.type == VIDEORESIZE:
            #     video_size = event.size
            #     screen = pygame.display.set_mode(video_size)
            #     print(video_size)
            if key is not None:
                observation, step_reward, done, info = env.step(key)
                print(observation)
                key = None

                env.render()
                current_steps += 1

        

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()

def experiment_manager(top_level, n):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    # Load configuration.
    config_1 = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    config_2 = neat.config.Config(neat.DefaultGenome, neat.SexReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    configs = []


    if not os.path.exists(top_level):
        os.makedirs(top_level)
    else:
        raise

    for i in range(n):
        exp_direct = os.path.join(top_level, str(i))
        os.makedirs(exp_direct)

        prefix = os.path.join(exp_direct, "neat-checkpoint-")

        train(config_2, filename_prefix=prefix, max_generations=350)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    # Load configuration.
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    #train(config)
    run_genome(config, genome_file="./exp-length/0/neat-checkpoint-349")
    #visualize_genome()

    #run_enviroment()
    #play(gym.make("evolve-v0"), step_max=300)

    #experiment_manager("exp-length", 1)
