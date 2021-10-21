import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

name = input("Enter Save Name:")

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200
score_requirement = 50
initial_games = 1000

def initial_population():
    # [OBS, MOVES] syntax for input data
    training_data = []
    # array for scores
    scores = []
    # array for scores that meet criteria
    accepted_scores = []
    # iterate through games
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        for _ in range(goal_steps):
            #env.render()
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        # save scores above set threshold
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (from tflearn tutorial)
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                # appened data to training data
                training_data.append([data[0], output])

        # reset env 
        env.reset()
        # save scores
        scores.append(score)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

#initial_population()


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')
    #less than 128 nodes in a layer seems to hinder learning more than about 550 kills my ram!
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8) #think of this a keeprate
    
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    
    
    

   
                                        #cartpole has 2 outputs, left or right
    network = fully_connected(network, 2, activation='softmax')
                                    #using adam as recomended in the tutorial http://tflearn.org/optimizers/ other optimizers here :)
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
                                                        #http://tflearn.org/objectives/ again recomended by tutorials need to explore other objective types
    model = tflearn.DNN(network, tensorboard_dir='logfolder')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id=name)
    return model



training_data = initial_population()
model = train_model(training_data)

#plying using data down here
scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        choices.append(action)
                
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)




