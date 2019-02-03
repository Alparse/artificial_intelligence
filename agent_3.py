import tensorflow as tf

tf.reset_default_graph()
graph1 = tf.Graph()
import math
import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

import tic_tac_toe as ttt

win_rate=[]
error_rate=[]
rounds=0

def calculate_mean_of_gradients(gradient_dictionary):
    gradients_in_count = len(gradient_dictionary)
    mean_gradients_out_placeholder = []

    for item in gradient_dictionary[0]:
        append_shape = np.shape(item)
        mean_gradients_out_placeholder.append(np.zeros(append_shape))

    for ind2, gradients_in in enumerate(gradient_dictionary):
        for ind, placeholder_gradient in enumerate(mean_gradients_out_placeholder):
            mean_gradients_out_placeholder[ind] = placeholder_gradient + gradients_in[ind] * game.reward_dictionary1[
                ind2]

    for ind, mean_gradient in enumerate(mean_gradients_out_placeholder):
        mean_gradients_out_placeholder[ind] = mean_gradient / gradients_in_count



    return (mean_gradients_out_placeholder)


with graph1.as_default():
    x_ = tf.placeholder(tf.float32, shape=[1, 18], name="x-input")  # Tensor, placeholder
    y_ = tf.placeholder(tf.float32, shape=[1, 9], name="y-input")  # Tensor, placeholder

    Theta1 = tf.Variable(tf.random_uniform([18, 36], -1, 1), name="Theta1", trainable=True)  # Tensor, variable
    Theta2 = tf.Variable(tf.random_uniform([36, 18], -1, 1), name="Theta2", trainable=True)  # Tensor, variable
    #Theta3 = tf.Variable(tf.random_uniform([18, 9], -1, 1), name="Theta3", trainable=True)  # Tensor, variable
    Theta3 = tf.Variable(tf.random_uniform([18, 9], -1, 1), name="Theta4", trainable=True)  # Tensor, variable

    Bias1 = tf.Variable(tf.zeros([36]), name="Bias1", trainable=True)  # Tensor, variable
    Bias2 = tf.Variable(tf.zeros([18]), name="Bias2", trainable=True)  # Tensor, variable
    #Bias3 = tf.Variable(tf.zeros([9]), name="Bias3", trainable=True)  # Tensor, variable
    Bias3 = tf.Variable(tf.zeros([9]), name="Bias4", trainable=True)  # Tensor, variable

    rewards1 = tf.placeholder(tf.float32, name="reward-1")

    A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)  # Operation

    A3 = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)  # Operation

    #A4 = tf.sigmoid(tf.matmul(A3, Theta3) + Bias3)  # Operation

    Hypothesis = tf.sigmoid(tf.matmul(A3, Theta3) + Bias3)  # Operation

    cost = tf.losses.mean_squared_error(y_, Hypothesis)

    optimizer = tf.train.AdamOptimizer(.001)

    grads_and_vars = optimizer.compute_gradients(cost)

    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    reward_dictionary = []
    discount_rate =.9
    duration_multiplier = 1
    games_won = 0
    games_lost = 0
    games_drawn = 0
    illegal_moves_made=0
    predicted_move = None
    step = None

    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))

    training_op = optimizer.apply_gradients(grads_and_vars_feed)

with tf.Session(graph=graph1) as sess:
    game = ttt.TicTacToeEnv()
    init = tf.global_variables_initializer()
    sess.run(init)

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    for i in range(100000):
        print("\nGAME # ", i)
        gradient_dictionary = []
        reward_dictionary = []
        grad_out_count = 0
        while game.game_status == 0:
            if game.turn in[2,4,6,8]:
                print("\nPLAYER 2 TURN", game.turn)
                player = 2.
                game.make_move(player, game.pick_random_legal_move(player))
                if game.check_if_draw():
                    game.game_status=3
                if game.check_if_game_won(player):
                    reward_dict_size=len(game.reward_dictionary1)
                    print("reward dict size",reward_dict_size)
                    game.reward_dictionary1[reward_dict_size-1][0]=0
                    game.game_status=2
                #print("Player 2 reward Dict ",game.reward_dictionary1)

            else:
                if game.turn in[1,3,5,7,9]:
                    print("\nPLAYER 1 TURN", game.turn)
                    player = 1
                    # Predict Move
                    #print("P1 Predict Move")
                    if game.turn == 1:
                        predicted_move = game.pick_random_legal_move(player)
                        step = predicted_move
                        #print("RANDOM PREDICT", step)
                    if game.turn > 1:
                        predicted_move = sess.run(Hypothesis, feed_dict={x_: game.obs_space})
                        step = np.argmax(predicted_move)
                       # print("NN PREDICT", step)

                    print("Hypothesis     ", np.array2string(np.asanyarray(predicted_move), max_line_width=np.inf),
                          "\n Predicted Move ", step)

                    #print("Game Status P1 ",game.game_status)

                    game.yrl_ = np.copy(game.allowed_total_action_space)
                    print("PRE ADJUSTED Y",game.yrl_)
                    game.yrl_[np.where(game.yrl_ == 1)] = 0
                    game.yrl_[np.where(game.yrl_ == 0)] = 0
                    game.yrl_[np.where(game.yrl_ == 2)] = 0
                    game.yrl_ = np.reshape(game.yrl_, [1, 9])
                    print("Pre ADJUSTED Y", game.yrl_)
                    game.yrl_.fill(0)
                    game.yrl_[0][step] = 1


                    if not game.check_if_move_legal(player,step):
                        game.game_status=2
                        illegal_moves_made=illegal_moves_made+1
                        print("Post ADJUSTED Y", game.yrl_)
                        print("Made illegal move & lost")

                    if game.check_if_move_legal(player,step):
                        game.make_move(player, step)

                        #print("Made legal move")

                    if game.check_if_draw()==True:

                        game.game_status=3


                    if game.check_if_game_won(player):
                        game.game_status=1
                        game.yrl_.fill(0)
                        game.yrl_[0][step] = 1
                        #print("Made winning move")

                    # Compute Cost

                    print("yrl_           ", game.yrl_)

                    move_cost = sess.run(cost, feed_dict={x_: game.obs_space, y_: game.yrl_})
                    print("Move Cost\n", np.array2string(np.asanyarray(move_cost), max_line_width=np.inf))

                    # Compute Rewards
                    #print("P1 Compute Rewards")
                    if game.game_status == 0:
                        game.reward1 = 1
                    if game.game_status == 1:
                        game.reward1 = 1.0 +((random.gauss(0,1))/100)
                    if game.game_status == 2:
                        game.reward1 = -1
                    if game.game_status == 3:
                        game.reward1 = 0

                    game.total_rewards1 = game.total_rewards1 + game.reward1
                    game.reward_dictionary1.append([game.reward1])



                    #print("\n",game.reward_dictionary1,"\n")

                    # Compute Gradients
                    #print("P1 Compute Gradients")
                    move_gradients = sess.run(grads_and_vars, feed_dict={x_: game.obs_space, y_: game.yrl_})

                    # Modify Gradients
                    #print("P1 Modify Gradients")
                    grad_out = [x[0] for x in move_gradients]
                    for i in range(len(grad_out)):
                        grad_out[i] = grad_out[i] #* game.reward1
                    vars_out = [x[1] for x in move_gradients]
                    grad_out_count = grad_out_count + 1
                    gradient_dictionary.append(grad_out)

            game.turn = game.turn + 1

            game.render()

        if game.game_status > 0:

            print("reward dict ",game.reward_dictionary1)
            #for ind, reward in enumerate(game.reward_dictionary1):
               #game.reward_dictionary1[ind]=game.reward_dictionary1[len(game.reward_dictionary1)-1][0]*(discount_rate**(len(game.reward_dictionary1)-ind))

            print("reward dict2", game.reward_dictionary1)
            mean_gradients = calculate_mean_of_gradients(gradient_dictionary)

            # Apply gradients
            #print("P1 Apply Gradients")
            feed_dict = {}
            for var_index, gradient_placeholder in enumerate(gradient_placeholders):
                feed_dict[gradient_placeholder] = mean_gradients[var_index]

            move_training = sess.run(training_op, feed_dict=feed_dict)

            if game.game_status == 1:
                games_won = games_won + 1
            if game.game_status == 2:
                games_lost = games_lost + 1
            if game.game_status == 3:
                games_drawn = games_drawn + 1

            print("Games Won ", games_won, " Games Lost ", games_lost, " Games Drawn ", games_drawn," Illegal Moves Made ", illegal_moves_made, " Winning % = ",100*(games_won/(games_lost+games_won+games_drawn)),"%")

            if games_lost + games_won + games_drawn == 100:

                win_rate_app=100*(games_won/(games_lost+games_won+games_drawn))
                error_rate_app=100*(illegal_moves_made/(games_won+games_lost+games_drawn))
                win_rate.append(win_rate_app)
                error_rate.append(error_rate_app)
                games_drawn, games_won, games_lost, illegal_moves_made = 0, 0, 0, 0
                rounds=rounds+1



            game = ttt.TicTacToeEnv()
            if rounds==50:
                cont=input("Continue Y/N")
                if cont =="Y":
                    rounds=0
                    continue
                else:
                    break

    plt.plot(win_rate)
    plt.ylabel('win rate')
    plt.show()
    plt.plot(error_rate)
    plt.ylabel('error rate')
    plt.show()