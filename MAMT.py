# -*- coding: utf-8 -*-
import numpy as np
import csv

tail_merchants_set = set()
user_item_set = set()


def my_norm(matrix_, t=2):
    t = 2
    m, n = matrix_.shape
    sum_ = 0
    for i in np.arange(m):
        for j in np.arange(n):
            try:
                sum_ += matrix_[i][j]*matrix_[i][j]
            except Exception as e:
                print(matrix_[i][j])
                print(e)
    return np.sqrt(sum_)


def random_split(percentage=0.3):
    print('percentage', "::", percentage)
    csv_file = open("dataset/user_item_rating.csv")
    rf = csv.reader(csv_file)
    user_reviews = dict()   # {user_id:[(user_id, item_id, rating),...],...}
    for each in rf:
        user_id = each[0]
        user_reviews.setdefault(user_id, list())
        user_reviews[user_id].append(tuple(each))
    train = list()
    test = list()
    # print("user number", len(user_reviews))
    for each_user in user_reviews.keys():
        ######################
        all_ratings_list = user_reviews[each_user]
        if len(all_ratings_list) <= 10:
            continue
        review_count = len(all_ratings_list)

        train_index = np.random.choice(np.arange(0, review_count), int(np.around(percentage*review_count)),
                                       replace=False)
        test_index = list(set(np.arange(0, review_count)).difference(set(train_index)))
        # one_user_reviews = user_reviews[each_user]
        if len(train_index) == 0:
            continue
        for each in train_index:
            train.append(all_ratings_list[each])
        for each in test_index:
            test.append(all_ratings_list[each])
        #######################
    user_index = set()
    item_index = set()
    for each in train:
        user_index.add(each[0])
        item_index.add(each[1])
    user_index = list(user_index)
    item_index = list(item_index)
    new_test = list()
    for each in test:
        if each[1] in item_index:
            new_test.append(each)
    test = new_test
    new_test = list()
    return train, test, user_index, item_index


def build_matrix(dimension=30, aspects=15, word_length=10):
    train_set, test_set, user_index, item_index = random_split()
    # train_set operation
    R = np.zeros((len(user_index), len(item_index)))
    I = np.zeros((len(user_index), len(item_index)))
    for each in train_set:
        index1 = user_index.index(each[0])
        index2 = item_index.index(each[1])
        R[index1][index2] = float(each[2])
        I[index1][index2] = 1
    # test_set operation
    R_T = np.zeros((len(user_index), len(item_index)))
    I_T = np.zeros((len(user_index), len(item_index)))
    for each in test_set:
        index1 = user_index.index(each[0])
        index2 = item_index.index(each[1])
        R_T[index1][index2] = float(each[2])
        I_T[index1][index2] = 1

    U = np.random.rand(len(user_index), dimension)
    V = np.random.rand(len(item_index), dimension)

    B = np.random.rand(len(user_index), aspects)/10
    Z = np.random.rand(len(item_index), aspects)/10

    S = np.random.rand(word_length, aspects)
    return R, I, R_T, I_T, U, V, B, Z, S, user_index, item_index, test_set, aspects, word_length, dimension


def factorization(alpha, lamda):
    R, I, R_T, I_T, U, V, B, Z, S, user_index, item_index, test_set, aspects, word_length, dimension = build_matrix()
    import json
    item_aspects = json.load(open('dataset/item_topics.json', 'r'))
    min_mae = 5.0
    min_rmse = 5.0
    diversity = 0.0
    steps = 90
    train_predict_error_old = 0.0
    cost_func_value_old = 0
    alpha_old = alpha
    for step in np.arange(steps):
        print("********** this is the ", step, " step **********")
        # alpha /= (1 + step*1.0/steps)
        cost_func_value = my_norm((np.dot(U, V.T)+np.dot(B, Z.T)-R)*I, 2)+lamda*(my_norm(U, 2)+my_norm(V, 2)+my_norm(B, 2))
        # update the Z matrix
        for i in np.arange(len(item_index)):
            item_name = item_index[i]
            W = np.array(item_aspects[item_name])
            WS = W.T * S
            Z[i, :] = np.sum(WS, axis=0)
        # S gradient update. S matrix is the sentiment polarities matrix
        for k in np.arange(aspects):
            s_gradient = np.zeros((1, word_length))[0, :]
            for i in np.arange(len(user_index)):
                for j in np.arange(len(item_index)):
                    if I[i][j] == 1:
                        tmp = (np.dot(U[i], V[j])+np.dot(B[i], Z[j])-R[i][j])*B[i][k]
                        wjk = np.array(item_aspects[item_index[j]]).T
                        s_gradient += tmp * wjk[:, k]
            S[:, k] -= s_gradient * alpha
        # U gradient update
        for i in np.arange(len(user_index)):
            u_gradient = np.zeros((1, dimension))[0, :]
            for j in np.arange(len(item_index)):
                if I[i][j] == 1:
                    temp = np.dot(U[i], V[j]) + np.dot(B[i], Z[j]) - R[i][j]
                    u_gradient += np.dot(temp, V[j])
            U[i] -= alpha*(u_gradient + lamda*U[i])
        # V gradient update
        for j in np.arange(len(item_index)):
            v_gradient = np.zeros((1, dimension))[0, :]
            for i in np.arange(len(user_index)):
                if I[i][j] == 1:
                    temp = np.dot(U[i], V[j]) + np.dot(B[i], Z[j]) - R[i][j]
                    v_gradient += np.dot(temp, U[i])
            V[j] -= alpha*(v_gradient + lamda*V[j])
        # B gradient update
        for i in np.arange(len(user_index)):
            b_gradient = np.zeros((1, aspects))[0, :]
            for j in np.arange(len(item_index)):
                if I[i][j] == 1:
                    temp = np.dot(U[i], V[j]) + np.dot(B[i], Z[j]) - R[i][j]
                    b_gradient += np.dot(temp, Z[j])
            B[i] -= alpha*(b_gradient + lamda*B[i])

        # change the learning rate based on the how much cost function value changed
        if cost_func_value < cost_func_value_old:
            if step < 10:
                alpha *= 1.15
            else:
                alpha *= 1.02
        else:
            alpha = alpha_old * 0.5
        alpha_old = alpha
        cost_func_value_old = cost_func_value

        if step >= 0:
            # calculate the predict error on train_set
            R_P = np.dot(U, V.T)+np.dot(B, Z.T)
            train_error_matrix = I * (R - R_P)
            train_predict_error = np.sum(train_error_matrix * train_error_matrix)
            print("cost  func   value = ", cost_func_value)
            print("error change scale = ", abs(train_predict_error - train_predict_error_old))
            if abs(train_predict_error - train_predict_error_old) < 4:
                print('Program exit after convergence!')
                break
            else:
                train_predict_error_old = train_predict_error

            # calculate the predict error on test_set
            test_set_error = R_T - I_T * R_P
            mae = np.sum(abs(test_set_error))/(len(test_set))
            rmse = np.sum(pow(test_set_error, 2))/(len(test_set))
            rmse = np.sqrt(rmse)
            if mae < min_mae:
                min_mae = mae
            if rmse < min_rmse:
                min_rmse = rmse

            print(mae, rmse)
    path = r'E:\workspace\python_workspace\2016_7_3_yelp_rmse_mae\result'
    fw = open(path+'\\result_MAMT.txt', 'a')
    fw.write('base: alpha='+str(alpha)+',lamda='+str(lamda)+'\n')
    fw.write('mae::'+str(min_mae)+', rmse::'+str(min_rmse)+'\n')
    fw.close()
    # // output console
    print('base: alpha='+str(alpha)+',lamda='+str(lamda))
    print('base_mae'+str(min_mae))
    print('base_rmse'+str(min_rmse))



print("***************")
factorization(0.0002, 3)

