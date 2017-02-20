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


def build_matrix(dimension=30):
    train_set, test_set, user_index, item_index = random_split()
    U = np.random.rand(len(user_index))
    I = np.random.rand(len(item_index))
    Q = np.random.rand(len(item_index), dimension)/10
    P = np.random.rand(len(user_index), dimension)/10
    Y = np.random.rand(len(item_index), dimension)/10
    II = np.zeros((len(user_index), len(item_index)))
    for each in train_set:
        index1 = user_index.index(each[0])
        index2 = item_index.index(each[1])
        II[index1][index2] = 1
    return U, I, Y,  P, Q, train_set, test_set, user_index, item_index, dimension, II


def factorization(alpha, lamda_1, lamda_2):
    U, I, Y,  P, Q, train_set, test_set, user_index, item_index, dimension, II = build_matrix()
    user_items_dict = dict()
    all_rating_sum = 0
    for each in train_set:
        user_items_dict.setdefault(each[0], list())
        user_items_dict[each[0]].append(each[1])
        all_rating_sum += float(each[2])
    all_rating_ave = all_rating_sum/len(train_set)

    min_mae = 10.0
    min_rmse = 10.0
    diversity = 0.0
    steps = 90
    train_predict_error_old = 0.0
    cost_func_value_old = 0
    alpha_old = alpha
    for step in np.arange(steps):
        # calculate the cost function value
        cost_func_value = 0
        for each in train_set:
            u_index = user_index.index(each[0])
            i_index = item_index.index(each[1])
            each_rating_predict = all_rating_ave + U[u_index] + I[i_index]
            rated_item_sum = np.zeros(dimension)
            for rated_item in user_items_dict[each[0]]:
                rated_item_index = item_index.index(rated_item)
                rated_item_sum += Y[rated_item_index]
            rated_item_sum /= np.sqrt(len(user_items_dict[each[0]])*1.0)
            each_rating_predict += np.dot(Q[i_index], P[u_index]+rated_item_sum)
            cost_func_value += np.power(float(each[2])-each_rating_predict, 2)
        for each_user in user_index:
            u_index = user_index.index(each_user)
            cost_func_value += lamda_1*( np.power(U[u_index], 2))
            cost_func_value += lamda_2*(np.dot(P[u_index], P[u_index]))
            for rated_item in user_items_dict[each_user]:
                rated_item_index = item_index.index(rated_item)
                rated_item_sum += Y[rated_item_index]

                cost_func_value += lamda_2 * np.dot(Y[rated_item_index], Y[rated_item_index])
        for each_item in item_index:
            i_index = item_index.index(each_item)
            cost_func_value += lamda_1*(np.power(I[i_index], 2))
            cost_func_value += lamda_2*(np.dot(Q[i_index], Q[i_index]))

        # cost_func_value += my_norm(P, 2) + my_norm(Q, 2)
        # gradient update
        for each in train_set:
            u_index = user_index.index(each[0])
            i_index = item_index.index(each[1])
            each_rating_predict = all_rating_ave + U[u_index] + I[i_index]
            rated_item_sum = np.zeros(dimension)
            for rated_item in user_items_dict[each[0]]:
                rated_item_index = item_index.index(rated_item)
                rated_item_sum += Y[rated_item_index]

            rated_item_sum /= np.sqrt(len(user_items_dict[each[0]])*1.0)
            each_rating_predict += np.dot(Q[i_index], P[u_index]+rated_item_sum)
            Eui = float(each[2]) - each_rating_predict
            U[u_index] += alpha*(Eui-lamda_1*U[u_index])
            I[i_index] += alpha*(Eui-lamda_1*I[i_index])
            Q[i_index] += alpha*(Eui*(P[u_index]+rated_item_sum)-lamda_2*Q[i_index])
            P[u_index] += alpha*(Eui*Q[i_index]-lamda_2*P[u_index])
            for rated_item in user_items_dict[each[0]]:
                rated_item_index = item_index.index(rated_item)
                Y[rated_item_index] += alpha*(Eui*np.sqrt(len(user_items_dict[each[0]])*1.0)*Q[i_index]-lamda_2*Y[rated_item_index])

        if cost_func_value < cost_func_value_old:
            if step < 10:
                alpha *= 1.15
            else:
                alpha *= 1.02
        else:
            alpha = alpha_old * 0.5
        alpha_old = alpha
        cost_func_value_old = cost_func_value

        print("*************  this is the ", step, " step **********")
        print('error func value = ', cost_func_value)

        if step > 0 and step % 1 == 0:
            mae_test_set_error = 0
            rmse_test_set_error = 0
            for each in test_set:
                u_index = user_index.index(each[0])
                i_index = item_index.index(each[1])
                each_rating_predict = all_rating_ave + U[u_index] + I[i_index]
                rated_item_sum = np.zeros(dimension)
                for rated_item in user_items_dict[each[0]]:
                    rated_item_index = item_index.index(rated_item)
                    rated_item_sum += Y[rated_item_index]
                rated_item_sum /= np.sqrt(len(user_items_dict[each[0]])*1.0)
                each_rating_predict += np.dot(Q[i_index], P[u_index]+rated_item_sum)
                mae_test_set_error += np.abs(each_rating_predict-float(each[2]))
                rmse_test_set_error += np.power(each_rating_predict-float(each[2]), 2)

            mae = mae_test_set_error/(1.0*len(test_set))
            rmse = rmse_test_set_error/(1.0*len(test_set))
            rmse = np.sqrt(rmse)
            if mae < min_mae:
                min_mae = mae
            if rmse < min_rmse:
                min_rmse = rmse
            print(mae, ',', rmse)
    path = r'E:\workspace\python_workspace\2016_7_3_yelp_rmse_mae\result'
    fw = open(path+'\\result_svd.txt', 'a')
    fw.write('base: alpha='+str(alpha)+',lamda='+str(lamda_1)+str(lamda_2)+'\n')
    fw.write('mae::'+str(min_mae)+', rmse::'+str(min_rmse)+'\n')
    fw.close()
    # // output console
    print('base: alpha='+str(alpha)+',lamda='+str(lamda_1)+str(lamda_2)+'\n')
    print('base_mae'+str(min_mae))
    print('base_rmse'+str(min_rmse))

print("***************")
factorization(0.0001, 1, 1)

