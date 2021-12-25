import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import math


def log_likelihood(data, mu, cov):
    like = 0
    for x in data:
        temp = x - mu
        like += -.5 * ((np.log(2 * np.pi * np.linalg.det(cov))) + (np.dot(temp.T, np.dot(np.linalg.inv(cov), temp))))
    return like


def log_likelihood_contour(data, mu_X, mu_Y, cov):
    like = 0
    result = np.zeros(mu_X.shape)
    for x in data:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                diff = np.zeros(2)
                diff[0] = x[0] - mu_X[i][j]
                diff[1] = x[1] - mu_Y[i][j]
                result[i, j] += -.5 * ((np.log(2 * np.pi * np.linalg.det(cov))) + (np.dot(diff.T, np.dot(np.linalg.inv(cov), diff))))
    return result


Text = open('Log.txt', 'w')
fig = plt.figure()
camera = Camera(fig)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
cov = [[3, 0], [0, 3]]
data = np.random.multivariate_normal(mean=[2, 2], cov=cov, size=500)
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
xx, yy = np.meshgrid(x, y)
z = log_likelihood_contour(data, xx, yy, cov)
actual_mean = data.mean(axis=0)
init = np.random.uniform(-3, 3, (3, 2))
flag = True
c = 0
while flag:
    print(c, file=Text)
    print(c)
    init_res = []
    print('init :', init, file=Text)
    for i in init:
        init_res.append({'point': i, 'log_likelihood': log_likelihood(data, i, cov)})
    init_res.sort(key=lambda x: x['log_likelihood'], reverse=True)
    B = init_res[0]['point']
    N = init_res[1]['point']
    W = init_res[2]['point']
    M = (B + N) / 2
    R = (2 * M) - W
    E = (3 * M) - (2 * W)
    C_R = (1.5 * M) - (0.5 * W)
    C_W = (0.5 * M) + (0.5 * W)
    if log_likelihood(data, R, cov) > log_likelihood(data, B, cov):
        if log_likelihood(data, E, cov) > log_likelihood(data, R, cov):
            print('||Point|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(B, N, W, R, E, C_R, C_W), file=Text)
            print('||Log Likelihood|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(log_likelihood(data, B, cov), log_likelihood(data, N, cov), log_likelihood(data, W, cov), log_likelihood(data, R, cov), log_likelihood(data, E, cov), log_likelihood(data, C_R, cov), log_likelihood(data, C_W, cov)), file=Text)
            init = np.array([B, N, E])
            print('state :', 'BNE', file=Text)
        elif log_likelihood(data, E, cov) < log_likelihood(data, R, cov):
            print('||Point|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(B, N, W, R, E, C_R, C_W), file=Text)
            print('||Log Likelihood|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(log_likelihood(data, B, cov), log_likelihood(data, N, cov), log_likelihood(data, W, cov), log_likelihood(data, R, cov), log_likelihood(data, E, cov), log_likelihood(data, C_R, cov), log_likelihood(data, C_W, cov)), file=Text)
            init = np.array([B, N, R])
            print('state :', 'BNR', file=Text)
    elif (log_likelihood(data, N, cov) < log_likelihood(data, R, cov)) and (log_likelihood(data, R, cov) < log_likelihood(data, B, cov)):
        print('||Point|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(B, N, W, R, E, C_R, C_W), file=Text)
        print('||Log Likelihood|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(log_likelihood(data, B, cov), log_likelihood(data, N, cov), log_likelihood(data, W, cov), log_likelihood(data, R, cov), log_likelihood(data, E, cov), log_likelihood(data, C_R, cov), log_likelihood(data, C_W, cov)), file=Text)
        init = np.array([B, N, R])
        print('state :', 'BNR', file=Text)
    elif log_likelihood(data, R, cov) < log_likelihood(data, N, cov):
        if (log_likelihood(data, W, cov) < log_likelihood(data, R, cov)) and (log_likelihood(data, R, cov) < log_likelihood(data, N, cov)):
            print('||Point|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(B, N, W, R, E, C_R, C_W), file=Text)
            print('||Log Likelihood|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(log_likelihood(data, B, cov), log_likelihood(data, N, cov), log_likelihood(data, W, cov), log_likelihood(data, R, cov), log_likelihood(data, E, cov), log_likelihood(data, C_R, cov), log_likelihood(data, C_W, cov)), file=Text)
            init = np.array([B, N, C_R])
            print('state :', 'BNC_R', file=Text)
        elif log_likelihood(data, R, cov) < log_likelihood(data, W, cov):
            print('||Point|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(B, N, W, R, E, C_R, C_W), file=Text)
            print('||Log Likelihood|| -> B:{0} , N:{1} , W:{2} ,R:{3} , E:{4} ,C_R:{5} ,C_W:{6}'.format(log_likelihood(data, B, cov), log_likelihood(data, N, cov), log_likelihood(data, W, cov), log_likelihood(data, R, cov), log_likelihood(data, E, cov), log_likelihood(data, C_R, cov), log_likelihood(data, C_W, cov)), file=Text)
            init = np.array([B, N, C_W])
            print('state :', 'BNC_W', file=Text)
    RMSE = math.sqrt(init_res[0]['log_likelihood']**2 - log_likelihood(data, actual_mean, cov)**2)
    print('RMSE :', RMSE, file=Text)
    if (abs((np.linalg.norm(actual_mean)) - (np.linalg.norm(init_res[0]['point'])))) <= 1e-4:
        flag = False
    point = [actual_mean]
    color = ['red', 'blue', 'darkorange', 'green']
    m = ['o', '$B$', '$N$', '$W$']
    for idx in init_res:
        point.append(idx['point'])
    point = np.array(point).reshape(4, 2)
    plt.contour(xx, yy, z, levels=14, colors='black')
    for i, im, ic in zip(point, m, color):
        plt.scatter(i[0], i[1], marker=im, c=ic, s=100)
    t1 = plt.Polygon(point[1:4, :], color='black', fill=False)
    plt.gca().add_patch(t1)
    plt.legend([f'Epoch :{c}'])
    camera.snap()
    print('################################', file=Text)
    c += 1
print('error_best_actual :', abs(np.linalg.norm(actual_mean) - np.linalg.norm(init_res[0]['point'])), file=Text)
animation = camera.animate(interval=400, repeat=False)
animation.save('Simplex.gif', writer='imagemagick')
Text.close()
