
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
import time
# from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# set up the neural net work basics
def set_weights_bias(layers):
    weights_list = []
    biases_list = []
    len_layers = len(layers)
    for l in range(0,len_layers-1):
        W = weight_compute(size_list=[layers[l],layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights_list.append(W)
        biases_list.append(b)
    return weights_list, biases_list

def weight_compute(size_list):
    input_dimsion = size_list[0]
    output_dimsion = size_list[1]
    xavier_stddev = np.sqrt(2/(input_dimsion + output_dimsion))
    return tf.Variable(tf.truncated_normal([input_dimsion, output_dimsion], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

def set_neural_net(original, weights_list, biases_list):
    num_layers = len(weights_list) + 1
    H = original
    for l in range(0,num_layers-2):
        Weights = weights_list[l]
        bias = biases_list[l]
        H = tf.sin(tf.add(tf.matmul(H, Weights), bias))
    Weights = weights_list[-1]
    bias = biases_list[-1]
    output = tf.add(tf.matmul(H, Weights), bias)
    return output

class DWPMR:

    def __init__(self, t, x, u,
                       u_layers, pde_layers,
                       lower_bound, upper_bound):


        # Domain Boundary
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        print(self.lower_bound)
        # Identification
        self.identification(t, x, u, u_layers, pde_layers)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def identification(self, t, x, u, u_layers, pde_layers):
        # Layers
        self.u_layers = u_layers
        self.pde_layers = pde_layers
        # raw data from input
        self.t = t
        self.x = x
        self.u = u

        # add a one and two for later computation
        self.one = np.full_like(self.u, 1.0)
        self.two = np.full_like(self.u, 2.0)

        # Initialize NNs for Identification
        self.u_weights_list, self.u_biases_list = set_weights_bias(u_layers)
        self.pde_weights_list, self.pde_biases_list = set_weights_bias(pde_layers)

        # tf placeholders for Identification
        self.to_feed_t = tf.placeholder(tf.float32, shape=[None, 1])
        self.to_feed_x = tf.placeholder(tf.float32, shape=[None, 1])
        self.to_feed_u = tf.placeholder(tf.float32, shape=[None, 1])
        self.to_feed_terms = tf.placeholder(tf.float32, shape=[None, pde_layers[0]])
        self.to_feed_one = tf.placeholder(tf.float32, shape=[None, 1])
        self.to_feed_two= tf.placeholder(tf.float32, shape=[None, 1])

        # tf graphs for Identification
        self.ide_u_predict = self.ide_net_u(self.to_feed_t, self.to_feed_x)
        self.pde_predict = self.net_pde(self.to_feed_terms)
        self.ide_f_predict = self.ide_net_f(self.to_feed_t, self.to_feed_x,self.to_feed_u)[0]
        self.ide_f_ut_predict = self.ide_net_f(self.to_feed_t, self.to_feed_x,self.to_feed_u)[1]

        # for regularization
        self.ide_regular_predict = self.ide_net_f(self.to_feed_t, self.to_feed_x,self.to_feed_u)[2]
        # loss for Identification
        self.ide_u_loss = tf.reduce_sum(tf.square(self.ide_u_predict - self.to_feed_u))

        # normal weights for f_loss
        # self.idn_f_loss = tf.reduce_sum(tf.square(self.idn_f_pred))

        # dynamic weights with tan for f_loss
        self.ide_f_loss = tf.reduce_sum(tf.multiply(tf.subtract(self.to_feed_one,tf.nn.tanh(tf.square(self.ide_u_predict - self.to_feed_u))),tf.square(self.ide_f_predict)))

        # dynamic weights with logistic for f_loss
        # self.idn_f_loss = tf.reduce_sum(tf.multiply(tf.multiply(tf.subtract(self.one_tf,tf.nn.sigmoid(tf.square(self.idn_u_pred - self.u_tf))),self.two),tf.square(self.idn_f_pred)))

        # regulation item
        self.regular_weights = tf.reduce_sum(self.to_feed_one)
        self.ide_regular = tf.reduce_sum(tf.square(self.ide_regular_predict)) * 1.0/2.0 * self.regular_weights



        # regularization
        self.ide_regular_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.ide_regular,
                               var_list =  self.u_weights_list + self.u_biases_list,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 50000,
                                          'maxfun': 50000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})


        # Optimizer for Identification
        self.ide_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.ide_u_loss,
                               var_list = self.u_weights_list + self.u_biases_list,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 50000,
                                          'maxfun': 50000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})

        self.ide_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.ide_f_loss,
                               var_list = self.pde_weights_list + self.pde_biases_list,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 50000,
                                          'maxfun': 50000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})



        self.ide_u_optimizer_Adam = tf.train.AdamOptimizer()
        self.ide_u_train_op_Adam = self.ide_u_optimizer_Adam.minimize(self.ide_u_loss,var_list = self.u_weights_list + self.u_biases_list)

        self.ide_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.ide_f_train_op_Adam = self.ide_f_optimizer_Adam.minimize(self.ide_f_loss,var_list = self.pde_weights_list + self.pde_biases_list)

        # for regularization
        self.ide_regular_optimizer_Adam = tf.train.AdamOptimizer()
        self.ide_regular_train_op_Adam = self.ide_regular_optimizer_Adam.minimize(self.ide_regular,var_list = self.u_weights_list + self.u_biases_list)



# first network to learn u
    def ide_net_u(self, t, x):
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lower_bound)/(self.upper_bound - self.lower_bound) - 1.0
        u = set_neural_net(H, self.u_weights_list, self.u_biases_list)
        return u

    def net_pde(self, terms):
        pde = set_neural_net(terms, self.pde_weights_list, self.pde_biases_list)
        return pde


# second network to learn f
    def ide_net_f(self, t, x,u0):
        u = self.ide_net_u(t, x)

        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        terms = tf.concat([u,u0,u_xx],1)

        f = u_tt - self.net_pde(terms)

        regular = u_t + u * u_x - 0.1*u_xx

        # for regulation
        return [f,u_t,regular]


#  for training two networks
    def ide_regular_train(self, itertime):
        tf_dict = {self.to_feed_t: self.t, self.to_feed_x: self.x, self.to_feed_u: self.u,self.to_feed_one:self.one}

        start_time = time.time()
        for i in range(itertime):

            self.sess.run(self.ide_regular_train_op_Adam, tf_dict)

            # For print check
            if i % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.ide_regular, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (i, loss_value, elapsed))
                start_time = time.time()

        self.ide_regular_optimizer.minimize(self.sess,feed_dict = tf_dict,fetches = [self.ide_regular],loss_callback = self.callback)

    def ide_u_train(self, itertime):
        tf_dict = {self.to_feed_t: self.t, self.to_feed_x: self.x, self.to_feed_u: self.u}

        start_time = time.time()
        for i in range(itertime):

            self.sess.run(self.ide_u_train_op_Adam, tf_dict)

            # Print
            if i % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.ide_u_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (i, loss_value, elapsed))
                start_time = time.time()

        self.ide_u_optimizer.minimize(self.sess,feed_dict = tf_dict,fetches = [self.ide_u_loss],loss_callback = self.callback)

    def ide_f_train(self, itertime):
        # tf_dict = {self.t_tf: self.t, self.x_tf: self.x}
        tf_dict = {self.to_feed_t: self.t, self.to_feed_x: self.x, self.to_feed_u: self.u, self.to_feed_one:self.one}

        start_time = time.time()
        for i in range(itertime):

            self.sess.run(self.ide_f_train_op_Adam, tf_dict)

            # For print check
            if i % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.ide_f_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (i, loss_value, elapsed))
                start_time = time.time()

        self.ide_f_optimizer.minimize(self.sess,feed_dict = tf_dict,fetches = [self.ide_f_loss],loss_callback = self.callback)



    def ide_predict(self, t_star, x_star ,u_star):

        tf_dict = {self.to_feed_t: t_star, self.to_feed_x: x_star, self.to_feed_u:u_star}

        u_star_output = self.sess.run(self.ide_u_predict, tf_dict)
        f_star_output = self.sess.run(self.ide_f_predict, tf_dict)

        return u_star_output, f_star_output

    def predict_pde(self, terms_star):

        tf_dict = {self.to_feed_terms: terms_star}

        pde_star = self.sess.run(self.pde_predict, tf_dict)

        return pde_star
    #

    def callback(self, loss):
        print('Loss: %e' % (loss))


if __name__ == "__main__":

    # Doman bounds
    lb_idn = np.array([0.0, 0.0])
    ub_idn = np.array([255.0, 255.0])

    ### Load Data ###

    # data_idn = scipy.io.loadmat('../Data/burgers.mat')
    im = plt.imread("Lenna.jpg")
    im_0 = im[:, :, 0]
    data_idn = im_0
    imsize = im_0.shape
    # t_idn = data_idn['t'].flatten()[:,None]
    # x_idn = data_idn['x'].flatten()[:,None]
    # Exact_idn = np.real(data_idn['usol'])
    #
    # T_idn, X_idn = np.meshgrid(t_idn,x_idn)
    x_list = []
    y_list = []
    u_list = []
    for i in range(imsize[0]):
        for j in range(imsize[0]):
            x_list.append(np.array([i]))
            y_list.append(np.array([j]))
            u_list.append(np.array([im_0[i][j]]))


    x_array = np.array(x_list)
    y_array = np.array(y_list)

    u_array = np.array(u_list)

    # For identification
    N_train = 10000

    idx = imsize[0]*imsize[0]
    # idx = np.random.choice(u_array.shape[0], N_train, replace=False)
    # y_train = y_array[idx,:]
    # x_train = x_array[idx,:]
    # u_train = u_array[idx,:]
    x_train = []
    y_train = []
    u_train = []
    for i in range(idx):
        x_train.append(np.array(x_array[i]))
        y_train.append(np.array(y_array[i]))
        u_train.append(np.array(u_array[i]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    u_train = np.array(u_train)

    noise = 0.05
    u_train = u_train + noise * np.std(u_train)* np.random.randn(u_train.shape[0], u_train.shape[1])
    u_noise = u_array + noise * np.std(u_array)* np.random.randn(u_array.shape[0], u_array.shape[1])


    # # set up layer
    # # Layers
    u_layers = [2, 50, 50, 50, 50, 1]
    pde_layers = [3, 100, 100, 1]

    layers = [2, 50, 50, 50, 50, 1]

    mainmodel = DWPMR(y_train, x_train, u_train,
                    u_layers, pde_layers,
                    lb_idn, ub_idn,
                    )

    # Train the identifier
    mainmodel.ide_u_train(itertime=0)

    mainmodel.ide_f_train(itertime=0)

    # model.idn_u_f_train(N_iter=0)

    u_pred_identifier, f_pred_identifier = mainmodel.ide_predict(y_array, x_array,u_array)

    error_u_identifier = np.linalg.norm(u_array-u_pred_identifier,2)/np.linalg.norm(u_array,2)
    print('Error u: %e' % (error_u_identifier))
    print(u_array)
    print(u_pred_identifier)
    im_learn = []
    im_noise = []
    for i in range(imsize[0]):
        newline = []
        newlinenoise = []
        for j in range(imsize[0]):
            newline.append(u_pred_identifier[i*imsize[0]+j][0])
            newlinenoise.append(u_noise[i*imsize[0]+j][0])
        im_learn.append(newline)
        # print(newlinenoise)
        im_noise.append(newlinenoise)

    im_noise = np.array(im_noise)
    im_learn = np.array(im_learn)


    print("origianl")
    print(im_0)
    print("im_noise")
    print(im_noise)
    print("im_learn")
    print(im_learn)
    plt.subplot(1,2,1)
    plt.imshow(im_noise)
    plt.subplot(1,2,2)
    plt.imshow(im_learn)
    plt.show()
    # #