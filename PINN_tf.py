import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import qmc

np.random.seed(seed=1234)
tf.random.set_seed(1234)
tf.config.experimental.enable_tensor_float_32_execution(False)

# Initalization of Network
def hyper_initial(size):
    in_dim = size[0]
    out_dim = size[1]
    std = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal(shape=size, stddev = std))

# Neural Network 
def DNN(X, W, b):
    A = X
    L = len(W)
    for i in range(L-1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

def train_vars(W, b):
    return W + b

def net_u(x, W, b):
    u = DNN(tf.concat([x],1), W, b)
    return u

@tf.function(jit_compile=True)
def net_u_xx(x, W, b):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x])
            u = net_u(x, W, b)
        u_x = tape2.gradient(u, x)
    u_xx = tape1.gradient(u_x, x)  
    return u_xx

def Mx(x, p, L):
    Mx = 0.5 * p * (L*x - x**2)
    return Mx

@tf.function(jit_compile=True)
def net_f_2ndOrder(x, W, b, p, EI, L):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x])
            u = net_u(x, W, b)
        u_x = tape2.gradient(u, x)
    u_xx = tape1.gradient(u_x, x)  
    f = u_xx - Mx(x, p, L)/EI
    return f

@tf.function(jit_compile=True)
#@tf.function()
def net_f_4thOrder(x, W, b, p, EI, L):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x])
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch([x])
                with tf.GradientTape(persistent=True) as tape4:
                    tape4.watch([x])
                    u = net_u(x, W, b)
                u_x = tape4.gradient(u, x)
            u_xx = tape3.gradient(u_x, x)
        u_xxx = tape2.gradient(u_xx, x)
    u_xxxx = tape1.gradient(u_xxx, x)  
    f = u_xxxx - p/EI
    return f

@tf.function(jit_compile=True)
def train_step(W, b, x_BC, u_BC, uxx_BC, x_train, opt, p, EI, L):
    x_u = x_BC
    x_f = x_train
    with tf.GradientTape() as tape:
        tape.watch([W,b])
        u_nn = net_u(x_u, W, b) 
        #f_nn = net_f_2ndOrder(x_f, W, b, p, EI, L)
        uxx_nn = net_u_xx(x_u, W, b)
        f_nn = net_f_4thOrder(x_f, W, b, p, EI, L)
        loss =  tf.reduce_mean(tf.square(u_nn - u_BC)) +\
                tf.reduce_mean(tf.square(uxx_nn - uxx_BC)) +\
                tf.reduce_mean(tf.square(f_nn)) 
    grads = tape.gradient(loss, train_vars(W, b))
    opt.apply_gradients(zip(grads, train_vars(W, b)))
    return loss

def u_elasticLine(x, p, L, EI):
    u = p*x*(x**3 - 2*L*x**2 + L**3)/24/EI
    return u

if __name__ == "__main__":
    EI = tf.constant(1.0) # Bending stiffness
    pLoad = tf.constant(1.0) # Distributed load
    N_u = 20 + 1
    Lbeam = tf.constant(1.0) # m
    dx = Lbeam / (N_u-1)
    print("dx: ", dx)
    layers = [1, 10, 1]
    Nmax = 5000 # Max number of iterations for training the NN

    L = len(layers)
    W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
    b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)]

    x_tot   = tf.convert_to_tensor( np.reshape(np.arange(0.0, Lbeam+dx, dx), (-1,1)), dtype=tf.float32)
    x_BC    = tf.convert_to_tensor( np.reshape(np.array([0.0, Lbeam]), (-1,1)), dtype=tf.float32)
    u_BC    = tf.convert_to_tensor( np.reshape(np.array([0.0, 0.0]), (-1,1)), dtype=tf.float32) # BC on displacement u(x)
    uxx_BC  = tf.convert_to_tensor( np.reshape(np.array([0.0, 0.0]), (-1,1)), dtype=tf.float32) # BC on moment M(x)
    x_train = tf.convert_to_tensor( np.reshape(np.arange(dx, Lbeam-dx, dx), (-1,1)), dtype=tf.float32)

    lr = 1e-2
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    start_time = time.time()
    n = 0
    loss = []
    while n <= Nmax:
        loss_ = train_step(W, b, x_BC, u_BC, uxx_BC, x_train, optimizer, pLoad, EI, Lbeam)
        loss.append(loss_)
        if n % 10 == 0:
            print(f"Iteration is: {n} and loss is: {loss_}")
        n+=1
        if loss_ < 1e-5:
            break

    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

    plt.figure(figsize=(8, 4))
    plt.plot( tf.sort(x_tot,axis=0), -u_elasticLine(tf.sort(x_tot,axis=0), pLoad, Lbeam, EI), label="analytic solution", linewidth=2, linestyle="dashed")
    plt.plot( tf.sort(x_tot,axis=0), -net_u(tf.sort(x_tot,axis=0), W, b), label="PINN", linewidth=2)
    plt.legend()
    plt.savefig("deformed_shape.png")

    plt.figure(figsize=(8, 4))
    plt.plot(loss, linewidth=2)
    plt.yscale("log")
    plt.legend({"loss of PDE + BC"})
    plt.savefig("loss.png")

    plt.show()
