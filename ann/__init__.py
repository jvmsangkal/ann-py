import numpy as np
import matplotlib.pyplot as plt


class ANN(object):
    def __init__(self, n_in, n_h1, n_h2, n_out, eta, max_epoch):
        self.n_in = n_in
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.n_out = n_out
        self.eta = eta
        self.max_epoch = max_epoch

        self.w_h1 = self.seed(n_h1, n_in)
        self.b_h1 = self.seed(n_h1, 1)
        self.w_h2 = self.seed(n_h2, n_h1)
        self.b_h2 = self.seed(n_h2, 1)
        self.w_out = self.seed(n_out, n_h2)
        self.b_out = self.seed(n_out, 1)

    def seed(self, rows, cols):
        return -0.1 + (0.1 + 0.1) * np.random.rand(rows, cols)

    def forward(self, x_in):
        v_h1 = self.w_h1.dot(x_in) + self.b_h1
        self.y_h1 = self.sigmoid(v_h1)

        v_h2 = self.w_h2.dot(self.y_h1) + self.b_h2
        self.y_h2 = self.sigmoid(v_h2)

        v_out = self.w_out.dot(self.y_h2) + self.b_out
        out = self.sigmoid(v_out)

        return out

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def backward(self, x_in, d_out, out):
        err_out = d_out - out
        delta_out = err_out * self.sigmoid_prime(out)

        err_h2 = self.w_out.T.dot(delta_out)
        delta_h2 = err_h2 * self.sigmoid_prime(self.y_h2)

        err_h1 = self.w_h2.T.dot(delta_h2)
        delta_h1 = err_h1 * self.sigmoid_prime(self.y_h1)

        self.w_out += self.eta * delta_out.dot(self.y_h2.T)
        self.b_out += self.eta * delta_out

        self.w_h2 += self.eta * delta_h2.dot(self.y_h1.T)
        self.b_h2 += self.eta * delta_h2

        self.w_h1 += self.eta * delta_h1.dot(x_in.T)
        self.b_h1 += self.eta * delta_h1

        return err_out

    def sigmoid_prime(self, s):
        return s * (1 - s)

    def train(self, X, y):
        total_err = []
        for q in range(self.max_epoch):
            for (x, y_target) in zip(X, y):
                x = self.to_2d_array(x).T

                d_out = np.zeros(self.n_out, dtype=np.int)
                d_out[y_target - 1] = 1
                d_out = self.to_2d_array(d_out).T

                out = self.forward(x)
                err = self.backward(x, d_out, out)

            error = np.sum(np.square(err))
            total_err.append(error)

            if q % 5 == 0:
                print('Iteration: {} Error: {}'.format(q, total_err[q]))

            if total_err[q] < 0.001:
                print('Iteration: {} Error: {}'.format(q, total_err[q]))
                break

        fig, ax = plt.subplots()
        ax.plot(total_err)

        ax.set(
            xlabel='epoch',
            ylabel='errors',
            title='errors vs epoch'
        )
        ax.grid()

        fig.savefig('plot.png')

    def to_2d_array(self, l):
        return np.array(l, ndmin=2)

    def predict(self, X):
        y = np.zeros(len(X), dtype=np.int)
        for i, x in enumerate(X):
            x = self.to_2d_array(x).T
            out = self.forward(x)
            y[i] = int(np.argmax(out) + 1)

        return y
