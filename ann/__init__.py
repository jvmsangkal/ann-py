import numpy as np


class ANN(object):
    def __init__(self, n_in, n_h1, n_h2, n_out, eta, max_epoch):
        self.x_in = np.zeros((n_in, 1))
        self.w_h1 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h1, n_in)
        self.b_h1 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h1, 1)
        self.w_h2 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h2, n_h1)
        self.b_h2 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h2, 1)
        self.w_out = -0.1 + (0.1 + 0.1) * np.random.rand(n_out, n_h2)
        self.b_out = -0.1 + (0.1 + 0.1) * np.random.rand(n_out, 1)
        self.d_out = np.zeros((n_out, 1))

        self.eta = eta
        self.max_epoch = max_epoch

    def train(self, X, Y):
        N = len(X)
        total_err = np.zeros(self.max_epoch)
        for q in range(self.max_epoch):
            p = np.random.permutation(N)
            for n in range(N):
                nn = p[n]

                self.x_in = np.transpose(np.array(X[nn], ndmin=2))
                self.d_out = np.transpose(np.array(Y[nn], ndmin=2))

                v_h1 = np.dot(self.w_h1, self.x_in) + self.b_h1
                y_h1 = 1 / (1 + np.exp(-v_h1))

                v_h2 = np.dot(self.w_h2, y_h1) + self.b_h2
                y_h2 = 1 / (1 + np.exp(-v_h2))

                v_out = np.dot(self.w_out, y_h2) + self.b_out
                out = 1 / (1 + np.exp(-v_out))

                err = self.d_out - out
                delta_out = err * out * (1 - out)
                delta_h2 = y_h2 * (1 - y_h2) * np.dot(
                    np.transpose(self.w_out), delta_out)
                delta_h1 = y_h1 * (1 - y_h1) * np.dot(
                    np.transpose(self.w_h2), delta_h2)

                self.w_out = self.w_out + (
                    self.eta * np.dot(delta_out, np.transpose(y_h2)))
                self.b_out = self.b_out + (self.eta * delta_out)

                self.w_h2 = self.w_h2 + (
                    self.eta * np.dot(delta_h2, np.transpose(y_h1)))
                self.b_h2 = self.b_h2 + (self.eta * delta_h2)

                self.w_h1 = self.w_h1 + (
                    self.eta * np.dot(delta_h1, np.transpose(self.x_in)))
                self.b_h1 = self.b_h1 + (self.eta * delta_h1)

            total_err[q] = total_err[q] + np.sum(err * err)

            if q % 5 == 0:
                print('Iteration: {} Error: {}'.format(q, total_err[q]))

            if total_err[q] < 0.001:
                print('Iteration: {} Error: {}'.format(q, total_err[q]))
                break

    def predict(self, X):
        N = len(X)

        nn_output = []
        for n in range(N):
            self.x_in = np.transpose(np.array(X[n], ndmin=2))

            v_h1 = np.dot(self.w_h1, self.x_in) + self.b_h1
            y_h1 = 1 / (1 + np.exp(-v_h1))

            v_h2 = np.dot(self.w_h2, y_h1) + self.b_h2
            y_h2 = 1 / (1 + np.exp(-v_h2))

            v_out = np.dot(self.w_out, y_h2) + self.b_out

            out = 1 / (1 + np.exp(-v_out))
            print(out)
            p = list(
                np.greater_equal(np.transpose(out), 0.5)[0]).index(True) + 1
            nn_output.append(p)

        return nn_output
