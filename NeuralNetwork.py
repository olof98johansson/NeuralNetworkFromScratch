import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import random


class Logger(object):
    def __init__(self, epochs=None):

        self.init_time = time.time()
        self.epochs = epochs

    def update(self, epoch, error, val_error=None):
        elapsed_time = time.time() - self.init_time
        tim = 'seconds'
        if elapsed_time > 60 and elapsed_time <= 3600:
            elapsed_time /= 60
            tim = 'minutes'
        elif elapsed_time > 3600:
            elapsed_time /= 3600
            tim = 'hours'
        elapsed_time = format(elapsed_time, '.2f')
        if val_error == None:
            print(f'Elapsed time: {elapsed_time} {tim}\tEpoch: {epoch}/{self.epochs}\tTraining error: {error:.3f}')
        else:
            print(
                f'Elapsed time: {elapsed_time} {tim}\tEpoch: {epoch}/{self.epochs}\tTraining error: {error:.3f}\tValidation error: {val_error:.3f}')


class Network(object):
    def __init__(self, nodes, input_size, epochs=10, learning_rate=0.1, init_type='glorot'):

        self.nr_hidden = len(nodes)
        self.nodes = nodes  # declaring the hidden layers
        self.input_size = input_size
        self.nodes.insert(0, self.input_size)  # adding input layer
        self.nodes.append(1)  # adding output layer
        if init_type == 'glorot':
            self.parameters = self.initialize_parameters_mod_glor(self.nodes)
        elif init_type == 'uniform':
            self.parameters = self.initialize_parameters_uniform(self.nodes)
        else:
            print('Invalid initialization type! Choose "glorot" or "uniform"...')
            return
        self.epochs = epochs
        self.feature_vectors = {}
        self.layer_errors = {}
        self.learning_rate = learning_rate
        self.predict_outputs = {}

    # Modified glorot normal initialization of parameters
    def initialize_parameters_mod_glor(self, layers_dims):
        parameters = {}
        L = len(layers_dims) - 1

        for l in range(1, L + 1):
            factor = np.sqrt(6 / (layers_dims[l] + layers_dims[l - 1]))
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * factor
            parameters['t' + str(l)] = np.zeros(shape=(layers_dims[l], 1))

        return parameters

    # Uniform intialization of parameters
    def initialize_parameters_uniform(self, layers_dims):
        parameters = {}
        L = len(layers_dims) - 1

        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.random.uniform(-1, 1, size=(layers_dims[l], layers_dims[l - 1]))
            parameters['t' + str(l)] = np.zeros(shape=(layers_dims[l], 1))

        return parameters

    # Linear activation function for the layers
    def activation_function(self, layer, inputs):
        return np.tanh(- self.parameters['t' + str(layer)] + (self.parameters['W' + str(layer)] @ inputs))

    # Forward propagation for training
    def forward_propagation(self, inputs):
        self.feature_vectors['V0'] = inputs
        for l in range(self.nr_hidden + 1):
            V = self.activation_function(l + 1, self.feature_vectors['V' + str(l)])
            self.feature_vectors['V' + str(l + 1)] = V
        self.feature_vectors['V' + str(self.nr_hidden + 1)] = self.feature_vectors['V' + str(self.nr_hidden + 1)][0]

    # Forward propagation for validation and prediction
    def predict(self, inputs):
        self.predict_outputs['P0'] = inputs
        for l in range(self.nr_hidden + 1):
            P = self.activation_function(l + 1, self.predict_outputs['P' + str(l)])
            self.predict_outputs['P' + str(l + 1)] = P
        self.predict_outputs['P' + str(self.nr_hidden + 1)] = self.predict_outputs['P' + str(self.nr_hidden + 1)][0]

    def classification_error(self, O, t):
        C = 1 / 2 * np.abs(t - O)
        return C

    # Backpropagation -> compute errors and update weights and thresholds
    def backpropagation(self, lab):
        self.layer_errors[str(self.nr_hidden + 1)] = \
            (1 - self.activation_function(self.nr_hidden + 1, self.feature_vectors['V' + str(self.nr_hidden)]) ** 2) * \
            (lab - self.feature_vectors['V' + str(self.nr_hidden + 1)][0])

        for l in reversed(range(self.nr_hidden + 1)):
            if l > 0:
                dV = (1 - self.activation_function(l, self.feature_vectors['V' + str(l - 1)]) ** 2)
                self.layer_errors[str(l)] = dV * (self.parameters['W' + str(l + 1)].T @ self.layer_errors[str(l + 1)])

        for l in range(self.nr_hidden + 1):
            delta = self.layer_errors[str(l + 1)]

            self.parameters['W' + str(l + 1)] += delta @ self.feature_vectors['V' + str(l)].T * \
                                                 self.learning_rate

            self.parameters['t' + str(l + 1)] -= self.layer_errors[str(l + 1)] * \
                                                 self.learning_rate

    # Training loop
    def train(self, inputs, labels, val_inputs=None, val_labels=None, validate=False):

        errors, val_errors = [], []
        logger = Logger(epochs=self.epochs)
        for epoch in range(self.epochs):
            idx = np.arange(len(labels))
            random.shuffle(idx)
            error_epoch = []

            if (epoch + 1) % 100 == 0:
                self.learning_rate /= 2

            for it, i in enumerate(idx):
                inp = inputs[i]
                lab = labels[i]
                inp = np.expand_dims(inp, axis=1)

                # Feed input to network
                self.forward_propagation(inp)

                output = self.feature_vectors['V' + str(self.nr_hidden + 1)]
                self.pred = np.sign(output if output != 0.0 else 1.0)

                # Compute training error
                e = self.classification_error(self.pred, lab)
                error_epoch.append(e)

                # Update weights and thresholds
                self.backpropagation(lab)

            errors.append(np.mean(error_epoch))

            # End of epoch validation
            if validate:
                val_epoch_errors = []
                for i in range(len(val_labels)):
                    val_inp = val_inputs[i]
                    val_lab = val_labels[i]
                    val_inp = np.expand_dims(val_inp, axis=1)

                    self.predict(val_inp)
                    val_output = self.predict_outputs['P' + str(self.nr_hidden + 1)]
                    val_pred = np.sign(val_output if val_output != 0.0 else 1.0)

                    e_val = self.classification_error(val_pred, val_lab)

                    val_epoch_errors.append(e_val)
                val_errors.append(np.mean(val_epoch_errors))

                # Log training update
                if epoch % 10 == 0:
                    logger.update(epoch, np.mean(error_epoch), np.mean(val_epoch_errors))
            else:
                if epoch % 10 == 0:
                    logger.update(epoch, np.mean(error_epoch))

        return errors, val_errors if validate else errors

    # Validation loop
    def validate(self, val_inputs, val_labels):
        pred_labels = []
        for i in range(len(val_labels)):
            val_inp = val_inputs[i]
            val_inp = np.expand_dims(val_inp, axis=1)
            self.predict(val_inp)
            val_output = self.predict_outputs['P' + str(self.nr_hidden + 1)]
            val_pred = np.sign(val_output if val_output != 0.0 else 1.0)
            pred_labels.append(val_pred)

        c_error = (1 / (2 * len(val_labels))) * sum(np.abs(np.squeeze(pred_labels) - val_labels))
        pred_labels = [pl[0] for pl in pred_labels]
        if self.input_size == 2:
            x1 = [row[0] for row in val_inputs]
            x2 = [row[1] for row in val_inputs]

            fig = px.scatter(x=x1, y=x2,
                             color=np.array(pred_labels),
                             opacity=0.7,
                             color_continuous_scale=px.colors.sequential.Plotly3)
            fig.update(layout_coloraxis_showscale=False)

            fig.update_traces(marker=dict(size=12,
                                          colorscale='PuBu',
                                          showscale=False),
                              selector=dict(mode='markers'))
            fig.update_layout(
                title={
                    'text': f"Predictions with classification error: {c_error:.3f}",
                    'font': dict(size=22),
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            fig.show('png')
        elif self.input_size == 3:
            x1 = [row[0] for row in val_inputs]
            x2 = [row[1] for row in val_inputs]
            x3 = [row[2] for row in val_inputs]

            fig = px.scatter_3d(x=x1, y=x2, z=x3,
                                color=np.array(pred_labels),
                                opacity=0.7,
                                color_continuous_scale=px.colors.sequential.Plotly3)
            fig.update(layout_coloraxis_showscale=False)

            fig.update_traces(marker=dict(size=9,
                                          colorscale='PuBu',
                                          showscale=False),
                              selector=dict(mode='markers'))
            fig.update_layout(
                title={
                    'text': f"Predictions with classification error: {c_error:.3f}",
                    'font': dict(size=22),
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            fig.show('png')
        else:
            print(f'Classification error for the validation set evaluated to: {c_error:.3f}')

    # Visualization of training and/or validation progress
    def plot_errors(self, training_errors, validation_errors=None, validate=False):

        x = np.arange(len(training_errors))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                mode='lines+markers',
                x=x,
                y=training_errors,
                marker=dict(
                    color='LightSkyBlue',
                    size=10,
                    opacity=0.9,
                    line=dict(
                        color='MediumPurple',
                        width=1
                    )
                ),
                line=dict(
                    color='MediumPurple',
                    width=2,

                ),
                name='$\mathcal{C}_{training}$',
                showlegend=True
            )
        )
        if validate:
            fig.add_trace(
                go.Scatter(
                    mode='lines+markers',
                    x=x,
                    y=validation_errors,
                    marker=dict(
                        symbol='star',
                        color='rgb(236, 138, 236)',
                        size=10,
                        opacity=0.9,
                        line=dict(
                            color='MediumPurple',
                            width=1
                        )
                    ),
                    line=dict(
                        color='MediumPurple',
                        width=2,

                    ),
                    name='$\mathcal{C}_{validation}$',
                    showlegend=True
                )
            )

        fig.update_layout(title=f'Classification error for {self.epochs} epochs',
                          xaxis_title='Epochs',
                          font=dict(family="Algerian", size=20))
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.98,
            bordercolor="Black",
            bgcolor='rgba(198, 207, 255, 0.4)',
            borderwidth=1
        ))

        fig.show('png')
