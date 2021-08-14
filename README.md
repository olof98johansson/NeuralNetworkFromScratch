<h1 align="center"> Quick explanation</h1>

Performing binary classification with the network from [NeuralNetwork.py](https://github.com/olof98johansson/NeuralNetworkFromScratch/blob/main/NeuralNetwork.py) file where an arbitrary architecture (depth and width) can be selected. The data consists of one 2D training set of <i>s<sub>t</sub> = 10000</i> data points of <i><b>x</b><sub>i</sub>&in; <b>R</b><sup>2</sup></i> for <i>i=1,...,s</i> with corresponding target labels <i>t = &pm; 1</i> evaluated on a validation set of <i>s<sub>v</sub> = 5000</i> and one 3D training set of <i>s<sub>t</sub> = 12000</i> data points with corresponding validation set <i>s<sub>v</sub>=6000</i> with same target labels. The classification error is defined as
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{C}&space;=&space;\frac{1}{2s}\sum_{\mu=1}^{s}\left|\text{sgn}\left(\mathcal{O}^{(\mu)}\right)&space;-&space;t^{(\mu)}\right|" target="_blank"><img src="https://latex.codecogs.com/png.latex?\mathcal{C}&space;=&space;\frac{1}{2s}\sum_{\mu=1}^{s}\left|\text{sgn}\left(\mathcal{O}^{(\mu)}\right)&space;-&space;t^{(\mu)}\right|" title="\mathcal{C} = \frac{1}{2s}\sum_{\mu=1}^{s}\left|\text{sgn}\left(\mathcal{O}^{(\mu)}\right) - t^{(\mu)}\right|" /></a></p>


where <i><b>0</b></i> is the output of the network and <i>s</i> the size of the dataset. Furthermore are the tanh function used as activation functions with a local field such that the output of node <i>i</i> in layer <i>l</i> for input <i>&mu;</i> is defined as
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\tiny&space;V_i^{(l,\mu)}&space;=&space;\tanh\left(\sum_{j=1}^{M_l}w_{ij}^{(l)}V_j^{(l-1,&space;\mu)}&space;-&space;\theta_i^{(l)}\right)&space;\longrightarrow&space;\mathbf{V}^{(l,\mu)}&space;=&space;\tanh\bigg(\mathbf{W}^{(l)}\mathbf{V}^{(l-1,&space;\mu)}&space;-&space;\mathbf{\Theta}^{(l)}\bigg)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\tiny&space;V_i^{(l,\mu)}&space;=&space;\tanh\left(\sum_{j=1}^{M_l}w_{ij}^{(l)}V_j^{(l-1,&space;\mu)}&space;-&space;\theta_i^{(l)}\right)&space;\longrightarrow&space;\mathbf{V}^{(l,\mu)}&space;=&space;\tanh\bigg(\mathbf{W}^{(l)}\mathbf{V}^{(l-1,&space;\mu)}&space;-&space;\mathbf{\Theta}^{(l)}\bigg)" title="\tiny V_i^{(l,\mu)} = \tanh\left(\sum_{j=1}^{M_l}w_{ij}^{(l)}V_j^{(l-1, \mu)} - \theta_i^{(l)}\right) \longrightarrow \mathbf{V}^{(l,\mu)} = \tanh\bigg(\mathbf{W}^{(l)}\mathbf{V}^{(l-1, \mu)} - \mathbf{\Theta}^{(l)}\bigg)" /></a></p>


where <i>M<sub>l</sub></i> is the number of nodes in layer <i>l</i>.


The network is trained by *stochastic gradient descent* sequential learning implying that the parameters are updated as
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\mathbf{W}^{(l)}&space;&\longleftarrow&space;\mathbf{W}^{(l)}&space;&plus;&space;\eta\mathbf{{\Delta}}^{(l)}(\mathbf{V}^{(l-1)})^T&space;\\&space;\mathbf{\Theta}^{(l)}&space;&\longleftarrow&space;\mathbf{\Theta}^{(l)}&space;-&space;\eta\mathbf{\Delta}^{(l)},&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;\mathbf{W}^{(l)}&space;&\longleftarrow&space;\mathbf{W}^{(l)}&space;&plus;&space;\eta\mathbf{{\Delta}}^{(l)}(\mathbf{V}^{(l-1)})^T&space;\\&space;\mathbf{\Theta}^{(l)}&space;&\longleftarrow&space;\mathbf{\Theta}^{(l)}&space;-&space;\eta\mathbf{\Delta}^{(l)},&space;\end{align*}" title="\begin{align*} \mathbf{W}^{(l)} &\longleftarrow \mathbf{W}^{(l)} + \eta\mathbf{{\Delta}}^{(l)}(\mathbf{V}^{(l-1)})^T \\ \mathbf{\Theta}^{(l)} &\longleftarrow \mathbf{\Theta}^{(l)} - \eta\mathbf{\Delta}^{(l)}, \end{align*}" /></a></p>

where <i>&eta;</i> is the learning rate and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{\Delta}^{(l)}&space;=&space;\left[\delta_1^{(l)},...,\delta_{M_l}^{(l)}\right]^T" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\mathbf{\Delta}^{(l)}&space;=&space;\left[\delta_1^{(l)},...,\delta_{M_l}^{(l)}\right]^T" title="\mathbf{\Delta}^{(l)} = \left[\delta_1^{(l)},...,\delta_{M_l}^{(l)}\right]^T" /></a> is the cost vector for each layer evaluated by the chain rule as

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\delta_{j}^{(l)}&space;\longleftarrow&space;\sum_{i=1}^{M_{l&plus;1}}\frac{\partial&space;V_j^{(l)}}{\partial&space;b_j^{(l)}}\delta_i^{(l&plus;1)}w_{ij}^{(l&plus;1)}," target="_blank"><img src="https://latex.codecogs.com/png.latex?\delta_{j}^{(l)}&space;\longleftarrow&space;\sum_{i=1}^{M_{l&plus;1}}\frac{\partial&space;V_j^{(l)}}{\partial&space;b_j^{(l)}}\delta_i^{(l&plus;1)}w_{ij}^{(l&plus;1)}," title="\delta_{j}^{(l)} \longleftarrow \sum_{i=1}^{M_{l+1}}\frac{\partial V_j^{(l)}}{\partial b_j^{(l)}}\delta_i^{(l+1)}w_{ij}^{(l+1)}," /></a></p>

with 

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\delta_i^{(L)}&space;\longleftarrow&space;\frac{\partial&space;\mathcal{O}_i}{\partial&space;b_i^{(L)}}\big(t_i&space;-&space;\mathcal{O}_i\big)," target="_blank"><img src="https://latex.codecogs.com/png.latex?\delta_i^{(L)}&space;\longleftarrow&space;\frac{\partial&space;\mathcal{O}_i}{\partial&space;b_i^{(L)}}\big(t_i&space;-&space;\mathcal{O}_i\big)," title="\delta_i^{(L)} \longleftarrow \frac{\partial \mathcal{O}_i}{\partial b_i^{(L)}}\big(t_i - \mathcal{O}_i\big)," /></a></p>

being the cost value for the output layer. 


Moreover, the weights are initiated with a modified [glorot uniform initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) as

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{W}^{(l)}&space;\longleftarrow&space;\mathcal{N}\big(\mu=0,&space;\sigma=1\big)\sqrt{\frac{6}{M_{l}&space;&plus;&space;M_{l&plus;1}}}," target="_blank"><img src="https://latex.codecogs.com/png.latex?\mathbf{W}^{(l)}&space;\longleftarrow&space;\mathcal{N}\big(\mu=0,&space;\sigma=1\big)\sqrt{\frac{6}{M_{l}&space;&plus;&space;M_{l&plus;1}}}," title="\mathbf{W}^{(l)} \longleftarrow \mathcal{N}\big(\mu=0, \sigma=1\big)\sqrt{\frac{6}{M_{l} + M_{l+1}}}," /></a></p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{80}&space;\mathcal{N}\big(\mu,&space;\sigma\big)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\dpi{80}&space;\mathcal{N}\big(\mu,&space;\sigma\big)" title="\mathcal{N}\big(\mu, \sigma\big)" /></a> is the univariate normal (gaussian) distribution with mean <i>&mu;</i> and variance <i>&sigma;</i> and <i>M<sub>l</sub></i>, <i>M<sub>l+1</sub></i> is the number of nodes in layers <i>l</i> and <i>l+1</i> respectively. The thresholds are initialized to zero.


<h1 align="center"> Results </h1>
<h3 align="center"> Initilization of network</h3>

Initializing the network for two hidden layers with <i>n<sub>1</sub> = n<sub>2</sub> = 5</i> hidden neurons each and training for <i>300</i> epochs with an initial learning rate of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{80}&space;\eta&space;=&space;2\cdot10^{-2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\dpi{80}&space;\eta&space;=&space;2\cdot10^{-2}" title="\eta = 2\cdot10^{-2}" /></a>. The weights are initiated with the modified glorot initializer.

<h3 align="center">2D data </h3>
The results with the above initialization and on the 2D data, which looks like

![2ddata](https://github.com/olof98johansson/NeuralNetworkFromScratch/blob/main/Images/Original_data_2d.png?raw=True)

is the following

![results2d](https://github.com/olof98johansson/NeuralNetworkFromScratch/blob/main/Images/Results_2d.png?raw=True)

And like that one can construct a custom machine learning classifier :)
