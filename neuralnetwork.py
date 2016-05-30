import numpy as np
import numexpr as ne
#import cldot


class NeuralNetworkClassifier(object):
    def __init__(self,n_layers=1,layer_sizes=[2,1],l2_penalties=[0.0],step_size = 1.0,n_iter=100,descent_method = 'gradient',batch_size = 1,activation = 'sigmoid',output='log'):
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.l2_penalties = l2_penalties
        self.step_size = step_size
        self.n_iter = n_iter
        self.descent_method = descent_method
        self.batch_size = batch_size
        self.activation = activation
        self.output = output

    def fit(self,X,y,initial_theta = None,verbose = False, debug = False):
        if(X.shape[0] != y.shape[0]):
            raise ValueError
        if(self.n_layers != len(self.layer_sizes)):
            raise ValueError
        self.classes_, y = np.unique(y, return_inverse=True)
        y_class = np.zeros((y.size,self.classes_.size))
        y_class[np.arange(y.size),y] = 1
        # initalize the initial theta
        if(initial_theta == None):
            n_in = X.shape[1]
            initial_theta=[0]*self.n_layers
            for i in range(self.n_layers):
                n_hidden = self.layer_sizes[i]
                if(self.activation == 'sigmoid'):
                    top = 4*np.sqrt(6./(n_in+n_hidden))
                elif(self.activation in ['tanh','softsign']):
                    top = np.sqrt(6./(n_in+n_hidden))
                else:
                    raise ValueError
                bot = -top
                initial_theta[i] = ((np.random.random_sample((n_hidden,n_in+1))).astype('float32',casting='same_kind'))*(top - bot) + bot
                n_in = n_hidden
        if(len(self.l2_penalties)==1):
            self.l2_penalties = self.l2_penalties*self.n_layers
        self.theta_ = initial_theta
        # add the bias terms
        X_bias = np.hstack([np.ones((X.shape[0],1)),X])
        if(self.descent_method == 'gradient'):
            self.theta_ = gradient_descent(initial_theta=initial_theta,exp_output=y_class,x_input=X_bias,l2_penalties=self.l2_penalties,
                                          step_size=self.step_size,max_iterations=self.n_iter,verbose=verbose,activation=self.activation,debug=debug)
        elif(self.descent_method == 'mini_batch'):
            self.theta_ = minibatch_gradient_descent(initial_theta=initial_theta,exp_output=y_class,x_input=X_bias,l2_penalties=self.l2_penalties,
                                          step_size=self.step_size,max_iterations=self.n_iter,batch_size = self.batch_size,verbose=verbose,activation=self.activation)
        elif(self.descent_method == 'stochastic_gradient' or self.descent_method == 'SGD'):
            self.theta_ = minibatch_gradient_descent(initial_theta=initial_theta,exp_output=y_class,x_input=X_bias,l2_penalties=self.l2_penalties,
                                          step_size=self.step_size,max_iterations=self.n_iter,batch_size = 1,verbose=verbose,activation=self.activation)
        else:
            raise ValueError

        return(self)

    def predict(self,X):
        X_bias = np.hstack([np.ones((X.shape[0],1)),X])
        D = forward_prop(self.theta_,X_bias,activation = self.activation,output = self.output)[-1]
        return self.classes_[np.argmax(D,axis=1)]

    def predict_proba(self,X):
        X_bias = np.hstack([np.ones((X.shape[0],1)),X])
        D = forward_prop(self.theta_,X_bias,activation = self.activation,output = self.output)[-1]
        return D

    def get_params(self,deep=True):
        return {'n_layers':self.n_layers,'layer_sizes':self.layer_sizes,'l2_penalties':self.l2_penalties,
                'step_size':self.step_size,'n_iter':self.n_iter,'descent_method':self.descent_method,'batch_size':self.batch_size,'activation':self.activation}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def score(self,X,y):
        if(self.classes_.size == 2):
            pred = self.predict_proba(X)
            area = roc_auc(y,pred)
        else:
            pred = self.predict(X)
            area = np.sum(np.equal(pred,y))/float(X.shape[0])
        print(area,self.l2_penalties,self.n_iter,self.step_size,self.layer_sizes)
        return area


    def roc_auc(y,pred_y,step_size=0.0002):
        p_tpr = 0.0
        p_fpr = 0.0
        area = 0.0
        positive = np.sum(y)
        n_examples = y.shape[0]
        for threshold in np.arange(1.0,0.0,-step_size):
            pred_positive = np.where(pred_y[:,1]>threshold,1,0)
            true_positive = np.sum(np.equal(pred_positive,y)[y>0])
            false_positive = (np.sum(pred_positive)) - true_positive
            false_positive_rate = false_positive / float(n_examples - positive)
            true_positive_rate = true_positive / float(positive)
            area += (false_positive_rate - p_fpr) * (p_tpr + 0.5*(true_positive_rate - p_tpr))
            p_tpr = true_positive_rate
            p_fpr = false_positive_rate
        return area



def forward_prop(theta,x_input,activation='sigmoid',output='log'):
    """ calculate the output of a neural network, with input given by a numpy array x_input, and parameters given by a list of numpy arrays theta.
    It expects the x_input to _not_ include the constant bias 1."""
    # detect whether we are working with one example (i.e. 1 input vector), or multiple examples:
    if(len(x_input.shape)==1):
        x_input = np.reshape(x_input,(x_input.size,1))
    layer_values = [x_input]
    #first do all the layers
    for th in theta[:-1]:
        y = np.dot(layer_values[-1],np.transpose(th))
        if(activation=='sigmoid'):
            value = logistic(y)
        elif(activation=='tanh'):
            value = tanh(y)
        elif(activation=='softsign'):
            value = softsign(y)
        # We don't want vectors, we want 2-d arrays
        if(len(value.shape)==1):
            value = np.reshape(value,(value.size,1))
        # add the bias (the input already has the bias term)
        layer_values.append(np.hstack([np.ones((value.shape[0],1)),value]))
    if(output == 'log'):
        y = np.dot(layer_values[-1],np.transpose(theta[-1]))
        # need a last step to have the outputs between 0 and 1
        if(activation=='sigmoid'):
            value = logistic(y)
        if(activation == 'tanh'):
            value = tanh(y,a=0.5,c=0.5)
        if(activation == 'softsign'):
            value = softsign(y,a=0.5,c=0.5)
        layer_values.append(value)
    elif(output == 'softmax'):
        pass
    return layer_values



def back_prop(theta,exp_output,layer_values,activation='sigmoid'):
    predictions      = layer_values[-1]
    # We have cost_i = - (y log \hat{y}(x) + (1- y) log(1- \hat{y}(x))
    # d/dx cost_i = - (y-\hat{y}(x))/(\hat{y}(1-\hat{y})) d\hat{y}/dx
    # For different activation functions, we get slightly different expressions for the derivative of the first layer, depending on dy/dx
    if(activation == 'sigmoid'):
        # d/dx \sigma(x) = \sigma(x)(1-\sigma(x)), so the 1/(\hat{y}(1-\hat{y})) term is precisely cancelled.
        small_delta = [predictions - exp_output]
    elif(activation == 'tanh'):
        # If we set y = a*tanh(b*x) + c (where a = 1.7159, b = 2.0/3, c = 0 for intermediate layers, a=1/2, b=2.0/3, c = 1/2 for the end layer)
        # we have y' = d/dx (a * tanh(b*x) + c) = a*b*(1 - tanh(b*x)^2) = 1/a * b*(1 - (a*tanh(b*x))^2) = b/a (1-(y-c)^2)
        # If c = 1/2, we cancel the 1/(\hat{y}(1-\hat{y})) is canceled, and we are left with b/a = 2.0/3 * 2
        alpha = 1.7159
        beta = 2.0/3
        small_delta = [(predictions - exp_output)*beta*2.0]
    elif(activation == 'softsign'):
        # for y = a x/(1+|x|) + c, we have dy/dx = a (1 - |x|/(1+|x|))^2 = 1/a (1- |1/a y - c| )^2
        # if a = 1/2, c= 1/2, we get 1/(y(1-y))* dy/dx = 2*(1-y)/y if y>0.5 and 2*y/(1-y) if y< 0.5
        small_delta = [(predictions - exp_output)*2.0*np.where(predictions<0.5,predictions/(1.0-predictions),(1.0-predictions)/predictions)]
    # now run the backward propagation steps
    for nr,(th,value) in enumerate(zip(reversed(theta[1:]),reversed(layer_values[1:-1]))):
        # no 1-d vectors, so replace by arrays with height 1.
        if(len(th.shape)==1):
            th = np.reshape(th,(1,th.size))
        # $delta^{(l)} = (\theta^{(l)})^t \delta^{(l+1)} .* a^{(l)}.*(1-a^{(l)})$
        if(nr>0):
            prev_delta = small_delta[0][:,1:]
        else:
            prev_delta = small_delta[0]
        nd          = np.dot(prev_delta,th)
        # see above for the derivatives.
        if(activation == 'sigmoid'):
            new_delta   = ne.evaluate('nd*value*(1-value)')
        elif(activation =='tanh'):
            new_delta = ne.evaluate('nd*beta*(alpha-value*value/alpha)')
        elif(activation == 'softsign'):
            new_delta = ne.evaluate('nd*(1-abs(value))*(1-abs(value))')
        small_delta = [new_delta] + small_delta
    return small_delta



def derivative(theta,exp_output,layer_values,l2_penalties=[0.05],activation = 'sigmoid'):
    # first run the neural network forward and calculate the errors using back propagation
    small_deltas = back_prop(theta,exp_output,layer_values,activation = activation)
    nr_examples  = layer_values[-1].shape[0]
    # if we only have 1 l2 penalty, just apply the same to all.
    if(len(l2_penalties) == 1):
        l2_penalties = l2_penalties * len(theta)
    der = []
    # now calculate the capital deltas
    # $\Delta^{(l)} = \delta^{(l+1)} (a^{(l)})^t
    # $Der^{(l)} = 1/m \Delta^{(l)} + \lambda/m \theta'^{(l)}$, where \theta' is the matrix with the bias term set to 0.
    for nr,(value_l,s_delta_l,th) in enumerate(zip(layer_values[:-1], small_deltas,theta)):
        # force theta to be a 2-d array
        if(len(th.shape)==1):
            th = np.reshape(th,(1,th.size))
        # $\Delta^{(l)} = \delta^{(l+1)} (a^{(l)})^t
        # \delta^{(l+1)} is of size (#examples,s_{l+1} + 1), a^{(l)} is of size (#examples,s_l + 1)
        # except for the last one, because then delta had no bias term included.
        if(nr < len(theta)-1):
            big_delta = np.delete(np.dot(np.transpose(s_delta_l),value_l),0,0)
        else:
            big_delta = np.dot(np.transpose(s_delta_l),value_l)

        # replace the bias part of theta by 0's
        thp = np.zeros(th.shape)
        thp[:,1:] = th[:,1:]
        # add the regularization cost
        l2_penalty = l2_penalties[nr]
        der_l = ne.evaluate('1.0/nr_examples * big_delta + l2_penalty * thp')
        der  += [der_l]
    return der

def find_minimum(theta,exp_output,layer_values,dir_vector):
    """find the minimum from theta along a vector dir_vector"""
    x_input = layer_values[0]
    pass


def cost_unvec(theta,exp_output,predictions,l2_penalties=[0.05]):
    """ Calculate the cost function in an unvectorized way, just for checking """
    m = exp_output.shape[0]
    if(len(l2_penalties)==1):
        l2_penalties = l2_penalties * len(theta)
    j = 0
    for (output,predicted) in zip(exp_output,predictions):
        tmp = np.dot(output,np.log(predicted)) + np.dot(1-output,np.log(1-predicted))
        j += tmp
    j = -1.0/m * j

    # add the regularization
    for i,th in enumerate(theta):
        if(len(th.shape)==1):
            th = np.reshape(th,(1,th.size))
        # remove the bias term from the regularization cost
        j += l2_penalties[i]/(2.0)* np.sum( th[:,1:] **2)
    return j



def cost(theta,exp_output,pred_output,l2_penalties=[0.05]):
    """ Calculate the cost function in a vectorized manner"""
    if(len(l2_penalties)==1):
        l2_penalties = l2_penalties * len(theta)
    # the number of data points
    m           = exp_output.shape[0]
    cost1       = np.einsum('ij,ij', exp_output,np.log(pred_output))
    cost2       = np.einsum('ij,ij', 1-exp_output,np.log(1-pred_output))
    j           = -1.0/m *(cost1+cost2)
    # add the regularization
    for i,th in enumerate(theta):
        if(len(th.shape)==1):
            th = np.reshape(th,(1,th.size))
        # remove the bias term from the regularization cost
        j += l2_penalties[i]/(2.0)* np.sum( th[:,1:] **2)
    return j



def array_flatten(input_matrices):
    """ Flatten a list of input into one long vector, and return that and the shape"""
    shapes = []
    vectorized  = np.array([])
    for matrix in input_matrices:
        if(len(matrix.shape)==1):
            matrix = np.reshape(matrix,(1,matrix.size))
        shapes.append(matrix.shape)
        vectorized = np.append(vectorized,matrix)
    return (vectorized,shapes)

def array_unflatten(input_vector,shapes):
    """ return the input_vector in the original shape"""
    output_array = []
    taken = 0
    for shape in shapes:
        size   = shape[0]*shape[1]
        matrix = np.reshape(input_vector[taken:taken+size],shape)
        output_array.append(matrix)
        taken += size
    return output_array

def gradient_check(theta,exp_output,x_input,l2_penalties=[0.05],epsilon=0.01,activation='sigmoid'):
    """check whether we calculate the gradient correctly
    by approximating the derivative by (f(theta_i + epsilon) - f(theta_i - epsilon)) / 2*epsilon"""
    # transform the theta matrices into a flat vector and store their shape
    if(len(l2_penalties)==1):
        l2_penalties = l2_penalties * len(theta)
    (theta_vec,shapes) = array_flatten(theta)
    gradient_check     = np.array([0.]*len(theta_vec))
    for i in range(len(theta_vec)):
        # store the original value
        theta_i_orig = theta_vec[i]

        #calculate the cost at theta_i + epsilon
        theta_vec.put(i,theta_i_orig+epsilon)
        theta_p_i = array_unflatten(theta_vec,shapes)
        pred_p_i = forward_prop(theta_p_i,x_input,activation=activation)[-1]
        cost_p_i = cost(theta_p_i,exp_output,pred_p_i,l2_penalties)

        # calculate the cost at theta_i - epsilon
        theta_vec.put(i,theta_i_orig-epsilon)
        theta_m_i = array_unflatten(theta_vec,shapes)
        pred_m_i = forward_prop(theta_m_i,x_input,activation=activation)[-1]
        cost_m_i = cost(theta_m_i,exp_output,pred_m_i,l2_penalties)

        # calculate th difference to approximate the derivative
        gradient_check[i] = (cost_p_i - cost_m_i)/(2.0*epsilon)

        # restore the original value
        theta_vec.put(i,theta_i_orig)

    print(array_unflatten(gradient_check,shapes))
    pred = forward_prop(theta,x_input,activation=activation)
    gradient = derivative(theta,exp_output,pred,l2_penalties,activation=activation)
    (gradient_vec,gradient_shape) = array_flatten(gradient)
    print(gradient)
    print(array_unflatten(gradient_check - gradient_vec,shapes))
    return 0

def conjugate_gradient(initial_theta,exp_output,x_input,l2_penalty):
    # description of conjugate gradient:
    # we start with (arbitrary) initial vectors g_0 = h_0 = - grad(f(P_0))
    # then find the the minimum of f from P_0 along h_0
    # we then set g_1 = - grad(f(P_0)) and
    # h_1 = g_1 + np.dot(g_1,g_1) / np.dot(g_0,g_0) h_0
    # and do this again and again
    theta = initial_theta
    cur_predictions = forward_prop(theta,x_input)
    return 0



def gradient_descent(initial_theta,exp_output,x_input,l2_penalties,step_size,batch_size = 0,
                     max_iterations=100,debug=False,epsilon=0.05,validation_input = [],
                     validation_output = [],verbose = False,activation='sigmoid'):
    """ description of gradient descent:
    start at some point P_0.
    set P_1 = P_0 - step_size * grad(f(P_0))
    and do this again and again, until we hit the bottom (i.e. cost decrease less than epsilon)"""
    theta = initial_theta
    cur_predictions = forward_prop(theta,x_input)
    prev_cost = cost(theta,exp_output,cur_predictions[-1],l2_penalties)
    print("starting cost is",prev_cost)

    prev_correct = 0.0
    prev_theta = []
    for i in range(max_iterations):
        # get the current predictions and the derivative
        cur_predictions = forward_prop(theta,x_input,activation=activation)
        grad = derivative(theta,exp_output,cur_predictions,l2_penalties,activation=activation)

        # flatten the theta and the derivative so we can easily add them
        (theta_vec,shape) = array_flatten(theta)
        grad_vec = array_flatten(grad)[0]

        # update theta with the derivative
        theta_vec = theta_vec - step_size*grad_vec
        # Restore theta into the original format.
        theta = array_unflatten(theta_vec,shape)
        if(debug):
            gradient_check(theta,exp_output,x_input,l2_penalties,activation=activation)
            n_cost = cost(theta,exp_output,cur_predictions[-1],l2_penalties)
            print("Iteration",i,"cost is",n_cost)
            #if(prev_cost - n_cost < epsilon):
            #    return prev_theta
            prev_cost = n_cost
        if(verbose and i>0 and i%50==0):
            n_cost = cost(theta,exp_output,cur_predictions[-1],l2_penalties)
            if(epsilon > 1 and prev_cost - n_cost < epsilon):
                return prev_theta
            prev_cost = n_cost
            prev_theta = theta
            print("Iteration",i,"cost is",n_cost)
            if(len(validation_input)>0):
                prediction = forward_prop(theta,validation_input,activation = activation)
                correct = 0
                val_size = validation_input.shape[0]
                val_cost = cost(theta,validation_output,prediction[-1],l2_penalties)
                for j in range(val_size):
                    nr = np.argmax(prediction[-1][j])
                    if(validation_output[j,nr]==1):
                        correct +=1
                if(correct < prev_correct):
                    return prev_theta
                else:
                    print("Validation cost is",val_cost)
                    print(correct*1.0/val_size)
                    prev_correct = correct
                    prev_theta = theta
    return theta

def minibatch_gradient_descent(initial_theta,exp_output,x_input,l2_penalties,step_size,batch_size = 10,
                               max_iterations=100,validation_input = [],validation_output = [],iterations_check = 1000,verbose=False,activation = 'sigmoid'):
    """ description of gradient descent:
    start at some point P_0.
    set P_1 = P_0 - step_size * grad(f(P_0))
    and do this again and again, until we hit the bottom (i.e. cost decrease less than epsilon)"""
    theta = initial_theta
    nr_examples = x_input.shape[0]
    nr_output = exp_output.shape[1]
    b_cost = 0
    shuffled = np.hstack([exp_output,x_input])
    (theta_vec,shape) = array_flatten(theta)
    for i in range(max_iterations):
        np.random.shuffle(shuffled)
        x_input = shuffled[:,nr_output:]
        exp_output = shuffled[:,:nr_output]
        for l in range(0,nr_examples,batch_size):
            batch = x_input[l:l+batch_size]
            batch_output = exp_output[l:l+batch_size]

            # get the current predictions and the derivative
            cur_predictions = forward_prop(theta,batch,activation = activation)
            grad = derivative(theta,batch_output,cur_predictions,l2_penalties,activation = activation)

            # flatten the derivative so we can easily add it
            grad_vec = array_flatten(grad)[0]

            # update theta with the derivative
            theta_vec = theta_vec - step_size*grad_vec
            # Restore theta into the original format.
            theta = array_unflatten(theta_vec,shape)
            if(verbose):
                b_cost += cost(theta,batch_output,cur_predictions[-1],l2_penalties)
            if(verbose and (i*nr_examples//batch_size +  l//batch_size +1) %iterations_check == 0):
                print("batch cost",1.0/iterations_check*b_cost)
                b_cost = 0
        if(i>1 and i%5==0):
            if(len(validation_input)>0):
                prediction = forward_prop(theta,validation_input,activation = activation)
                correct = 0
                val_size = validation_input.shape[0]
                for j in range(val_size):
                    nr = np.argmax(prediction[-1][j])
                    if(validation_output[j]==nr):
                        correct +=1
                print("Iteration",i,"prediction succes is:",correct*1.0/val_size)
    theta_avg = np.zeros_like(theta_vec)
    for l in range(0,nr_examples,batch_size):
        batch = batch = x_input[l:l+batch_size]
        batch_output = exp_output[l:l+batch_size]
        cur_predictions = forward_prop(theta,batch,activation = activation)
        grad = derivative(theta,batch_output,cur_predictions,l2_penalties,activation = activation)
        grad_vec = array_flatten(grad)[0]
        theta_vec = theta_vec - step_size*grad_vec
        theta_avg += theta_vec
        theta = array_unflatten(theta_vec,shape)
    theta_avg = theta_avg / float(nr_examples//batch_size)
    theta = array_unflatten(theta_avg,shape)
    return theta

def logistic(z):
    return ne.evaluate('1.0/(1+exp(-z))')
    #return 1.0/(1+np.exp(-z))
    #return 1.0/(1+pyopencl.clmath.exp(-z))
    #return 1.0/(1+cldot.clexp(-z))

def tanh(z,a=1.7159,b=2.0/3,c=0.0):
    #return a*np.tanh(b*z)+c
    return ne.evaluate('a*tanh(b*z)+c')

def inv_logistic(t):
    return -1 * np.log(1.0/t - 1)

def softmax(x):
    exp = np.exp(x - x[0])
    return exp/np.sum(exp)

def softsign(z,a=1.0,b=1.0,c=0.0):
    return ne.evaluate('a*z/(1+abs(b*z)) + c')

def main():
    and_theta = np.array([-30,20,20])
    nand_theta = np.array([30,-20,-20])
    or_theta = np.array([-10,20,20])
    nor_theta = np.array([10,-20,-20])

    test_theta = [np.vstack([and_theta,and_theta]),or_theta]
    test_input = np.array([1,1])

    xor_theta = [np.vstack([or_theta, nand_theta]),and_theta]

    xor_input = np.hstack([np.ones((4,1)),np.array([[0,0],[1,0],[0,1],[1,1]])])

    xor_output = np.array([[0],[1],[1],[0]])
    print(forward_prop(xor_theta,xor_input)[-1])
    print(cost_unvec(xor_theta,xor_output,forward_prop(xor_theta,xor_input)[-1],[0.05]))
    print(cost(xor_theta,xor_output,forward_prop(xor_theta,xor_input)[-1],[0.05]))


    xor_nn = NeuralNetworkClassifier(n_layers=2,layer_sizes=[2,2], l2_penalties=[0.05,0.05],n_iter=10,step_size=1.0,activation = 'softsign')
    xor_nn = xor_nn.fit(np.array([[0,0],[1,0],[0,1],[1,1]]),np.array([[0],[1],[1],[0]]),verbose=True,debug=False)
    print(xor_nn.predict(np.array([[0,0],[1,0],[0,1],[1,1]])))
    #print(gradient_check(xor_theta,xor_output,xor_input,l2_penalties=[0.05,0.05]))
    #print(xor_theta)
    #print(gradient_descent(xor_theta,xor_output,xor_input,0.05,1.0,debug=False))
    #initial_theta =[]
    #initial_theta.append(np.random.rand(2,3)*2 - 1)
    #initial_theta.append(np.random.rand(1,3)*2 - 1)
    #found_theta = gradient_descent(initial_theta,xor_output,xor_input,0,1.0,debug=False,epsilon=0,max_iterations=1000)
    #print(forward_prop(found_theta,xor_input)[-1])
    #print(xor_output)
    #print(xor_theta)

if(__name__ == "__main__"):
    main()
