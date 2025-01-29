"""
################################################################################################################
Lab #2 - Brodie's Weight
The point of this lab is to show how cost minimization via gradient descent can find the "best" Weights for a
non-linear model.  For this lab, the model is quadratic: y=ax^2+by+c.  Since gradient descent only works on linear
models, we can change this equation to linear by considering x^2 to be an input value - let's call it Q (Q=x^2).
Then we can rewrite the model equation as y=aQ+by+c... which is linear.   Cost minimization via gradient descent
can solve this.  This lab...
    -uses Tensorflow 2 on a simple linear model having 2 input Features (x and Q) and 3 Weights (a, b, and c)
     (i.e. a 4-D model of cost).
    -describes how the Cost model used for finding the minimum is multi-dimensional.  The number of dimensions is
     the number of Features+1.
    -introduces data scaling using manually derived scaling factors to adjust the learning rates,
    -uses a plot to show how the cost declines as the number of training Epochs increases, and
    -provides an example of how to read a csv data file into a Python program.

Needed files:  utilities.py, Brodie_Weights.csv
Author: R. Bourquard - Dec 2020
################################################################################################################
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from Utilities import plot_points, plot_points_and_a_line, plot_points_on_a_surface, plot_points_and_lines



#READ THE INPUT DATA FILE
input_filename = 'Brodie_Weights.csv'
input_file = np.loadtxt(input_filename, dtype='float32', delimiter=",", skiprows=1)
print(input_file.shape)
x_train = input_file[:,0]
y_truth = input_file[:, 1]
print('x_train:', x_train)
print('y_truth:', y_truth)
nPoints = len(x_train)


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION Plot #1:  Plot the Training Examples
# ---------------------------------------------------------------------------------------------------------------
# This program compares the weight of my dog Brodie as he aged from a puppy to a young adult.  The 30 measurements
# of age (in weeks) and weight (in pounds) are contained in 2 columns of the csv file, "Brodie_Weights.csv".
#
# This plot shows the 30 Training Examples of age vs weight.  We want to find a relationship between the two so
# we can predict his weight for any given value of age.  The points don't appear to be in a straight line, so
# it's obvious a straight line equation won't fit them.
#
# Instead, we will try to fit them with a quadratic equation: y = ax**2 + bx + c.
# The equation has 2 Features (x and x**2) and 3 weights (a, b, and c).  The gradient descent procedure will
# have to determine the 3 weights by examining the 30 Training Examples.  Each Training Example has a Feature
# value (age), and a Ground Truth value (Brodie's weight).  The 'age' Feature values will need to be squared
# to form the second Feature values.
#
# Gradient descent will find the 3 weight values (a, b, and c) which best align the resulting equation's line to
# the Ground Truth values of the 30 points.
#
# Note that 'c' is called a "Bias weight" because it doesn't get multiplied by anything, it's just added to the
# output.  Most AI models have Bias weights.  In a deep neural network there will be one Bias weight for each
# layer.
plot_points('Plot #1:  Training Examples', 'Age (weeks)', 'Weight (pounds)',
            'ro|Training Examples', x_train, y_truth)
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


#SQUARE THE AGE FEATURE VALUE AND STORE IT IN THE SECOND COLUMN OF X_TRAIN
x_train_original = x_train   # save the original x_train values for a later display
x_train = np.c_[input_file[:,0], input_file[:,0]**2]   # x_train is now a (30,2) matrix



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION Plot #2:  Scale the Training Examples
# ---------------------------------------------------------------------------------------------------------------
# The variable x_train has 30 rows, one for each Training Example.  Each row has 2 columns; one for each Feature.
# The first column is Brodie's age; the second column is his age squared.  Of course, the 2nd column's values
# will be much bigger than the 1st column's values.  This difference in magnitude will make it impossible to find
# a learning_rate that will be appropriate for both.  A learning_rate appropriate for the age values will be too
# big for the squared-age values - and a learning_rate appropriate for the squared-age values will be too small
# for the age values.
#
# A way to compensate for this is to scale all the input Feature values.  A good method to use is Standardization.
# This method is applied to each Feature's values in turn, and it reduces the Feature's values so they are all
# normalized (centered on zero with a standard deviation of 1).  This means that after scaling they will mostly
# be between -1 and +1, with statistical outliers slightly outside this range.  Each Feature column is scaled
# individually across its 30 rows.
#
# Typically, after this is done, the scaling object - which contains the scale factors - is saved since it MUST
# BE APPLIED to the Feature values of every subsequent input to the model.  In this program, the scaling object
# is used again at the bottom of the code to predict weights for some additional input values.
#
# This plot shows the 30 input age Features (the red dots), and what they look like after standardized scaling (the
# blue dots are ages, and the green dots are squared-ages).  Note that they have been scaled separately, so they
# BOTH fall mostly within the -1 to +1 range.
#
# (Programming note:  The np.transpose flips the matrix so that the rows become columns and the columns become
# rows.  This transforms the matrix to match what the 'plot_points_and_lines' method requires for input.)

# SCALE x_train's VALUES
scaler_obj = StandardScaler()
scaler_obj.fit(x_train)
x_train = scaler_obj.transform(x_train)
line_labels = ["bo|x's After Scaling", "go|x^2's After Scaling"]
y_truth_to_plot = np.c_[y_truth, y_truth]
print('line_labels:', line_labels)
print('x_train:', x_train)
print('y_truth_to_plot:', y_truth_to_plot)
plot_points_and_lines('Plot #2:  The Training Examples before and after Scaling', 'age (weeks)', 'weight (pounds)',
                      'ro|Original Training Examples', x_train_original, y_truth,
                      line_labels,np.transpose(x_train),np.transpose(y_truth_to_plot))
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::




#SET INITIAL VALUES FOR a, b AND c
a = tf.Variable(0., shape=(), dtype='float32')
b = tf.Variable(0., shape=(), dtype='float32')
c = tf.Variable(0., shape=(), dtype='float32')



# ROUTINES THAT ARE NEEDED FOR COMPUTATIONS
# ******************************************************************
#PREDICTIONS
#function to predict y, given x
def predict_y_value(x):
    y = a * x[:,1] + b * x[:,0] + c  # y = ax**2 + bx + c  (Note that x[:,1] will contain the x**2 values)
    return y

#COST OF ERRORS
#Sum of squares of differences between predicted and true y values
def squared_error(prediction_values, truth_values):
    errors = tf.subtract(prediction_values, truth_values)
    return tf.reduce_mean(tf.square(errors))
# ******************************************************************



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION Plot #3:  Hypothetical Graphic of the Cost
# ---------------------------------------------------------------------------------------------------------------
# The Cost depends on 3 parameters (a, b, and c).  Therefore, a plot to show this would require 4
# dimensions: 'a', 'b', 'c' on the x, y, and z-axes, and the Cost at that point on some 4th dimension!  This
# can't be plotted!  However, you can imagine the possible Cost values for 'a', 'b', and 'c' as forming a 3-D solid,
# where the Cost at each 3-D point is represented by a color.  Assuming the Cost increases uniformly from the
# lowest-Cost point, the color gradations would form the concentric shells of a sphere.  The sphere would be made
# up of concentric layers (concentric shells) whose center would be the point of lowest Cost.  As you proceed out
# from the center in any direction, the Cost would increase and the colors would change.  That means the
# objective of this AI program would be to use gradient descent to find the 'a', 'b', and 'c' coordinates of the
# sphere's center.  (Plot #3 is actually a diagram of the Earth's concentric interior layers, but use your
# imagination to view it as a "Cost sphere".)
#
# For this AI program, the gradient descent procedure will always step from a given measurement point towards
# lower Cost (that is, towards the sphere's center) with each Epoch.  This will be the direction perpendicular to
# the shell surfaces.  The red x's illustrate this for a succession of Epochs.  (The x's don't go all the way to
# the sphere's center because I got tired of drawing them.)
#
# This model has 3 weights, and its Cost plot can be visualized with colors using 3 dimensions.  Consider that if
# an AI model has 4, 10, or even 10,000 weights (not unusual for neural networks), the resulting Cost
# 'hyper-sphere' will have 4, 10, or 10,000 dimensions!  Such a hyper-sphere is impossible to visualize, but
# the gradient descent math will still work!
photo = plt.imread('earth2.jpg')
print(photo.shape)
plt.title('Plot #3:  How the Cost varies in a 3D space (3 weights)')
plt.imshow(photo)
plt.show()
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION Plot #4:  Show the Costs for one slice through the sphere
# ---------------------------------------------------------------------------------------------------------------
# Let's look at a horizontal slice through the sphere.  The planar surface defined by the slice will show the
# various colors of the sphere's layers which intersect it.  (Plot #4 is also a diagram of the Earth's concentric
# interior layers, but use your imagination to view it as a "Cost sphere" that has been split open.)
photo = plt.imread('split_earth.jpg')
print(photo.shape)
plt.title('Plot #4:  Planer surface cutting through the Cost Sphere')
plt.imshow(photo)
plt.show()



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION Plot #5:  Show the actual Costs for one slice through the sphere
# ---------------------------------------------------------------------------------------------------------------
# This plot is as if we cut the actual Cost sphere for the Brodie Weight problem with a horizontal plane (the plane
# defined by c=45 - which I know to be the plane that passes through the center of the sphere).  Thus the plane's
# 2 axes are 'a' and 'b'.  It shows the inner structure of the actual Cost sphere for this problem.  And what we
# actually see is that the lowest Cost point should have an 'a' value of about -50 and a 'b' value of about 60.
#
# To "slice" through the Cost sphere at different levels, try setting 'c.assign(45)' to different values.
a_3Dgrid, b_3Dgrid, _ = np.meshgrid(np.linspace(-100.0,100.0,50), np.linspace(-100.0,100.0,50),
                                    np.linspace(1,1,nPoints), indexing='xy')
c.assign(45)   # Look at the Cost for a constant c value of 45
cost_grid = np.mean(np.square((a_3Dgrid * (x_train[:,1]) + b_3Dgrid * x_train[:,0] + c) - y_truth),axis=2)
# Generate a 2-D plot of the Costs vs m & b, where the Cost is represented by contours.
plot_points_on_a_surface('Plot #5:  Contour Plot of Cost vs a and b (for c=45)', 'a', 'b',
                         '', 0., 0., a_3Dgrid[1,:,1], b_3Dgrid[:,1,1], cost_grid)
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



# Create needed arrays
iterations = []
costs = []
count_save = []
cost_save = []
a_save = []
b_save = []
c_save = []

print('x_train:',x_train)
print('y_truth:',y_truth)
# COMPUTE THE INITIAL COST
a.assign(0)
b.assign(0)
c.assign(0)
predictions = predict_y_value(x_train)
print('predictions:', predictions)
for value in predictions:
    print('predictions values:', value)
cost = squared_error(predictions, y_truth)
print('Initial cost:', cost.numpy())





# BACKWARD PROPAGATION
# Backward Propagation requires:
#   1) a cost function = squared_error()
#   2) an optimizer = tape.gradient
#   3) a value for the learning rate = learning_rate
learning_rate = 0.5
nEpochs = 250
print_step_increment = 10
for i in range(nEpochs):
    # compute the gradients
    with tf.GradientTape() as tape:
        predictions = predict_y_value(x_train)
        cost = squared_error(predictions,y_truth)
    gradients = tape.gradient(cost, [a,b,c])
    iterations.append(i)
    costs.append(cost.numpy())

    # print what's happening
    if(i % print_step_increment) == 0:
        print('Epoch %d: a=%f, b=%f,  c=%f,  Cost %f' % (i, a.numpy(), b.numpy(), c.numpy(), cost.numpy()))
        count_save.append(i)
        cost_save.append(cost.numpy())
        a_save.append(a.numpy())
        b_save.append(b.numpy())
        c_save.append(c.numpy())

    # update the slope and constant
    a.assign_sub(gradients[0] * learning_rate)  # a = a - (a_gradient * learning_rate)
    b.assign_sub(gradients[1] * learning_rate)  # b = b - (b_gradient * learning_rate)
    c.assign_sub(gradients[2] * learning_rate)  # c = c - (c_gradient * learning_rate)


# display the trained values for m and b
print('FINAL VALUES AFTER %d STEPS:  a=%f, b=%f, c=%f, cost=%f' % (nEpochs, a.numpy(), b.numpy(), c.numpy(),
                                                                   cost.numpy()))





# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION Plot #6:  Plot of the Cost vs the number of Epochs (iterations)
# ---------------------------------------------------------------------------------------------------------------
# With more Epochs (iterations of the gradient descent procedure), the Cost gets lower and lower - meaning the
# weights provide a better and better fit.   Eventually, hopefully, it gets so low that it's not productive to
# compute more Epochs.
# This can also be seen in the screen print of the Cost values.
plot_points_and_a_line('Plot #6:  Costs = f(Epochs)', 'Epochs (Iterations)', 'Cost',
                       '', 0, 0, 'b-|Cost', count_save,
                       cost_save)
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION Plot #5:  Plot some predictions for new age values
# ---------------------------------------------------------------------------------------------------------------
# Now that we have found "best" values for the weights 'a', 'b', and 'c', we can use the method, 'predict_y_value',
# to easily predict the weights for a bunch of new ages.  These predictions form the dotted green curve.
#
# Notice that the bunch of new ages was first formed into an (n,2) matrix, then scaled using the scaling object
# we created at the beginning of this program, and finally input to the 'predict_y_value' method to predict the
# weights.  When you have scaled the Training Examples, it becomes necessary to use that same object to identically
# scale any subsequent inputs.
#
# As you look at the plot, 3 questions should come to mind that are important concepts to consider for *ALL* AI
# applications...

#   1) Was a quadratic equation the best algorithm to use in our model for matching the Training Examples?

#   2) The gradient descent Cost equation will fit best where there are a lot of Training Examples.  Are the
#        examples that were used evenly distributed?

#   3) The 30 Training Examples span the ages of 8-64 weeks.  Can any predictions outside these ages be assumed
#        dependable?
#
# It's valuable to recognize these concepts now, because as AI models (algorithms and weights) become more complex
# the same 3 limitations will exist, but will be much harder to discern.  This is why building successful AI models
# is very intuitive, and essentially an "art form".
new_ages = []
x_values = np.empty((0,2))
predicted_weights = []
min_age = 4
max_age = 71
for iAge in range(min_age, max_age):
    new_ages.append(iAge)
    x_values = np.append(x_values, np.array([[iAge, iAge ** 2]]), axis=0)
x_values = scaler_obj.transform(x_values)
predicted_weights = predict_y_value(x_values).numpy()
plot_points_and_a_line('Plot #5:  Predicted Weights for some New Ages (the green line)', 'Age (weeks)',
                       'Weight (pounds)', 'ro|Training Examples', x_train_original, y_truth,
                       'g:|Predicted Weights', new_ages, predicted_weights)
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
