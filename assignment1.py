# Common conventions wherever not otherwise stated =>
# arr1 -> An array containing the input values of the training examples
# arr2 -> An array containing the expected output values of the training examples
# Theta -> Array containing parameters of the curve

import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from numpy import *
from sympy import var
from sympy import Eq
from sympy.solvers import solve
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig2 = plt.figure(2)
ax = fig.gca(projection='3d')


# COMMON FUNCTIONS ALONG ALL THE QUESTIONS
# ----------------------------------------
# Reading the input file
def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    arr = []
    for row in reader:
    	for i in range(len(row)):
    		row[i] = float(row[i])
    	arr = arr + [row]
    return arr

# ----------------------------------------



# QUESTION 1
# ----------

# Evaluate the linear function h(), given the values of the parameters Theta and the input array.
def gety_val(x, theta):
	ans_arr = np.dot(theta.transpose(), x)
	return ans_arr

# Get the value of the error function J(), error metric function for least squares
# J = 1/2* ( sum_over_all (y - h(x))^2 ) 
def get_error(arr1, arr2, theta):
	temp_term = arr2 - gety_val(arr1, theta)
	sum_arr = np.dot(temp_term, temp_term.transpose())
	sum = np.sum(sum_arr)
	return sum/2

# Get the derivative of J() wrt the j-th parameter
def get_slope_par(arr1, arr2, theta):
	temp_term = np.dot((arr2 - gety_val(arr1, theta)), arr1.transpose())
	temp_term = -1*temp_term
	return temp_term

# Implements Linear Regression
def lin_reg(arr1, arr2, l_rate, err_thres=0.0000001):
	ext = np.zeros([len(arr1), 1])
	for i in range(0, len(arr1)):
		ext[i] = 1
	# Addition of the intercept term
	arr1 = np.append(arr1, ext, axis=1)
	arr1 = np.transpose(arr1)
	arr2 = np.transpose(arr2)
	last_error = 1000
	new_error = 0
	# Initialising the value of 'theta' to zero
	theta = np.zeros([len(arr1), 1])
	ite = 0
	batch_error = 0
	# Stopping criteria -> When change in error is less than a certain error threshold
	while(abs(last_error - new_error)>err_thres):
		last_error = new_error
		diff = l_rate*get_slope_par(arr1, arr2, theta)
		diff = diff.transpose()
		# Updating the value of theta
		theta = theta - diff
		ite += 1
		new_error = get_error(arr1, arr2, theta)
		x = theta[1]
		y = theta[0]
		z = new_error
		# Contour and 3-d mesh plotting
		ax.scatter(x, y, z, color='r')
		plt.scatter(x, y, z, color ='r')
		plt.pause(0.02)

		# Will be executed in case of divergence
		# Either the number of iterations are too many (too low error threshold criteria)
		# Or the error is increasing with every iteration (too big learning rate)
		if ite%20==0:
			if new_error - batch_error > 1000:
				return theta, ite, new_error, False
		if ite>1000:
			return theta, ite, new_error, False
	return theta, ite, new_error, True

# Plots Linear Regression
def plot_lin_reg(arr1, arr2, l_rate, err_thres):

	# Creates the 3-d mesh
	theta1 = np.arange(-0.15,1.8,0.01)
	theta2 = np.arange(-0.7,0.7,0.01)
	theta1, theta2 = np.meshgrid(theta1, theta2)
	for i in range(len(arr2)):
	    if i==0:
	        Z = (theta1 + theta2 * arr1[i][0] - arr2[i][0])**2
	    else:
	        Z += (theta1 + theta2 * arr1[i][0] - arr2[i][0])**2
	Z = Z /2.0


	surf = ax.plot_wireframe(theta1, theta2, Z,linewidth=0.5)
	surf2 = plt.contour(theta1, theta2, Z)
	plt.ion()

	# Calls the linear regression implementation function
	theta, ite, fin_err, conv = lin_reg(arr1, arr2, l_rate, err_thres)
	plt.pause(2)
	if not(conv):
		print("The algorithm did not converge at the given learning rate and error threshold.")
		return False
	print("Error Threshold :", err_thres)
	print("Learning Rate : ", l_rate)
	print("Theta :" , theta)
	print("Num of Iterations: ", ite)
	print("Final Error :", fin_err)

	x = np.linspace(-2, 6, 1000)
	y = theta[0]*x + theta[1]
	plt.plot(x, y)
	plt.plot(arr1, arr2, 'ro', color='r')
	plt.savefig('lin_reg')
	plt.close()
	return True

def thr_switch(argument):
	return{
        '1': 1e-2,
        '2': 1e-5,
        '3': 1e-7,
        '4': 1e-10,
        '5': 1e-16
    }.get(argument, 1e-7)

def lr_switch(argument):
	return{
        '1': 0.001,
        '2': 0.005,
        '3': 0.009,
        '4': 0.013,
        '5': 0.017,
        '6': 0.021,
        '7': 0.025
    }.get(argument, 0.009)

# Main function implementing Question 1
def question1():

	csv_input = "Assignment_1_datasets/linearX.csv"
	with open(csv_input, "r") as f_obj:
	    arr1 = csv_reader(f_obj)

	csv_output = "Assignment_1_datasets/linearY.csv"
	with open(csv_output, "r") as f_obj:
	    arr2 = csv_reader(f_obj)

	# Normalisation
	mean = np.mean(arr1)
	var = (np.var(arr1))**(0.5)
	arr1 = np.transpose(arr1)
	arr1 = (arr1 - mean)/var
	arr1 = np.transpose(arr1)

	print("Error Threshold : \n1 -> 1e-2\n2 -> 1e-5\n3 -> 1e-7\n4 -> 1e-10\n5 -> 1e-16\n")
	th = input("Choose error Threshold :")

	print("Learning Rate : \n1 -> 0.001\n2 -> 0.005\n3 -> 0.009\n4 -> 0.013\n5 -> 0.017\n6 -> 0.021\n7 -> 0.025\n")
	lr = input("Choose learning rate :")

	print("\n\n\n")
	boole = plot_lin_reg(arr1, arr2, lr_switch(lr), thr_switch(th))

	if boole:
		print("\n\nThe plot is stored in the file named <lin_reg>")

# ----------





# QUESTION 2
# ----------

# Evaluate the weighted function h(), given the values of the parameters Theta and the input array.
def gety_val_wei(x, theta, i):
	ans_arr = np.dot(theta.transpose(), x)
	return ans_arr[i]

# Evaluate the sigmoid function, given the values of the parameters Theta and the input array.
def gety_val_exp(x, theta):
	mid_ans = np.dot(theta.transpose(), x)
	mid_ans = -1*mid_ans
	exp_ans = np.exp(mid_ans)
	ans_arr = np.power((1 + exp_ans), -1)
	return ans_arr

# Evaluate the derivative of the sigmoid function, given the values of the parameters Theta and the input array.
def get_derivative(x, theta):
	col = gety_val_exp(x, theta)
	suple = 1 - col
	diag1 = np.diag(col[0])
	diag2 = np.diag(suple[0])
	ans = np.dot(diag1, diag2)
	return ans

# Exponential value calculator for locally weighted linear regression
def exp_calc(x_m, x_c, band):
	power = ((x_m - x_c)**2)/(2*(band**2))
	return math.exp(-1*power)

# Remembering the weights for the input values of x, in locally weighted linear regression
def create_weight(arr, band, j):
	main_x = arr[0][j]
	weights = np.zeros([len(arr[0]), len(arr[0])])
	for i in range(len(arr[0])):
		weights[i][i] = exp_calc(main_x, arr[0][i], band)
	return weights

# Using normal eq to calculate parameters for unweighted linear regression
def normal_eq(arr1, arr2):
	ext = np.zeros([len(arr1), 1])
	for i in range(0, len(arr1)):
		ext[i] = 1
	arr1 = np.append(arr1, ext, axis=1)
	arr1 = np.transpose(arr1)
	arr2 = np.transpose(arr2)
	inverse = np.linalg.inv(np.dot(arr1, arr1.transpose()))
	left = np.dot(arr2, arr1.transpose())
	theta = np.dot(left, inverse)
	return theta.transpose()

# Using normal eq to calculate parameters for weighted linear regression
def normal_eq_wei(arr1, arr2, band):
	ext = np.zeros([len(arr1), 1])
	for i in range(0, len(arr1)):
		ext[i] = 1
	arr1 = np.append(arr1, ext, axis=1)
	arr1 = np.transpose(arr1)
	arr2 = np.transpose(arr2)
	final_theta = np.empty([0, 2])
	for i in range(len(arr1[0])):
		weights = create_weight(arr1, band, i)
		mid_term = np.dot(arr1, weights)
		inverse = np.linalg.inv(np.dot(mid_term, arr1.transpose()))
		mid_term_2 = np.dot(arr2, weights)
		left = np.dot(mid_term_2, arr1.transpose())
		theta = np.dot(left, inverse)
		final_theta = np.concatenate((final_theta, theta), axis = 0)
	return final_theta.transpose()

# Plotting unweighted normal eq linear regression and weighted linear regression
def plot_weighted_reg(arr1, arr2, bp):
	theta = normal_eq(arr1, arr2)

	print("Normal Equation for unweighted Linear Regression -> Value of theta :", theta)

	plt.plot(arr1, arr2, 'ro')
	x = arr1
	y = np.zeros([len(arr1)])
	for i in range(len(arr1)):
		y[i] = gety_val(np.array([arr1[i][0], 1]), theta)
	plt.plot(x, y, color='g')
	plt.savefig('normal_eq_unweighted_lin_reg')
	plt.close()


	theta = normal_eq_wei(arr1, arr2, bp)
	print("BandWidth Parameter :", bp)

	plt.plot(arr1, arr2, 'ro')
	x = arr1
	y = np.zeros([len(arr1)])
	for i in range(len(arr1)):
		y[i] = gety_val_wei(np.array([arr1[i][0], 1]), theta, i)
	plt.plot(x, y, color='g')
	plt.savefig('normal_eq_weighted_lin_reg')
	plt.close()

def min_index(arr, allowed):
	min = 1000
	min_ind = -1
	for i in range(len(arr)):
		if(arr[i][0]<min and allowed[i][0]==0):
			min = arr[i][0]
			min_ind = i
	return min_ind

def sort_arr(arr1, arr2):
	ans1 = np.zeros([len(arr1), 1])
	ans2 = np.zeros([len(arr1), 1])
	allow = np.zeros([len(arr1), 1])
	for i in range(len(arr1)):
		min_ind = min_index(arr1, allow)
		allow[min_ind][0] = 1
		ans1[i][0] = arr1[min_ind][0]
		ans2[i][0] = arr2[min_ind][0]
	return ans1, ans2

def bp_switch(argument):
	return{
        '1': 0.1,
        '2': 0.3,
        '3': 0.8,
        '4': 2,
        '5': 10
    }.get(argument, 0.8)

# Main function implementing Question 2
def question2():

	csv_input = "Assignment_1_datasets/weightedX.csv"
	with open(csv_input, "r") as f_obj:
	    arr1 = csv_reader(f_obj)

	csv_output = "Assignment_1_datasets/weightedY.csv"
	with open(csv_output, "r") as f_obj:
	    arr2 = csv_reader(f_obj)

	# Normalisation
	mean = np.mean(arr1)
	var = (np.var(arr1))**(0.5)
	arr1 = np.transpose(arr1)
	arr1 = (arr1 - mean)/var
	arr1 = np.transpose(arr1)

	arr1, arr2 = sort_arr(arr1, arr2)

	print("Bandwidth Parameter : \n1 -> 0.1\n2 -> 0.3\n3 -> 0.8\n4 -> 2\n5 -> 10\n")
	bp = input("Choose Bandwidth Parameter :")

	print("\n\n\n")
	plot_weighted_reg(arr1, arr2, bp_switch(bp))

	print("\n\nThe plot of Unweighted Linear Regression is stored in the file named <normal_eq_unweighted_lin_reg> and the plot of Weighted Linear Regression is stored in the file named <normal_eq_weighted_lin_reg>")

# ----------





# QUESTION 3
# ----------

# Logistic Regression implementation
def log_reg(arr1, arr2, err_thres):
	ext = np.zeros([len(arr1), 1])
	for i in range(0, len(arr1)):
		ext[i] = 1
	arr1 = np.append(arr1, ext, axis=1)
	arr1 = np.transpose(arr1)
	arr2 = np.transpose(arr2)
	theta = np.zeros([len(arr1), 1])
	ite = 0
	last_deriv = np.array([1])
	last_theta = np.array([1000])
	new_theta = np.array([0])
	diff = [[1], [1], [1]]
	while(diff[0][0] > err_thres or diff[1][0] > err_thres or diff[2][0] > err_thres):
		last_theta = new_theta
		spec = arr2 - gety_val_exp(arr1, theta)
		grad_log = np.dot(spec, arr1.transpose())
		deriv = -1*get_derivative(arr1, theta)
		if(not deriv.any()):
			break
		hess = np.dot(np.dot(arr1, deriv), arr1.transpose())
		hess_inv = np.linalg.inv(hess)
		change = np.dot(grad_log, hess_inv)
		theta = theta - change.transpose()
		last_deriv = deriv
		ite += 1
		new_theta = theta
		diff = abs(last_theta - new_theta)
	return theta, ite

# Main function for implementing Question 3
def question3():
	csv_input = "Assignment_1_datasets/logisticX.csv"
	with open(csv_input, "r") as f_obj:
	    arr1 = csv_reader(f_obj)

	csv_output = "Assignment_1_datasets/logisticY.csv"
	with open(csv_output, "r") as f_obj:
	    arr2 = csv_reader(f_obj)

	mean = np.mean(arr1)
	var = (np.var(arr1))**(0.5)
	arr1 = np.transpose(arr1)
	arr1 = (arr1 - mean)/var
	arr1 = np.transpose(arr1)

	theta, ite = log_reg(arr1, arr2, 1e-16)
	print("Value of theta :", theta)
	print("Number of Iterations :", ite)

	arr1 = arr1.transpose()
	for i in range(len(arr1[0])):
		if arr2[i][0] == 0:
			plt.plot(arr1[0][i], arr1[1][i], 'ro', color='b')
		else:
			plt.plot(arr1[0][i], arr1[1][i], 'ro', color='r')

	x = np.linspace(-3, 3, 1000)
	y = -1*(theta[1][0]*x + theta[2][0])/theta[0][0]
	plt.plot(x, y, color='g')
	plt.savefig('log_regression')
	plt.close()

	print("\n\nThe plot is stored in the file named <log_regression>")

# ----------






# QUESTION 4
# ----------

# Create the quadratic equation for the given values of u0, u1, sigma0, sigma1 and the value of x1, x2
def quadratic(me0, me1, sigma0, sigma1, x):
	inv_sigma0 = np.multiply(-0.5, np.linalg.inv(sigma0))
	inv_sigma1 = np.multiply(-0.5, np.linalg.inv(sigma1))
	A = np.dot((np.dot(x, np.add(inv_sigma0, np.multiply(-1, inv_sigma1)))), np.transpose(x))
	btemp1 = np.multiply(-2, np.dot(np.transpose(me0), inv_sigma0))
	btemp2 = np.multiply(2, np.dot(np.transpose(me1), inv_sigma1))
	B = np.dot(np.add(btemp1, btemp2), np.transpose(x))
	C = np.dot(np.dot(np.transpose(me0), inv_sigma0), me0) - np.dot(np.dot(np.transpose(me1), inv_sigma1), me1)
	C = C + 0.5*math.log(np.linalg.det(sigma1)/np.linalg.det(sigma0))
	return A + B + C

# Plot the boundary line given whether the covariance matrices need to be same or different
def plot_gda(arr1, arr2, quad, file, me0, me1, sigma0, sigma1, sigma):

	var('x')
	y = np.linspace(60, 160, 100)
	x1 = np.zeros([100])
	x2 = np.zeros([100])
	for i in range(len(y)):
		yc = y[i]
		var_arr = [[yc, x]]
		if quad:
			term = quadratic(me0, me1, sigma0, sigma1, var_arr)
		else:
			term = quadratic(me0, me1, sigma, sigma, var_arr)
		expr = term[0][0]
		y_ans = solve(expr, x)
		x1[i] = y_ans[0]
		if quad:
			x2[i] = y_ans[1]

	plt.plot(y, x1, 'g')
	if quad:
		plt.close()
		plt.plot(y, x2, 'y')

	for i in range(len(arr1)):
		if arr2[i][0] == 0:
			plt.plot(arr1[i][0], arr1[i][1], 'ro', color='b')
		else:
			plt.plot(arr1[i][0], arr1[i][1], 'ro', color='r')

	plt.text(60, 500, 'Alaska', color='b')
	plt.text(170, 320, 'Canada', color='r')

	if not(quad):
		plt.savefig(file)
		plt.close()

# Main function implementing the Question 4
def question4():

	arr1 = loadtxt("Assignment_1_datasets/q4x.dat")
	arr2t = [i.strip().split() for i in open("Assignment_1_datasets/q4y.dat").readlines()]
	
	arr2 = np.zeros([len(arr2t), len(arr2t[0])])

	for i in range(len(arr2t)):
		if arr2t[i][0]=='Alaska':
			arr2[i][0] = 0
		else:
			arr2[i][0] = 1

	phi = 0.
	me0 = np.zeros([len(arr1[i])])
	me1 = np.zeros([len(arr1[i])])
	for i in range(len(arr2)):
		if arr2[i][0]==0:
			me0 = np.add(me0, arr1[i])
		else:
			phi += 1
			me1 = np.add(me1, arr1[i])

	me0 = me0/(len(arr2) - phi)
	me1 = me1/phi


	sigma0 = np.zeros([len(arr1[i]), len(arr1[i])])
	sigma1 = np.zeros([len(arr1[i]), len(arr1[i])])
	for i in range(len(arr2)):
		if arr2[i][0]==0:
			term = arr1[i] - me0
			term = np.reshape(term, [-1, 1])
			sigma0 = np.add(sigma0, np.dot(term, np.transpose(term)))
		else:
			term = arr1[i] - me1
			term = np.reshape(term, [-1, 1])
			sigma1 = np.add(sigma1, np.dot(term, np.transpose(term)))

	sigma = np.add(sigma0, sigma1)/len(arr2)

	me0 = np.reshape(me0, [-1, 1])
	me1 = np.reshape(me1, [-1, 1])

	sigma0 = sigma0/(len(arr2) - phi)
	sigma1 = sigma1/phi

	phi = phi/len(arr2)

	print("Phi : ", phi)
	print("u0 : ", me0)
	print("u1 : ", me1)
	print("In case both the covariance matrices are considered same :")
	print("Sigma : ", sigma)
	print("In case both the covariance matrices are considered different :")
	print("Sigma0 : ", sigma0)
	print("Sigma1 : ", sigma1)


	plot_gda(arr1, arr2, True, 'gda_dif_cov', me0, me1, sigma0, sigma1, sigma)
	plot_gda(arr1, arr2, False, 'gda', me0, me1, sigma0, sigma1, sigma)

	print("\n\nThe plot is stored in the file named <gda>")

# ----------

while(True):
	ques = input("\n\n\nChoose question number :")
	print("\n\n\n")
	fig = plt.figure()
	fig2 = plt.figure(2)
	ax = fig.gca(projection='3d')
	if ques== '1':
		question1()
	elif ques =='2':
		question2()
	elif ques =='3':
		question3()
	elif ques=='4':
		question4()
	else:
		print("Incorrect Question Number")