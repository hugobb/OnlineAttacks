from itertools import permutations 
from heapq import nlargest
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#NOTE: both functions take n and k, where k <=  math.floor(n/math.e) as per paper, if not function will return out of range error

#OPTIMISTIC algorithm
def optimistic(n, k):
    print("optimistic")
    #calculate sampiling period
    t = math.floor(n/math.e)
    #create list with elements 1,...n
    a_list = list(range(1, n+1))
    #create n! permutations - stored in list as tuples
    perm =  permutations(a_list) 
    #sum of all chosen elements for all permutations 
    #to see how good algo is divide it by n! and by sum of top k elements
    suma = 0
    for array_per in perm:
        #convert tuple to list
        array = list(array_per)
        #create list of observed elements
        observation = array[:t]
        #create reference set 
        reference_set = nlargest(k,observation)
        #sort of reference set from largest to smallest
        reference_set.sort(reverse=True)
        #create list of future elements - that are ariving after sampiling phase
        future = array[t:]
        #create list of elements you will select
        selected = []
        #num of selected elements
        num_selected = 0
        # Getting length of "future" list 
        length = len(future) 
        i = 0   
        while (i < length) and (num_selected < k): 
            if (future[i] > reference_set[-1]):
                #pop out the element from reference set
                reference_set.pop()
                #select the item
                selected.append(future[i])
                num_selected = num_selected + 1
            i = i + 1
        #add sum of selected elements for this permuation to total sum
        suma = suma + sum(selected)
    faktorijal = math.factorial(n)
    return((2.0*suma)/((n*(n+1)-(n-k)*(n-k+1))*faktorijal))



#VIRTUAL algorithm
def virtual(n, k):
    print("virtual")
    #calculate sampiling period
    t = math.floor(n/math.e)
    #create list with elements 1,...n
    a_list = list(range(1, n+1))
    #create n! permutations - stored in list as tuples
    perm =  permutations(a_list) 
    #sum of all chosen elements for all permutations 
    #to see how good algo is divide it by n! and by sum of top k elements
    suma = 0
    for array_per in perm:
        #convert tuple to list
        array = list(array_per)
        #create list of observed elements
        observation = array[:t]
        #create reference set 
        reference_list = nlargest(k,observation)
        #sort of reference set from largest to smallest
        reference_list.sort(reverse=True)
        #set with also values of sampled before - 1, or not - 0
        reference_set = [reference_list, [1]*len(reference_list)]
        #create array of future elements - that are ariving after sampiling phase
        future = array[t:]
        #create list of elements you will select
        selected = []
        #num of selected elements
        num_selected = 0
        # Getting length of "future" list 
        length = len(future) 
        i = 0 
        while (i < length) and (num_selected < k): 
            if ((future[i] > reference_set[0][-1])and(reference_set[1][-1] > 0.5)):
                #update reference set
                reference_set[0][-1] = future[i] 
                reference_set[1][-1] = 0
                #sort reference set
                reference_set = zip(*reference_set)
                reference_set = list(map(list, reference_set))
                reference_set.sort(key=lambda x: x[0], reverse = True)
                reference_set = zip(*reference_set)
                reference_set = list(map(list, reference_set))

                #select the item
                selected.append(future[i])
                num_selected = num_selected + 1
            elif ((future[i] > reference_set[0][-1])and(reference_set[1][-1] < 0.5)):
                #update reference set
                reference_set[0][-1] = future[i]
                #sort reference set
                reference_set = zip(*reference_set)
                reference_set = list(map(list, reference_set))
                reference_set.sort(key=lambda x: x[0], reverse = True)
                reference_set = zip(*reference_set)
                reference_set = list(map(list, reference_set))
                
            i = i + 1
        #add sum of selected elements for this permuation to total sum
        suma = suma + sum(selected)
    faktorijal = math.factorial(n)
    return((2.0*suma)/((n*(n+1)-(n-k)*(n-k+1))*faktorijal))

n = 9
k = 3
optimistic_values = []
virtual_values = []

# for x in range(11):
#     if (x < 6):
#         optimistic_values.append(0)
#         virtual_values.append(0)
#     else:   
#         optimistic_values.append(optimistic(x,2))
#         virtual_values.append(virtual(x,2))
    
# optimistic_plot, = plt.plot(optimistic_values,'g*')
# virtual_plot, = plt.plot(virtual_values, 'ro')
# print(optimistic_values)
# print(virtual_values)
# plt.title('k = 2')
# plt.xlabel("total number of points - n")
# plt.ylabel("Competitive ratio")
# plt.ylim([0, 1])
# plt.legend([optimistic_plot, virtual_plot], ["optimistic", "virtual"])
# #plt.show()    
# plt.savefig('k_two.png')

# for x in range(11):
#     if (x < 9):
#         optimistic_values.append(0)
#         virtual_values.append(0)
#     else:   
#         optimistic_values.append(optimistic(x,3))
#         virtual_values.append(virtual(x,3))
    
# optimistic_plot, = plt.plot(optimistic_values,'g*')
# virtual_plot, = plt.plot(virtual_values, 'ro')
# print(optimistic_values)
# print(virtual_values)
# plt.title('k = 3')
# plt.xlabel("total number of points - n")
# plt.ylabel("Competitive ratio")
# plt.ylim([0, 1])
# plt.legend([optimistic_plot, virtual_plot], ["optimistic", "virtual"])
# #plt.show()    
# plt.savefig('k_three.png')

for x in range(11):
    if (x < 3):
        optimistic_values.append(0)
        virtual_values.append(0)
    else:   
        optimistic_values.append(optimistic(x,1))
        virtual_values.append(virtual(x,1))
    
optimistic_plot, = plt.plot(optimistic_values,'g*')
virtual_plot, = plt.plot(virtual_values, 'ro')
print(optimistic_values)
print(virtual_values)
plt.title('k = 1')
plt.xlabel("total number of points - n")
plt.ylabel("Competitive ratio")
plt.ylim([0, 1])
plt.legend([optimistic_plot, virtual_plot], ["optimistic", "virtual"])
#plt.show()    
plt.savefig('k_one.png')


#print(optimistic(n,k))
#print(virtual(n,k))