import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import copy
from torchvision import datasets, transforms
from torch.utils.data import random_split,Subset
from functools import reduce
import cvxpy as cp
import gurobipy as gp
import torchtext.legacy 
from torchtext.datasets import IMDB
from torchvision.datasets import CIFAR10
'''

Enough workers 

'''

def get_type(num_per_type,ID):
    sum=0
    i=0
    for i in range(len(num_per_type)):
        sum += num_per_type[i]
        if sum >=ID:
            break
    return i
    
###basic function use
##mathematical
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def cal_mean(numbers):
    return sum(numbers) / len(numbers)

def cal_variance(numbers):
    mean = cal_mean(numbers)
    squared_diff = [(x - mean) ** 2 for x in numbers]
    variance = math.sqrt(sum(squared_diff) / len(numbers)) 
    return variance

def max_standard(array):
    array=np.array(array)
    m=np.max(array)
    array=array/m
    return array
    
def vector_projection(g, h):
    g=np.array(g)
    h=np.array(h)
    dot_product = np.dot(g, h)
    norm_h_squared = np.linalg.norm(h)

    projection = (dot_product / norm_h_squared) 

    return projection    

def cosine_similarity(vector1, vector2):
    vector1=np.array(vector1)
    vector2=np.array(vector2)
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm_vector1 * norm_vector2)

    return similarity 


##data pre-training operation

def load_MNIST_data(num_individuals = 30,training_set_size = 1000,validation_set_size=2000,test_set_size=2000):
    # Define the transformations
    transform = transforms.Compose([transforms.ToTensor()])
    # Load MNIST dataset
    mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    total=validation_set_size+test_set_size
    # Split datasets for each individual
    mnist_individual_datasets = [torch.utils.data.Subset(mnist_train_dataset, range(i * training_set_size, (i + 1) * training_set_size)) for i in range(num_individuals)]
    mnist_individual_test_valid=[torch.utils.data.Subset(mnist_test_dataset, range(i * total, (i + 1) * total)) for i in range(num_individuals)]
    dataset=[]
    for i in range(len(mnist_individual_datasets)):
        sub={}
        sub["train"]=list(mnist_individual_datasets[i%(int)(len(mnist_train_dataset)/training_set_size)])
        # sub["validation"],sub['test']=random_split(mnist_individual_test_valid[i], [validation_set_size, test_set_size])
        sub["validation"],sub['test']=random_split(mnist_individual_test_valid[i%(int)((len(mnist_test_dataset)/(validation_set_size+test_set_size)))], [validation_set_size, test_set_size])
        sub["validation"]=list(sub["validation"])
        sub['test']=list(sub['test'])
        dataset.append(sub)
    return dataset

def load_Fashion(num_individuals = 30,training_set_size = 1000,validation_set_size=2000,test_set_size=2000):
    transform = transforms.Compose([transforms.ToTensor()])
    # Load MNIST dataset
    mnist_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    total=validation_set_size+test_set_size
    # Split datasets for each individual
    mnist_individual_datasets = [torch.utils.data.Subset(mnist_train_dataset, range(i * training_set_size, (i + 1) * training_set_size)) for i in range(num_individuals)]
    mnist_individual_test_valid=[torch.utils.data.Subset(mnist_test_dataset, range(i * total, (i + 1) * total)) for i in range(num_individuals)]
    dataset=[]
    for i in range(len(mnist_individual_datasets)):
        sub={}
        sub["train"]=list(mnist_individual_datasets[i%(int)(len(mnist_train_dataset)/training_set_size)])
        # sub["validation"],sub['test']=random_split(mnist_individual_test_valid[i], [validation_set_size, test_set_size])
        sub["validation"],sub['test']=random_split(mnist_individual_test_valid[i%(int)((len(mnist_test_dataset)/(validation_set_size+test_set_size)))], [validation_set_size, test_set_size])
        sub["validation"]=list(sub["validation"])
        sub['test']=list(sub['test'])
        dataset.append(sub)
    return dataset
        
def change_labels(dataset, percentage=0.1):
    total_samples = len(dataset)
    num_samples_to_change = int(percentage * total_samples)
    indices_to_change = np.random.choice(total_samples, num_samples_to_change, replace=False)
    new_data=copy.deepcopy(dataset)
    for idx in indices_to_change:
        old_label = new_data[idx][1]  # Get the old label as an integer
        new_label = (old_label + np.random.randint(1, 10)) % 10  # Change label to a different one
        new_data[idx] =(new_data[idx][0],new_label)  # Assign the new label    
    return new_data

def load_CIFAR10(num_individuals = 30,training_set_size = 20000,validation_set_size=2000,test_set_size=2000):
    transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    # Load cifar10 dataset
    cifar10_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    total=validation_set_size+test_set_size
    # Split datasets for each individual
    cifar10_individual_datasets = [torch.utils.data.Subset(cifar10_train_dataset, range(i * training_set_size, (i + 1) * training_set_size)) for i in range(num_individuals)]
    cifar10_individual_test_valid=[torch.utils.data.Subset(cifar10_test_dataset, range(i * total, (i + 1) * total)) for i in range(num_individuals)]
    dataset=[]
    for i in range(len(cifar10_individual_datasets)):
        sub={}
        sub["train"]=list(cifar10_individual_datasets[i%(int)(len(cifar10_train_dataset)/training_set_size)])
        # sub["validation"],sub['test']=random_split(cifar10_individual_test_valid[i], [validation_set_size, test_set_size])
        sub["validation"],sub['test']=random_split(cifar10_individual_test_valid[i%(int)((len(cifar10_test_dataset)/(validation_set_size+test_set_size)))], [validation_set_size, test_set_size])
        sub["validation"]=list(sub["validation"])
        sub['test']=list(sub['test'])
        dataset.append(sub)
    return dataset



##model operation
# Function to flatten parameters
def flatten_parameters(model):
    return np.concatenate([param.detach().cpu().numpy().flatten() for param in model.parameters()])

# Function to reconstruct model from flattened parameters
def reconstruct_model(flattened_params, model):
    state_dict = model.state_dict()
    start_idx = 0
    for key in state_dict.keys():
        param = state_dict[key]
        end_idx = start_idx + param.numel()
        new_param_shape = param.shape
        flat_param = flattened_params[start_idx:end_idx].reshape(new_param_shape)
        state_dict[key] = torch.from_numpy(flat_param)
        start_idx = end_idx
    model.load_state_dict(state_dict)

def test_accuracy(old_dataset,new_dataset):
    num=0
    for i in range(len(new_dataset)):
        _,label=new_dataset[i]
        _,old_label=old_dataset[i]
        if label == old_label:
            num +=1
    return (num/len(new_dataset))



## Define the SimpleNN model 

##the data pass in need to be transformed first

##MNIST

class SimpleNN_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN_MNIST, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.input_size=input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # def train_nn(self,data,learning_rate=0.05, num_of_batch=64):
    def train_nn(self,data_loader,epochs,learning_rate=0.05):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        # Training loop
        # data_iter = iter(data)
        # for i in range(num_of_batch):
        #     images, labels=next(data_iter)
        for epoch in range(epochs):
            for images,labels in data_loader:
                images,labels=images.cuda(),labels.cuda() 
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()     
        return self.state_dict() 
             
        
    def test_nn(self, data):
        accuracies = []
        losses = []

        # Evaluate accuracy and loss
        self.eval()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in data:
                images,labels=images.cuda(),labels.cuda()
                outputs = self(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                accuracies.append((predicted == labels).sum().item() / labels.size(0))
                losses.append(loss.item())

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = sum(losses) / len(losses)

        return avg_accuracy, avg_loss

##
class CNN_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN_MNIST, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1_cnn = nn.Linear(64 * 7 * 7, 128)
        self.fc2_cnn = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward_cnn(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1_cnn(x))
        x = self.fc2_cnn(x)
        return x

    def train_nn(self, data_loader, epochs, learning_rate=0.05):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = self.forward_cnn(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.state_dict()

    def test_nn(self, data):
        accuracies = []
        losses = []

        self.eval()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in data:
                images, labels = images.cuda(), labels.cuda()
                outputs = self.forward_cnn(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                accuracies.append((predicted == labels).sum().item() / labels.size(0))
                losses.append(loss.item())

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = sum(losses) / len(losses)

        return avg_accuracy, avg_loss
    
#CIFAR10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
re_device=  torch.device( "cpu")
class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def train_nn(self, data_loader, epochs, learning_rate=0.001):
        self.to(device)
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            # correct = 0
            # total = 0
            if epoch>9:
                optimizer = optim.Adam(self.parameters(), lr=learning_rate/10)
            # accuracies = []
            # losses = []
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        self.to(re_device)        
                
        
               # _, predicted = torch.max(outputs.data, 1)
                ## total += labels.size(0)
                ## correct += (predicted == labels).sum().item()
            #     accuracies.append((predicted == labels).sum().item() / labels.size(0))
            #     losses.append(loss.item())

            # avg_accuracy = sum(accuracies) / len(accuracies)
            # avg_loss = sum(losses) / len(losses)
            # print(f"Epoch {epoch + 1} Training Accuracy: {avg_accuracy},loss{avg_loss}")

        return self.state_dict()

    def test_nn(self, data):
        accuracies = []
        losses = []
        self.to(device)
        self.eval()
        # correct = 0
        # total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for da in data:
                images, labels = da[0].to(device), da[1].to(device)
                outputs = self(images.to(device))
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                accuracies.append((predicted == labels).sum().item() / labels.size(0))
                losses.append(loss.item())

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = sum(losses) / len(losses)
        self.to(re_device)
        return avg_accuracy, avg_loss




class Request_Set:
    def __init__(self,workers,requesters,num_per_type,budget):
        #init by one for connectivity and convenience
        #visit matrix first is the requester than is the workers
        '''
        HOW THE PAGE RANK SHOULD BE MODIFIED?
        '''    
        '''
        visit:        
                  requester 
        requester[        ]
        
        '''
        self.budget=budget
        self.visit=[[1 for i in range(len(requesters))]for j in range(len(requesters))] ##1 to make sure the graph is connected 
        self.workers=workers
        self.requesters=requesters
        self.num_per_type=num_per_type
        #for experimental usage
        self.reward_avg=[0 for i in range(sum(num_per_type))]
        self.rep=[0 for i in range(sum(num_per_type))]
        self.accuracy={'BMW_FL_g':[],'BMW_FL_s':[],'Ran_Pri_seperate':[],'Ran_Pri_overall':[],'RRAFL_seperate':[],'RRAFL_overall':[],"greedy_s":[],"greedy_g":[]}
        self.rep_per_round={'BMW_FL_g':[],'BMW_FL_s':[],'Ran_Pri_seperate':[],'Ran_Pri_overall':[],'RRAFL_seperate':[],'RRAFL_overall':[],"greedy_s":[],"greedy_g":[]}
    
     ## pagerank
    #(formula 6.P_r_k)    
    def cal_pagerank(self):
        visit=self.visit
        # cal row averages
        row_sum = np.sum(visit, axis=1)
        # Divide each row by its average
        prob = (visit / row_sum[:, np.newaxis]).T
        I=np.eye(prob.shape[0])
        A=prob-I
        b=np.array([0 for i in range(A.shape[0])])
        b[-1]=1
        for i in range(A.shape[1]):
            A[-1][i]=1
        self.page_rank=np.linalg.solve(A,b)       
        
    #(formula 6.P_r'_k & formula 7)
    def cal_modified_pagerank(self):
        self.cal_pagerank()
        mu=cal_mean(self.page_rank)
        sigma=cal_variance(self.page_rank)
        if sigma:
            for i in range(len(self.page_rank)):
                self.page_rank[i]=sigmoid((self.page_rank[i]-mu)/sigma)
               

    def get_type_sum(self,type_ID):
        sum=0
        for i in range(type_ID):
            sum +=self.num_per_type[i]
        return sum

    def reset_for_ALG(self,bid_group_num=0):
        for worker in self.workers:
            worker.bid=worker.bid_test[bid_group_num]
        for requester in self.requesters:
            requester.restart()            
###
#integrate the selction of participants in global model for convenience since the global can operate the local        
    def run(self,mode="get_rep",size_of_selection=10):
        #m requester, n workers, l groups
        m=len(self.requesters)
        n=sum(self.num_per_type)
        l=len(self.num_per_type)
        sum_rep=0
        len_rep=0
        #budget more first
        if mode=='greedy_g':
            for requester in self.requesters:
                requester.init() 
            sum_pay=0
            each_pay=[0 for i in range(m)]
            conf=[{} for i in range(m)]  
            sequence=list(range(n))
            random.shuffle(sequence)
            num_sel=0
            high=max([worker.range_of_bid["high"] for worker in self.workers])
            low=min([worker.range_of_bid["low"] for worker in self.workers])
            for i in range(n):
                unsort_req=[i for i in range(m)]
                sort_req=sorted(unsort_req,key=lambda x:self.requesters[x].budget-each_pay[x],reverse=True)
                requester=sort_req[0]
                price=random.randint(low,high)
                if price>=self.workers[sequence[i]].bid and sum_pay+self.workers[sequence[i]].bid<self.budget  and not(self.workers[sequence[i]].type_ID in conf[requester]):
                    self.requesters[requester].participants[sequence[i]]=1
                    conf[requester][self.workers[sequence[i]].type_ID]=self.workers[sequence[i]].type_ID
                    print(f"requester {requester}, worker:{sequence[i]},accuracy{self.workers[sequence[i]].accuracy},type:{self.workers[sequence[i]].type_ID},rep:{self.rep[sequence[i]]}")
                    sum_pay += price
                    each_pay[requester]+=price 
                    sum_rep+=self.rep[sequence[i]]
                    len_rep+=1
                    num_sel+=1
                elif num_sel<m:
                    continue
                else:
                    break
            print(f"sum_pay :{sum_pay}")   
            
        if mode== "greedy_s":
            for requester in self.requesters:
                requester.init() 
            each_pay=[0 for i in range(m)]
            sum_pay=0
            num_sel=0
            conf=[{} for i in range(m)]
            sequence=list(range(n))
            high=max([worker.range_of_bid["high"] for worker in self.workers])
            low=min([worker.range_of_bid["low"] for worker in self.workers])
            random.shuffle(sequence)
            for i in range(n):
                unsort_req=[i for i in range(m)]
                sort_req=sorted(unsort_req,key=lambda x:self.requesters[x].budget-each_pay[x],reverse=True)
                requester=sort_req[0]
                price=random.randint(low,high)
                # print(price,self.workers[sequence[i]].bid)
                if price>=self.workers[sequence[i]].bid  and each_pay[requester]+self.workers[sequence[i]].bid <self.requesters[requester].budget and not(self.workers[sequence[i]].type_ID in conf[requester]):
                    self.requesters[requester].participants[sequence[i]]=1
                    conf[requester][self.workers[sequence[i]].type_ID]=self.workers[sequence[i]].type_ID
                    print(f"requester {requester}, worker:{sequence[i]},accuracy{self.workers[sequence[i]].accuracy},type:{self.workers[sequence[i]].type_ID},rep:{self.rep[sequence[i]]}")
                    each_pay[requester]+=price
                    sum_pay += price 
                    num_sel +=1
                    sum_rep+=self.rep[sequence[i]]
                    len_rep+=1
                elif num_sel<m:
                    continue
                else:
                    break
            for j in range(m):
                print(f"pay_wokrer{j}:{each_pay[j]}",end=" ") 
            print(f"sum_pay :{sum_pay}")  
        if mode=="BMW_FL_g":
            #ID is the ID of worker
            quality=[{"ID":i,"type_ID":self.workers[i].type_ID,"rep":self.rep[i],"q" :self.workers[i].bid/self.rep[i]} for i in range(n)]
            quality=sorted(quality,key=lambda x:x["q"])
            sum_pay=0  
            model=gp.Model()
            X=model.addVars(n,m,name="x",vtype=gp.GRB.INTEGER)
            model.setObjective(gp.quicksum(quality[i]['rep']*X[i,j]for i in range(n) for j in range(m)),sense=gp.GRB.MAXIMIZE)
            #0 1 programming
            for i in range(n):
                for j in range(m):
                    model.addConstr(X[i,j] <= 1) 
                    model.addConstr(X[i,j] >= 0)
            #heterogenity(l constraints)        
            for num in range(l):
                het=[[0 for j in range(m)]for i in range(n)]
                for i in range(n):
                    for j in range(m):
                        if quality[i]["type_ID"]==num:
                            het[i][j]=1
                for j in range(m):
                    model.addConstr(gp.quicksum(het[i][j]*X[i,j] for i in range(n)) <= 1)
            #one requester(n constraints)
            for i in range(n):
                model.addConstr(gp.quicksum(X[i,j] for j in range(m)) <=1)
            model.setParam('OutputFlag', 0)
            #constraints to be removed
            for i in range(n-1,-1,-1):
                for j in range(m):  
                    model.addConstr(X[i,j]== 0)
            #the constraints param is only set aftet the .optimize() is run !!!
            model.optimize()       
            for i in range(n):
                model.remove(model.getConstrs()[-m:])
                model.optimize()
                if model.ObjVal*quality[i]['q']>self.budget:
                    for j in range(m):
                        model.addConstr(X[i,j] == 0)
                    model.optimize()     
                    break  
            for requester in self.requesters:
                requester.init()  
            to_be_mod=[[]for i in range(m)]
            for i in range(n):
                for j in range(m):
                    if X[i,j].X==1:
                        to_be_mod[j].append(i)
            re_allocate=[{'ID':i,'allo':to_be_mod[i] }for i in range(m)]  
            re_allocate=sorted(re_allocate,key=lambda x:len(x['allo'])) 
            for i in range(m):
                if len(re_allocate[i]['allo'])==0:
                    re_allocate[i]['allo'].append(re_allocate[m-1]['allo'][-1])
                    re_allocate[m-1]['allo'].pop()
                else:
                    break 
            for k in range(m):
                y=re_allocate[k]['ID']
                for x in re_allocate[k]['allo']:
                    sum_rep += self.rep[quality[x]['ID']]
                    len_rep += 1
                    pay=quality[x]['rep']*min(quality[x]['q'],self.budget/model.ObjVal)
                    sum_pay += pay
                    self.requesters[y].participants[quality[x]['ID']]=1
                    self.requesters[y].eval_models[quality[x]['ID']].interaction=1
                    print(f"requester:{y}, worker:{quality[x]['ID']},accuracy;{self.workers[quality[x]['ID']].accuracy},type:{quality[x]['type_ID']} rep:{quality[x]['rep']}")      
            # for i in range(n):
            #     for j in range(m):
            #         if X[i,j].X==1:
            #             sum_rep += self.rep[quality[i]['ID']]
            #             len_rep += 1
            #             pay=quality[i]['rep']*min(quality[i]['q'],self.budget/model.ObjVal)
            #             sum_pay += pay
            #             self.requesters[j].participants[quality[i]['ID']]=1
            #             self.requesters[j].eval_models[quality[i]['ID']].interaction=1
            #             print(f"requester:{j}, worker:{quality[i]['ID']},accuracy;{self.workers[quality[i]['ID']].accuracy},type:{quality[i]['type_ID']} rep:{quality[i]['rep']}") 
            print(f"sum_pay :{sum_pay}")   
                
        if mode=='BMW_FL_s':
            sum_pay=0
            each_pay=[0 for i in range(m)]
            v_min=min(self.rep)
            v_max=max(self.rep)
            # print(f'v_min {v_min} v_max {v_max}')
            #the division num in alg2 line2
            gamma=math.ceil(np.log10(v_max/v_min))
            groups=[[]for i in range(gamma)]
            for i in range(n):
                self.rep[i] /=v_min
                #the index start from 0
                if self.rep[i]>1:
                    groups[math.ceil(np.log10(self.rep[i]))-1].append(i)
                else:
                    groups[0].append(i)
            g_max=max(self.num_per_type)
            alpha=max(g_max,m)   
            possible_pay=[]
            for group in groups:
                s_r=len(group)
                if s_r>m :
                    # #alg3 line 3
                    # if random.random()<alpha/(2*alpha+1):
                    #     # print(f'(\u03B1)/(2\u03B1+1) rep around {self.rep[group[0]]},len{s_r}')
                    #     #coherence with the else case
                    #     bid_sub=[{'ID':x,'rep':self.rep[x],'type_ID':self.workers[x].type_ID,'bid':self.workers[x].bid} for x in group]
                    #     pos=[[0 for j in range(m)] for i in range(s_r)]    
                    #     for i in range(min(m,s_r)):
                    #         pos[i][i]=1
                    #     r=min([req.budget for req in self.requesters])
                    #     possible_pay.append({'pay':r,'pay_flag':pos,'bid_set':bid_sub,'s_r':s_r,'type':'(\u03B1)/(2\u03B1+1)'})
                                
                    # #alg3 line4
                    # else:
                        # print(f'(\u03B1+1)/(2\u03B1+1) rep around {self.rep[group[0]]},len{s_r}')
                        #create model
                        model=gp.Model()
                        X=model.addVars(s_r,m,name="x",vtype=gp.GRB.INTEGER)
                        model.setObjective(gp.quicksum(X[i,j]for i in range(s_r) for j in range(m)),sense=gp.GRB.MAXIMIZE)
                        model.setParam('OutputFlag', 0)
                        #alg 4 line2
                        bid_sub=[{'ID':x,'rep':self.rep[x],'type_ID':self.workers[x].type_ID,'bid':self.workers[x].bid} for x in group]
                        bid_sub=sorted(bid_sub,key=lambda x:x['bid'])
                        # #alg 4 line 3
                        # for i in range(s_r):
                        #     bid_sub[i]['weight']=np.power(2,i+1)
                        #alg 4 line 4
                        R=[]
                        #R_b
                        for i in range(m):
                            for j in range(s_r):
                                R.append(self.requesters[i].budget/(j+1))
                        #R_s
                        for x in bid_sub:
                            R.append(x['bid'])
                        R=sorted(R)
                        r=0
                        #alg 4 line 5 6
                        for r in R:
                            #M_b(r)
                            M_b=0
                            for j in range(m):
                                M_b += math.floor(self.requesters[j].budget/r)   
                            #renew contraints
                            model.remove(model.getConstrs()) 
                            #0 1 programming
                            for i in range(s_r):
                                for j in range(m):
                                    model.addConstr(X[i,j] <= 1) 
                                    model.addConstr(X[i,j] >= 0)
                            #heterogenity(l constraints)        
                            for num in range(l):
                                het=[[0 for j in range(m)]for i in range(n)]
                                for i in range(s_r):
                                    for j in range(m):
                                        if bid_sub[i]["type_ID"]==num:
                                            het[i][j]=1
                                for j in range(m):
                                    model.addConstr(gp.quicksum(het[i][j]*X[i,j] for i in range(s_r)) <= 1)
                            #one requester(n constraints)
                            for i in range(s_r):
                                model.addConstr(gp.quicksum(X[i,j] for j in range(m)) <=1)
                            #With ability M_b(r)
                            for j in range(m):
                                model.addConstr(gp.quicksum(X[i,j] for i in range(s_r)) <= math.floor(self.requesters[j].budget/r))
                            model.optimize()
                            if model.ObjVal==M_b:
                                break
                        #append one posiible payment omit alg 4 line 9 - line 22
                        pos=[[0 for j in range(m)] for i in range(s_r)]    
                        for i in range(s_r):
                            for j in range(m):
                                if X[i,j].X==1:
                                    pos[i][j]=1
                        possible_pay.append({'pay':r,'pay_flag':pos,'bid_set':bid_sub,'s_r':s_r,'type':'(\u03B1+1)/(2\u03B1+1)'})  
            #alg 2 line 6
            payscheme=random.choice(possible_pay[1:])
            for requester in self.requesters:
                requester.init()     
            for i in range(payscheme['s_r']):
                for j in range(m):
                    if payscheme['pay_flag'][i][j]==1:
                        sum_rep += self.rep[payscheme['bid_set'][i]['ID']]
                        len_rep+= 1
                        pay=payscheme['pay']
                        each_pay[j] += pay
                        sum_pay += pay
                        self.requesters[j].participants[payscheme['bid_set'][i]['ID']]=1
                        self.requesters[j].eval_models[payscheme['bid_set'][i]['ID']].interaction=1
                        print(f"requester:{j},worker:{payscheme['bid_set'][i]['ID']},accuracy{self.workers[payscheme['bid_set'][i]['ID']].accuracy},type:{payscheme['bid_set'][i]['type_ID']},rep:{payscheme['bid_set'][i]['rep']}")
            print(f"type {payscheme['type']}")
            for j in range(m):
                print(f"pay_wokrer{j}:{each_pay[j]}",end=" ") 
            print(f"sum_pay :{sum_pay}")   
                  
        if mode=='Ran_Pri_overall':
            for requester in self.requesters:
                requester.init() 
            sum_pay=0
            conf=[{} for i in range(m)]  
            sequence=list(range(n))
            random.shuffle(sequence)
            num_sel=0
            high=max([worker.range_of_bid["high"] for worker in self.workers])
            low=min([worker.range_of_bid["low"] for worker in self.workers])
            for i in range(n):
                requester=random.randint(0,m-1)
                price=random.randint(low,high)
                if price>=self.workers[sequence[i]].bid and sum_pay+self.workers[sequence[i]].bid<self.budget  and not(self.workers[sequence[i]].type_ID in conf[requester]):
                    self.requesters[requester].participants[sequence[i]]=1
                    conf[requester][self.workers[sequence[i]].type_ID]=self.workers[sequence[i]].type_ID
                    print(f"requester {requester}, worker:{sequence[i]},accuracy{self.workers[sequence[i]].accuracy},type:{self.workers[sequence[i]].type_ID},rep:{self.rep[sequence[i]]}")
                    sum_pay += price 
                    sum_rep+=self.rep[sequence[i]]
                    len_rep+=1
                    num_sel+=1
                elif num_sel<m:
                    continue
                else:
                    break
            print(f"sum_pay :{sum_pay}")   
            
        if mode== "Ran_Pri_seperate":
            for requester in self.requesters:
                requester.init() 
            each_pay=[0 for i in range(m)]
            sum_pay=0
            num_sel=0
            conf=[{} for i in range(m)]
            sequence=list(range(n))
            high=max([worker.range_of_bid["high"] for worker in self.workers])
            low=min([worker.range_of_bid["low"] for worker in self.workers])
            random.shuffle(sequence)
            for i in range(n):
                requester=random.randint(0,m-1)
                price=random.randint(low,high)
                # print(price,self.workers[sequence[i]].bid)
                if price>=self.workers[sequence[i]].bid  and each_pay[requester]+self.workers[sequence[i]].bid <self.requesters[requester].budget and not(self.workers[sequence[i]].type_ID in conf[requester]):
                    self.requesters[requester].participants[sequence[i]]=1
                    conf[requester][self.workers[sequence[i]].type_ID]=self.workers[sequence[i]].type_ID
                    print(f"requester {requester}, worker:{sequence[i]},accuracy{self.workers[sequence[i]].accuracy},type:{self.workers[sequence[i]].type_ID},rep:{self.rep[sequence[i]]}")
                    each_pay[requester]+=price
                    sum_pay += price 
                    num_sel +=1
                    sum_rep+=self.rep[sequence[i]]
                    len_rep+=1
                elif num_sel<m:
                    continue
                else:
                    break
            for j in range(m):
                print(f"pay_wokrer{j}:{each_pay[j]}",end=" ") 
            print(f"sum_pay :{sum_pay}")   
        
        if mode=='RRAFL_overall':
            sum_pay=0
            participants=[0 for i in range(n)]
            conf=[{} for i in range(m)]
            bid=[{'ID':i,"type":self.workers[i].type_ID,'bid':self.workers[i].bid,'q':self.workers[i].bid/self.rep[i],'rep':self.rep[i]} for i in range(n)]
            bid=sorted(bid,key=lambda x:x['q'])
            rep_sum=0
            q=0
            for i in range(n-1):
                rep_sum +=bid[i]['bid']/bid[i]['q']
                if rep_sum*bid[i+1]['q']<=self.budget:
                    participants[bid[i]['ID']]=1
                else:
                    q =bid[i]['q']
                    break
            for requester in self.requesters:
                requester.init() 
            sum_pay=0
            # for i in range(m):
            #     if participants[bid[i]['ID']]:
            #         requester=i
            #         sum_rep+=self.rep[bid[i]['ID']]
            #         len_rep+=1
            #         sum_pay += bid[i]['q']*q
            #         self.requesters[requester].participants[bid[i]['ID']]=1
            #         print(f"requester {requester}, worker:{bid[i]['ID']},accuracy{self.workers[bid[i]['ID']].accuracy},type:{self.workers[bid[i]['ID']].type_ID},rep:{self.rep[bid[i]['ID']]}")         
            for i in range(n):
                requester=random.randint(0,m-1)
                if participants[bid[i]['ID']] and not(bid[i]['type'] in conf[requester]):
                    sum_rep+=self.rep[bid[i]['ID']]
                    len_rep+=1
                    self.requesters[requester].participants[bid[i]['ID']]=1
                    conf[requester][bid[i]['type']]=bid[i]['type']
                    print(f"requester {requester}, worker:{bid[i]['ID']},accuracy{self.workers[bid[i]['ID']].accuracy},type:{self.workers[bid[i]['ID']].type_ID},rep:{self.rep[bid[i]['ID']]}")
                    sum_pay += bid[i]['rep']*q 
                #if confilct break
                elif participants[i]:
                    break
            print(f"sum_pay :{sum_pay}")   
           
        
        if mode=='RRAFL_seperate':
            sum_pay=0
            each_pay=[0 for i in range(m)]
            conf=[{} for i in range(m)]
            participants=[0 for i in range(n)]
            bid=[{'ID':i,"type":self.workers[i].type_ID,'bid':self.workers[i].bid,'q':self.workers[i].bid/self.rep[i],'rep':self.rep[i]} for i in range(n)]
            bid=sorted(bid,key=lambda x:x['q'])
            rep_sum=0
            q=0
            for i in range(n-1):
                rep_sum +=bid[i]['bid']/bid[i]['q']
                if rep_sum*bid[i+1]['q']<=self.budget:
                    participants[bid[i]['ID']]=1
                else:
                    q=bid[i]['q']
                    break
            for requester in self.requesters:
                requester.init()
            sum_pay=0
            for i in range(n):
                requester=random.randint(0,m-1)
                if participants[bid[i]['ID']] and not(bid[i]['type'] in conf[requester]) and each_pay[requester]+bid[i]['q']*q<=self.requesters[requester].budget:
                    sum_rep+=self.rep[bid[i]['ID']]
                    len_rep+=1
                    self.requesters[requester].participants[bid[i]['ID']]=1
                    conf[requester][bid[i]['type']]=bid[i]['type']
                    print(f"requester {requester}, worker:{bid[i]['ID']},accuracy{self.workers[bid[i]['ID']].accuracy},type:{self.workers[bid[i]['ID']].type_ID},rep:{self.rep[bid[i]['ID']]}")
                    sum_pay += bid[i]['rep']*q
                    each_pay[requester] +=   bid[i]['q']*q 
                #conflict or not enough budget break  
                elif participants[bid[i]['ID']]:
                    break
            for j in range(m):
                print(f"pay_wokrer{j}:{each_pay[j]}",end=" ") 
            print(f"sum_pay :{sum_pay}")    
          
        
        if mode=="get_rep":
            self.cal_modified_pagerank()
            #( step 2 generate bid)
            #renew bid 
            for worker in self.workers:
                worker.generate_bid()
            print(self.requesters[0].rounds)
            for requester in self.requesters:
                requester.cal_rep(set_req=self)
                requester.init()
                #random selection of workers  
                selected_number=np.random.choice([i for i in range(n)],size=size_of_selection,replace=False)  
                for x in selected_number:          
                    requester.participants[x]=1
                    #record the interaction(Interact means one direction?)
                    requester.eval_models[x].interaction=1
                print(f"requester:{requester.ID} result of global_accuracy & loss:{requester.aggregate(mode=mode)}")    
            #renew avg rep    
            self.rep=[0 for i in range(n)]  
            for i in range(n):
                for req in self.requesters:
                    self.rep[i] += req.eval_models[i].comprehensive_reputation
                self.rep[i] /= m
            output=[{"ID":i,"rep":self.rep[i],"accuracy":self.workers[i].accuracy,"success":np.average(self.workers[i].success),"fail":np.average(self.workers[i].fail) }for i in range(n)]
            output=sorted(output,key=lambda x:x['rep'])
            if self.requesters[0].rounds and self.requesters[0].rounds%10==0:
                for i in range(n):
                    print(output[i],end='  ')
                    if (i+1)%3==0:
                        print("")
                print("")
                
        else:
            self.accuracy[mode].append(0)
            for i in range(5):
                print(f"rep_sum:{sum_rep} len_of_rep{len_rep} avg{sum_rep/len_rep}")
                for requester in self.requesters:
                    ans=requester.aggregate(mode=mode)
                    if i==4:
                        self.accuracy[mode][-1] += ans[0]        
                        print(f"requester:{requester.ID},result of global_accuracy & loss:{ans}")
            self.accuracy[mode][-1] /=m
            self.rep_per_round[mode].append(sum_rep)              
                            
        
class eval_worker:
   
    def __init__(self,ID,type_ID,interaction=0,comprehensive_reputation=1,indirect_reputation=0,eval_reputation=0,direct_reputation=1):
        self.ID = ID
        self.type_ID=type_ID
        self.indirect_reputation=indirect_reputation
        #interaction==1 means thet have had interaction(before or now)
        #define of interaction(pass the quality test or have been test)? 
        self.interaction = interaction
        self.eval_reputation = eval_reputation
        self.direct_reputation = direct_reputation
        self.comprehensive_reputation = comprehensive_reputation
        

class Requester:
    def __init__(self,ID,budget,workers,num_per_type,num_requester,data,Lambda=0.2,beta1=0.5,beta2=4,delta_threshold=-0.01,omega=0.4,ksi=0.5,batch_size=100,mode="MNIST"):
        if mode=="MNIST":
            self.global_model=SimpleNN_MNIST(input_size=workers[0].input_size,hidden_size=workers[0].hidden_size,output_size=workers[0].output_size).cuda()  
        else:
            self.global_model=CNN_CIFAR10() 
        self.init_param=flatten_parameters(self.global_model)
        self.params=[flatten_parameters(self.global_model)]
        self.local_updates={}
        #select flag->self.participants,can also be used to indicate participants set,the one to be evaluate
        self.participants=[0 for i in range(sum(num_per_type))]
        self.type_check=[0 for i in range(len(num_per_type))]
        self.num_requester=num_requester
        self.phi=[0 for i in range(num_requester)]
        self.weight=[0 for i in range(num_requester)]
        #effective_recommender
        self.num_effective=0
        self.ID=ID
        self.workers=workers
        self.rounds=0
        self.num_per_type=num_per_type
        self.beta1=beta1
        self.beta2=beta2
        self.delta_threshold=delta_threshold
        self.omega=omega
        self.ksi=ksi
        self.Lambda=Lambda
        self.beta=0
        self.budget=budget
        self.eval_models=[eval_worker(ID=i,type_ID=workers[i].type_ID) for i in range(sum(num_per_type))]
        #intermidiate variant
        self.mean_dir_rep=1
        #accuracy for the data test
        self.results={'get_rep':[],'BMW_FL_g':[],'BMW_FL_s':[],'Ran_Pri_seperate':[],'Ran_Pri_overall':[],'RRAFL_seperate':[],'RRAFL_overall':[],"greedy_s":[],"greedy_g":[]}
        self.batch_size=batch_size
        self.test_set=DataLoader(dataset=data['test'],shuffle=True,batch_size=self.batch_size)
        self.validation_set=DataLoader(dataset=data['validation'],shuffle=True,batch_size=self.batch_size)
        
    
    def recieve_upload(self,ID):
        ##self.ID for requester's ID 
        # ID for worker's ID
        self.local_updates[ID]=np.array(self.workers[ID].updates[self.ID][-1])
    
    def restart(self):
        self.params.append(self.init_param)
        for worker in self.workers:
            worker.restart()
        
    def cal_rep(self,set_req):
        if self.rounds:
            self.cal_eval_reputation_all()
            self.cal_direct_reputation_all()
            self.cal_mean_direct_reputation()
        # cal comprehensive reputation for requester(formula 11 formula 10 and indirect_reputation)
        self.cal_indirect_reputation_all(set_req)
        self.cal_beta()
        self.cal_comprehensive_reputation_all()

    def init(self):
        #init participant set , num of effetive,local_updates
        self.participants=[0 for i in range(sum(self.num_per_type))]
        self.num_effective=0
        self.local_updates={}

    def aggregate(self,mode):    
        flag=False
        for worker in self.workers:
            ##(step 5 download global model step 6 train local model)
            worker.download(self)
            worker.train(self)
        
        for worker in self.workers:
        ##(step 5 download global model step 6 train local model)
            if self.participants[worker.ID]==1:
                ##(step 7 upload local model->self.recieve_upload)
                self.recieve_upload(worker.ID)
                flag=True
        params=self.params[-1]
        if flag:         
            l=0
            for k,v in self.local_updates.items():
                l=np.add(l,v)
            params=self.params[-1]+l/len(self.local_updates)
            reconstruct_model(model=self.global_model,flattened_params=params)
            loss=self.global_model.test_nn(data=self.validation_set)[1]
            threshold=self.delta_threshold
            #first delta_j then s_j after that a_j finally local updates
            a_j={}
            for k,v in self.local_updates.items():
                lj=np.subtract(l,v)
                if len(self.local_updates)>1: 
                    params=self.params[-1]+lj/(len(self.local_updates)-1)
                else:
                    params=self.params[-1]
                reconstruct_model(model=self.global_model,flattened_params=params)
                loss_j=self.global_model.test_nn(data=self.validation_set)[1]
                #(formula 16)
                if loss_j-loss>=threshold :
                    self.workers[k].success[self.ID] += 1
                    a_j[k]=loss_j-loss
                else:
                    self.workers[k].fail[self.ID] += 1
            flag=False
            if len(a_j):
                flag=True   
                #(formula 17  a_j is now s_j)
                min_j=1
                sum_j=0
                #Neffective for next round
                self.num_effective=len(a_j)
                #find min
                for k,v in a_j.items():
                    min_j=min(min_j,v)
                #divisor
                for k,v in a_j.items():
                    a_j[k]=v-min_j
                    sum_j+=v-min_j
                #finish of formula 17
                if sum_j:
                    for k,v in a_j.items():
                        a_j[k]=v/sum_j
                sum_a=0   
                #(formula 18 nominator and divisor)
                for k,v in a_j.items():
                    a_j[k]  = v +self.workers[k].base_score
                    sum_a += a_j[k]
                updates=0
                # print(f"requester {self.ID} total-num:{len(self.local_updates)} participants:{[k for k,_ in self.local_updates.items()]}  pass-num:{len(a_j)}")
                #(formula 18 & formula 19 )
                for k,v in a_j.items():
                    updates =  np.add(updates,v/sum_a*self.local_updates[k])
                    #set a_j to be local_updates
                    a_j[k]=self.local_updates[k]
                #reset the local_updates being actually used
                self.local_updates=a_j
                params=updates+self.params[-1]
        if flag:      
            # print(f"flag:{flag} requester {self.ID}'s selection of round {self.rounds}:",end=" ")
            self.rounds += 1 
            # for k,v in self.local_updates.items():
            #     print(f"{k},accuracy:{self.workers[k].accuracy}",end=" ")
            # print("")            
            self.params.append(params)
            reconstruct_model(flattened_params=params,model=self.global_model)
        else:
            for worker in self.workers:
                worker.rounds_stay(self)
            # print(f"flag:{flag} requester {self.ID}'s bad selection of round {self.rounds}:",end=" ")
            if len(self.local_updates)==0: 
                reconstruct_model(model=self.global_model,flattened_params=self.params[-1])
            else:
                l=0
                for k,v in self.local_updates.items():
                    l=np.add(l,v)
                #     print(f"{k},accuracy{self.workers[k].accuracy}",end=" ")
                # print("")    
                params=self.params[-1]+l/len(self.local_updates)
                self.params.append(params)
                reconstruct_model(flattened_params=params,model=self.global_model)
             
        self.results[mode].append(self.global_model.test_nn(self.test_set))    
        return self.global_model.test_nn(self.test_set)
        
           
                
     ### Reputation computation        
    # cal y_j (formula 25 (in other words formula 23)) a=1,b=-1,c=-5.5
    def cal_y_j(self,model_j):
        #(formula 24)
        divisor=(model_j.success[self.ID]*self.omega+(1-self.omega)*model_j.fail[self.ID])
        ka=0
        if divisor:
            ka=(model_j.success[self.ID]*self.omega-(1-self.omega)*model_j.fail[self.ID])/divisor
        #a=1,b=-1,c=-5.5
        return 1*math.exp(-1*(math.exp(-5.5*ka)))
        
    #(formula 26 for all used in formula 5)    get the historical r_e_(i,j) 
    def cal_eval_reputation_all(self):
        contri=[]
        workers=self.workers
        for model_j in workers:
            contri.append(0)
            for i in range(self.rounds):
                #(formula 20 difference between current and final global model)    
                contri[-1] += cosine_similarity(model_j.updates[self.ID][i], self.params[-1]-self.params[i])*vector_projection(model_j.updates[self.ID][i],self.params[-1]-self.params[i])        
            #(formula 21)
            contri[-1] =max(0,contri[-1])
       #(formula 22) 
        z_j=max_standard(contri)
        for i in range(sum(self.num_per_type)):
            # (formula 25 && formula 26)
            y_j=self.cal_y_j(workers[i])
            self.eval_models[i].eval_reputation=z_j[i]*y_j  
    
    #(formula 5) get the R_e_(i,j)
    def cal_direct_reputation_all(self):
        for i in range(sum(self.num_per_type)) :
            self.eval_models[i].direct_reputation=self.Lambda*self.eval_models[i].direct_reputation+(1-self.Lambda)*self.eval_models[i].eval_reputation
    
    #mean of dir
    def cal_mean_direct_reputation(self):
        total=0    
        for ev_mod in self.eval_models:
            total += ev_mod.direct_reputation
        self.mean_dir_rep=total/len(self.eval_models)
    
    #(formula 8 & formula 9) 
    #after all the direct_reputation is calculate
    # only calculate phi for recommender
    def cal_phi_all(self,set_req):
        for requester in set_req.requesters:
            phi=0
            iset=0
            kset=0
            #it is the recommender phi_(i,k) for every k, i can only be the requester
            if requester.ID!=self.ID:
                for ev_mod in requester.eval_models:
                    #ev_mod.interaction for recommender k's(model)
                    if  ev_mod.interaction==self.eval_models[ev_mod.ID].interaction and ev_mod.interaction==1:
                        phi += (self.eval_models[ev_mod.ID].direct_reputation-self.mean_dir_rep)*(ev_mod.direct_reputation -requester.mean_dir_rep)
                    if  ev_mod.interaction==1:
                        kset += (ev_mod.direct_reputation-requester.mean_dir_rep)**2
                    if self.eval_models[requester.ID].interaction==1:
                        iset += (self.eval_models[requester.ID].direct_reputation-self.mean_dir_rep)**2      
                if kset and iset:
                    set_req.visit[self.ID][requester.ID]
                    #calculate thr divisor of phi'
                    divisor=math.sqrt(kset)*math.sqrt(iset)
                    self.phi[requester.ID]=max(0,phi/divisor)
                else:
                    self.phi[requester.ID]=0
    
    #(weight between formula 9 and formula 10)
    def cal_weight_all(self,set_req):
        divisor=0
        for i in range(self.num_requester): 
            self.weight[i] = self.phi[i]*set_req.page_rank[i]   
            divisor +=  self.weight[i]
        for i in range(self.num_requester):
            if divisor:
                self.weight[i] /= divisor
    
    #(cal the indirect reputation between formula 9 and formula 10)
    #After all the weight  of all model's direct_reputation is calculate
    def cal_indirect_reputation_all(self,set_req):
        self.cal_phi_all(set_req)
        self.cal_weight_all(set_req)
        #initialize
        for requester in set_req.requesters:
            self.eval_models[requester.ID].indirect_reputation=0
            
        for requester in set_req.requesters:    
            for ev_mod in requester.eval_models:
                #(ev_mod is the analysis of model)
                self.eval_models[ev_mod.ID].indirect_reputation += self.weight[requester.ID]*ev_mod.direct_reputation 
                # self.eval_models[ev_mod.ID].indirect_reputation += self.eval_models[requester.ID].weight*ev_mod.direct_reputation 
               
        
    #(formula 10)(Neffective of last round?)
    def cal_beta(self):
        self.beta=-2*(1-self.beta1)/math.pi*math.atan(self.num_effective/(self.beta2*math.pi))+1
  
    #(formula 11)
    def cal_comprehensive_reputation_all(self):
        for ev_mod in self.eval_models:
            ev_mod.comprehensive_reputation=self.beta*ev_mod.direct_reputation+(1-self.beta)*ev_mod.indirect_reputation
class Worker:
    #init
    def __init__(self,input_size,hidden_size,output_size,accuracy,data,ID,type_ID,num_requesters,base_score=1,epochs=1,learning_rate=0.005,\
        range_of_bid={"low":4,"high":6},batch_size=100,mode="MNIST"):
        self.base_score=base_score
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_requesters=num_requesters
        self.models=[]
        #keep on e model for a requester
        if mode=="MNIST":
            for i in range(num_requesters):
                self.models.append(SimpleNN_MNIST(input_size=input_size,hidden_size=hidden_size,output_size=output_size).cuda())
        else:
            for i in range(num_requesters):
                self.models.append(CNN_CIFAR10())
        self.accuracy=accuracy
        self.batch_size=batch_size
        self.train_test_acu=change_labels(data['train'],percentage=1-accuracy)
        self.train_set=DataLoader(dataset=self.train_test_acu,shuffle=True,batch_size=self.batch_size)
        self.ID=ID
        self.type_ID=type_ID
        self.range_of_bid=range_of_bid
        self.bid=random.uniform(self.range_of_bid["low"],self.range_of_bid["high"])
        self.data=data
        self.success=[0 for i in range(self.num_requesters)]
        self.fail=[0 for i in range(self.num_requesters)]
        self.updates=[[] for i in range(self.num_requesters)]
        #for ALG test use
        self.bid_test=[self.bid]
        
    def restart(self):
        self.updates=[[] for i in range(self.num_requesters)]
        
    def rounds_stay(self,requester):
        self.updates[requester.ID].pop()
        
    def generate_bid(self):
        self.bid=random.uniform(self.range_of_bid["low"],self.range_of_bid["high"])
        self.bid_test.append(self.bid)
    
    ###upload and download(self)  
    def download(self,requester):
        self.updates[requester.ID].append(requester.params[-1])
        ##set the params to global params
        reconstruct_model(flattened_params=requester.params[-1],model=self.models[requester.ID])
    
    #training    
    def train(self,requester):
        #change the download param
        self.models[requester.ID].train_nn(self.train_set,epochs=self.epochs,learning_rate=self.learning_rate)
        self.updates[requester.ID][-1]=flatten_parameters(self.models[requester.ID])-self.updates[requester.ID][-1]
    
    # def upload(self,global_model):
        ##Due to the inastance in global is excact the reference of local you don't need to tell the global
        ##you only need to calculate the global and change the visit matrix,which is integrate in the global model
        
          

               
               
    

        
    
               