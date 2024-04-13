from BMW_FL import *
import pickle
import pandas as pd
import matplotlib.pyplot as plt
if __name__=="__main__":   
    random.seed(22)
    total_workers=100
    num_per_type=[[20,20,20,20,20],[16,16,17,17,17,17],[14,14,14,14,14,15,15],[12,12,12,12,13,13,13,13],[11,11,11,11,11,11,11,11,12],[10,10,10,10,10,10,10,10,10,10],\
                  [9,9,9,9,9,9,9,9,9,9,10],[8,8,8,8,8,8,8,8,9,9,9,9],[7,7,7,7,8,8,8,8,8,8,8,8,8],[7,7,7,7,7,7,7,7,7,7,7,7,8,8],[6,6,6,6,6,7,7,7,7,7,7,7,7,7,7]]
    num_per_type=[num_per_type[i] for i in range(1,9,2)]
    budget_each=[28 for i in range(5)]
    groups={}
    flag=True
    rep_set=[]
    ##note that the cifar need more data to train a good model
    train_size=10000
    #train_size=1000
    test_size=2000
    val_size=2000
    batch_size=100
    input=28*28
    hidden=100
    output=10
    epochs=10
    ##note that the cifar need smaller learning rate to train a good model
    #learning_rate=0.005
    learning_rate=0.001
    
    ##change data to get result of different types of data 
    data=load_CIFAR10(num_individuals=100,training_set_size=train_size,test_set_size=test_size,validation_set_size=val_size)
    #data=load_MNIST_data(num_individuals=100,training_set_size=train_size,test_set_size=test_size,validation_set_size=val_size)
    #data=load_Fashion(num_individuals=100,training_set_size=train_size,test_set_size=test_size,validation_set_size=val_size)
    accuracy=[]
    high=[]
    low=[]
    #same dataset for workers
    for i in range(sum(num_per_type[0])):
        accuracy.append(random.randint(1,10)/10)
        # accuracy=0.1+min(0.9,3*(int)(i/(sum(num_per_type)/6))/10)
        # print(accuracy)
        high.append(3+(int)(accuracy[i]*10/3))
        low.append(1+(int)(accuracy[i]*10/3))
    print("Data ready")    
    mode=['BMW_FL_s','BMW_FL_g','Ran_Pri_seperate','Ran_Pri_overall','RRAFL_seperate','RRAFL_overall','greedy_s','greedy_g']
    
    '''different division of group'''
    for x in num_per_type:     
        budget_all=sum(budget_each)
        req_size=len(budget_each)
        workers=[]
        requester=[]
        for i in range(sum(x)):
            workers.append( Worker(input_size=input,hidden_size=hidden,output_size=output,accuracy=accuracy[i],data=data[i],ID=i,type_ID=get_type(num_per_type=x,ID=i),\
                range_of_bid={"high":high[i],"low":low[i]},batch_size=batch_size,num_requesters=req_size,epochs=epochs,mode='CIFAR') )
        for i in range(req_size):
            requester.append(   Requester(ID=i,budget=budget_each[i],workers=workers,num_per_type=x,num_requester=req_size,data=data[i],batch_size=batch_size,mode='CIFAR')  )   
        req_set=Request_Set(workers=workers,requesters=requester,num_per_type=x,budget=budget_all)
        #the reputation evaluationset fix across all divison of group
        if flag:
            for i in range(10):
                req_set.run(mode='get_rep',size_of_selection=(int)(sum(x)/10))
            rep_set=req_set.rep
            req_set.reset_for_ALG()
            flag=False
        v_max=max(rep_set)
        req_set.rep=rep_set
        print(rep_set)
        for mod in mode:
            print(mod,end='\n\n') 
            for i in range(10):
                print(f'round{i}')
                req_set.run(mode=mod)
                req_set.reset_for_ALG()
       
        #get avg accuracy for all rounds 
        ac=req_set.accuracy
        rep_per_round=req_set.rep_per_round
        final_data={}
        middle_data={}
        for k,v in ac.items():
            if len(v):
                final_data[k]=(sum(v)/len(v),sum(rep_per_round[k])/len(rep_per_round[k]))
        for k,v in ac.items():
            if len(v):
                middle_data[k]=[(x,y/v_max) for x,y in zip(ac[k],rep_per_round[k])]          
        groups[len(x)]=(final_data)
        print(f'groups { groups[len(x)]}')   
        print(f'middle data{middle_data}')    
        df=pd.DataFrame.from_dict(groups)
        #df.to_csv("fashion_group_div.csv") 
        # df.to_csv("group_div.csv")  
        df.to_csv("cifar_group_div.csv") 
        df=pd.DataFrame.from_dict(middle_data)
        # df.to_csv(f"groups_size_{len(x)}.csv") 
        #df.to_csv(f"fashion_groups_size_{len(x)}.csv")  
        df.to_csv(f"cifar_groups_size_{len(x)}.csv")  
    '''num of workers'''
    budget_each=[[28 for i in range(0,x+1)] for x in range(1,10,2)]
    num_per_type=[10,10,10,10,10,10,10,10,10,10]
    reqs={}
    for x in budget_each:
        budget_all=sum(x)
        req_size=len(x)
        workers=[]
        requester=[]
        for i in range(sum(num_per_type)):
            workers.append(Worker(input_size=input,hidden_size=hidden,output_size=output,accuracy=accuracy[i],data=data[i],ID=i,type_ID=get_type(num_per_type=num_per_type,ID=i),\
                range_of_bid={"high":high[i],"low":low[i]},batch_size=batch_size,num_requesters=req_size))
        for i in range(req_size):
            requester.append(Requester(ID=i,budget=x[i],workers=workers,num_per_type=num_per_type,num_requester=req_size,data=data[i],batch_size=batch_size))   
        req_set=Request_Set(workers=workers,requesters=requester,num_per_type=num_per_type,budget=budget_all)
        #the reputation evaluationset fix across all divison of group
        req_set.rep=rep_set
        for mod in mode:
            print(mod,end='\n\n') 
            for i in range(10):
                print(f'round{i}')
                req_set.run(mode=mod)
                req_set.reset_for_ALG()
        #get avg accuracy for all rounds
        ac=req_set.accuracy
        rep_per_round=req_set.rep_per_round
        final_data={}
        for k,v in ac.items():
            if len(v):
                final_data[k]=(sum(v)/len(v),sum(rep_per_round[k])/len(rep_per_round[k])) 
        reqs[len(x)]=(final_data)
        print(f'reqs { reqs[len(x)]}')     
        print(reqs)    
        df=pd.DataFrame.from_dict(reqs)
        # df.to_csv("num_req_greedy.csv")
        #df.to_csv("fashion_greedy_num_req.csv")
        df.to_csv("cifar_num_req.csv")
        middle_data={}
        for k,v in ac.items():
            if(len(v)):
                middle_data[k]=[(x,y) for x,y in zip(ac[k],rep_per_round[k])]
        df=pd.DataFrame.from_dict(middle_data)
        #df.to_csv(f"fashion_greedy_num_req_{len(x)}.csv")  
        # df.to_csv(f"greedy_num_req_{len(x)}.csv") 
        df.to_csv(f"cifar_num_req_{len(x)}.csv") 
     
    '''(num of worker)/(num of requster) rate is constant change num of worker'''    
    budget_each=[[28 for j in range(i) ] for i in range(4,11,2)]
    num_per_type=[[5 for j in range(i) ] for i in range(4,11,2)] 
    # for i in range(len(budget_each)):
    #     print(budget_each[i],num_per_type[i])
    groups={}
    for index in range(len(num_per_type)):
        print(f'num_per_type{num_per_type[index]},budget_each{budget_each[index]}')
        flag=True
        rep_set=[]
        train_size=1000
        test_size=2000
        val_size=2000
        batch_size=100
        input=28*28
        hidden=100
        output=10
        epochs=5
        learning_rate=0.005
        accuracy=[]
        high=[]
        low=[]
        #same dataset for workers
        for i in range(sum(num_per_type[index])):
            accuracy.append(random.randint(1,10)/10)
            high.append(3+(int)(accuracy[i]*10/3))
            low.append(1+(int)(accuracy[i]*10/3)) 
        # print(accuracy,len(accuracy))   
        mode=['BMW_FL_s','BMW_FL_g','Ran_Pri_seperate','Ran_Pri_overall','RRAFL_seperate','RRAFL_overall']  
        budget_all=sum(budget_each[index])
        req_size=len(budget_each[index])
        workers=[]
        requester=[]
        for i in range(sum(num_per_type[index])):
            workers.append(Worker(input_size=input,hidden_size=hidden,output_size=output,accuracy=accuracy[i],data=data[i],ID=i,type_ID=get_type(num_per_type=num_per_type[index],ID=i),\
                range_of_bid={"high":high[i],"low":low[i]},batch_size=batch_size,num_requesters=req_size))
        for i in range(req_size):
            requester.append(Requester(ID=i,budget=budget_each[index][i],workers=workers,num_per_type=num_per_type[index],num_requester=req_size,data=data[i],batch_size=batch_size))   
        req_set=Request_Set(workers=workers,requesters=requester,num_per_type=num_per_type[index],budget=budget_all)
        #the reputation evaluationset fix across all divison of group
        if flag:
            for i in range(10):
                req_set.run(mode='get_rep',size_of_selection=(int)(sum(num_per_type[index])/10))
            rep_set=req_set.rep
            req_set.reset_for_ALG()
            flag=False    
        v_max=max(rep_set)
        req_set.rep=rep_set
        for mod in mode:
            print(mod,end='\n\n') 
            for i in range(10):
                print(f'round{i}')
                req_set.run(mode=mod)
                req_set.reset_for_ALG()
    
        #get avg accuracy for all rounds 
        ac=req_set.accuracy
        rep_per_round=req_set.rep_per_round
        final_data={}
        middle_data={}
        for k,v in ac.items():
            if len(v):
                final_data[k]=(sum(v)/len(v),sum(rep_per_round[k])/len(rep_per_round[k]))
        for k,v in ac.items():
            if len(v):
                middle_data[k]=[(x,y/v_max) for x,y in zip(ac[k],rep_per_round[k])]          
        groups[len(num_per_type[index])]=(final_data)
        print(f'num_workers { groups[len(num_per_type[index])]}')   
        print(f'middle data{middle_data}')    
        df=pd.DataFrame.from_dict(groups)
        df.to_csv("fashion_rate_div.csv")   
        # df.to_csv("rate_div.csv")    
        df=pd.DataFrame.from_dict(middle_data)
        df.to_csv(f"fashion_rate_with_group_size_{len(num_per_type[index])}.csv") 
        # df.to_csv(f"rate_with_group_size_{len(num_per_type[index])}.csv")  
        
       
    
 
    
    

       

   