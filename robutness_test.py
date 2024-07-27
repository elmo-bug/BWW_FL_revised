from BMW_FL import *
import pickle
import copy
import pandas as pd
'''
[[3, 4, 1, 8], [2, 1, 3, 1, 2, 3, 1, 3], [3, 1, 3, 3, 3, 2, 3, 1, 3, 1, 3, 2], 
[2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

[[3, 3, 1, 3], [3, 1, 1, 1, 2, 1, 1, 3], [2, 1, 1, 2, 3, 3, 1, 3, 2, 3, 3, 2], [2, 3, 1, 3, 3, 2, 1, 3, 2, 1, 1, 1, 3, 2, 3, 1],
[2, 1, 1, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 2, 3, 1, 1, 3, 2, 2], [1, 1, 1, 2, 1, 3, 2, 3, 1, 2, 2, 1, 3, 3, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3]]
'''
if __name__=="__main__":   
    random.seed(22)
    total_workers=120
    num_per_type=[[30,30,30,30],[15 for _ in range(8)],[10 for _ in range(12)],[7+i%2 for i in range(16) ],[6 for _ in range(20)],[5 for i in range(24)]]
    type_select_time=[[random.randint(1,(int)(num_per_type[i][j]*0.3)) for j in range(len(num_per_type[i])) ]for i in range(len(num_per_type))]
    type_select_time=[[random.randint(1,3) for j in range(len(num_per_type[i])) ]for i in range(len(num_per_type))]
    print(type_select_time)
    budget_each=[80 for i in range(5)]
    groups={}
    
    rep_set=[]
    ##note that the cifar need more data to train a good model
    # train_size=1000
    train_size=2000
    test_size=2000
    val_size=2000
    batch_size=100
    epochs=2
    learning_rate=0.005
       
    ##differnet mode  
    ##param to tune
    #type of data non_iid(True) or accuracy of data non iid(False)
    non_iid_type=True
    mode_flag="CIFAR10"
    # mode_flag="FASHIONMNIST"
    mode_cold_start=False
    #flag to show if reputation need to be calculate,stay true if not for recover from killed
    flag=False
    if not(flag):
        with open(f"rep_set_{mode_flag}.pkl","rb") as file:
        # with open(f"rep_set.pkl","rb") as file:
            rep_set=pickle.load(file)    
            file.close()
        # with open(f"workers.pkl","rb") as file:    
        with open(f"workers_{mode_flag}.pkl","rb") as file:
            workers=pickle.load(file)    
            if mode_cold_start:
                workers_test=copy.deepcopy(workers)
            file.close()
    else:
        ##change data to get result of different types of data 
        if mode_flag=='MNIST':
            data=load_MNIST_data(num_individuals=total_workers,training_set_size=train_size,test_set_size=test_size,validation_set_size=val_size,non_IID=non_iid_type)
        elif mode_flag=="FASHIONMNIST":
            data=load_Fashion(num_individuals=total_workers,training_set_size=train_size,test_set_size=test_size,validation_set_size=val_size,non_IID=non_iid_type)
        elif mode_flag=="CIFAR10":
            ##note that the cifar need smaller learning rate to train a good model and more epochs to converge
            epochs=10 
            learning_rate=0.001
            data=load_CIFAR10(num_individuals=total_workers,training_set_size=train_size,test_set_size=test_size,validation_set_size=val_size,non_IID=non_iid_type)
        accuracy=[]
        high=[]
        low=[]
        #same dataset for workers
        for i in range(sum(num_per_type[0])):
            if non_iid_type==False:
                accuracy.append(random.randint(5,10)/10)
                data[i]['train']=change_labels(data[i]['train'],1-accuracy[-1])
                # accuracy=0.1+min(0.9,3*(int)(i/(sum(num_per_type)/6))/10)
                # print(accuracy)
                high.append(3+(int)(accuracy[i]*10/3))
                low.append(1+(int)(accuracy[i]*10/3))     
            else:
                accuracy.append(1)
                # if get_num_types(data[i])==10:
                #     #get_max_label_percent(data[i]) in [0.1,0.5+]
                #     high.append(3+(int)((6-(int)(get_max_label_percent(data[i])*10))*2/3))
                #     low.append(3+(int)((6-(int)(get_max_label_percent(data[i])*10))*2/3))
                # else:
                high.append(3+(int)(get_num_types(data[i])/3))
                low.append(1+(int)(get_num_types(data[i])/3))
                # print(f'high {high} low {low}')
                
        print("Data ready")    
        workers=[]
        workers_test=[]
        req_size=len(budget_each)
        for i in range(sum(num_per_type[0])):
            workers.append( Worker(learning_rate=learning_rate,accuracy=accuracy[i],data=data[i],ID=i,type_ID=get_type(num_per_type=num_per_type[0],ID=i),\
                range_of_bid={"high":high[i],"low":low[i]},batch_size=batch_size,num_requesters=req_size,epochs=epochs,mode=mode_flag,\
                    type_select_time=type_select_time[0][get_type(num_per_type=num_per_type[0],ID=i)]) )
            if mode_cold_start:
                workers_test.append( Worker(learning_rate=learning_rate,accuracy=accuracy[i],data=data[i],ID=i,type_ID=get_type(num_per_type=num_per_type[0],ID=i),\
                range_of_bid={"high":high[i],"low":low[i]},batch_size=batch_size,num_requesters=req_size,epochs=epochs,mode=mode_flag,\
                    type_select_time=type_select_time[0][get_type(num_per_type=num_per_type[0],ID=i)]) )
        # with open(f"workers_{mode_flag}.pkl", "wb") as file:
        with open(f"workers.pkl", "wb") as file:    
                pickle.dump(workers,file)
                file.close()
                print("worker backup ready")   
    mode=['BMW_FL_s','BMW_FL_g','RRAFL_seperate','RRAFL_overall','Ran_Pri_seperate','Ran_Pri_overall']
        #,'greedy_s','greedy_g']
        # mode=['BMW_FL_g']
    '''different division of group'''
    for index in range(len(num_per_type)):     
        budget_all=sum(budget_each)
        req_size=len(budget_each)
        data=[0]
        requesters=[]
        #_test for cold start
        requesters_test=[]
        for i in range(sum(num_per_type[0])):
            workers[i].type_ID=get_type(num_per_type=num_per_type[index],ID=i)
            workers[i].num_requesters=req_size
            workers[i].type_select_time=type_select_time[index][workers[i].type_ID]
            if mode_cold_start:
                workers_test[i].type_ID=get_type(num_per_type=num_per_type[index],ID=i)
                workers_test[i].num_requesters=req_size
                workers_test[i].type_select_time=type_select_time[index][workers_test[i].type_ID]
        for i in range(req_size):
            if mode_cold_start:
                requesters_test.append(Requester(ID=i,budget=budget_each[i],workers=workers_test,num_per_type=num_per_type[index],num_requester=req_size,data=workers[0].data,batch_size=batch_size,mode=mode_flag))
            requesters.append(Requester(ID=i,budget=budget_each[i],workers=workers,num_per_type=num_per_type[index],num_requester=req_size,data=workers[0].data,batch_size=batch_size,mode=mode_flag))        
        req_set=Request_Set(workers=workers,requesters=requesters,num_per_type=num_per_type[index],budget=budget_all)
        if mode_cold_start:
            req_test_set=Request_Set(workers=workers_test,requesters=requesters_test,num_per_type=num_per_type[index],budget=budget_all)
        #the reputation evaluationset fix across all divison of group
        if flag and not(mode_cold_start):
            for i in range(10):
                req_set.run(mode='get_rep',size_of_selection=(int)(sum(num_per_type[index])/10),mode_flag=mode_flag)    
            rep_set=req_set.rep
            flag=False    
            # with open(f"rep_set_{mode_flag}.pkl", "wb") as file:
            with open(f"rep_set.pkl", "wb") as file:    
                pickle.dump(rep_set,file)
                file.close()
                print("rep ready")     
        for i in range(10):
            # #for cold start
            if mode_cold_start:
                req_set.run(mode='get_rep',size_of_selection=(int)(sum(num_per_type[index])/10),mode_flag=mode_flag)
                req_test_set.rep=req_set.rep
                for worker_test,worker in zip(workers_test,workers):
                    worker_test.bid_test=worker.bid_test
                for mod in mode:
                    req_test_set.reset_for_ALG(i)
                    print(mod,end='\n\n')
                    print(f'round{i}')
                    req_test_set.run(mode=mod)    
            else:
                req_set.rep=rep_set 
                for mod in mode:
                    req_set.reset_for_ALG(i)
                    print(mod,end='\n\n')
                    print(f'round{i}')
                    req_set.run(mode=mod)     
       
        #get avg accuracy for all rounds 
        if mode_cold_start:
            ac=req_test_set.accuracy
            rep_per_round=req_test_set.rep_per_round
            del req_test_set
            del requesters_test
        else:
            ac=req_set.accuracy
            rep_per_round=req_set.rep_per_round   
        del req_set
        del requesters    
        final_data={}
        middle_data={}
        for k,v in ac.items():
            if len(v):
                final_data[k]=(sum(v)/len(v),sum(rep_per_round[k])/(len(rep_per_round[k])))
        for k,v in ac.items():
            if len(v):
                middle_data[k]=[(x,y) for x,y in zip(ac[k],rep_per_round[k])]          
        groups[len(num_per_type[index])]=(final_data)
        print(f'groups { groups[len(num_per_type[index])]}')   
        print(f'middle data{middle_data}')    
        df=pd.DataFrame.from_dict(groups)
        if mode_flag=='MNIST':
            df.to_csv("group_div.csv")
        elif mode_flag=="FASHIONMNIST":
            df.to_csv("fashion_group_div.csv")
        else:
            df.to_csv("cifar_group_div.csv")
        df=pd.DataFrame.from_dict(middle_data)
        if mode_flag=='MNIST':
            df.to_csv(f"group_div_{len(num_per_type[index])}.csv")
        elif mode_flag=="FASHIONMNIST":
            df.to_csv(f"fashion_group_div_{len(num_per_type[index])}.csv")
        else:
            df.to_csv(f"cifar_group_div_{len(num_per_type[index])}.csv")  
    '''num of requesters'''
    budget_each=[[40 for i in range(0,x)] for x in range(2,13,2)]
    num_per_type=[10 for i in range(12)]
    type_select_time=[random.randint(1,(int)(num_per_type[i]*0.3)) for i in range(len(num_per_type))]
    # flag=True
    reqs={}
    for x in budget_each:
        budget_all=sum(x)
        req_size=len(x)
        requesters=[]
        requesters_test=[]
        for i in range(sum(num_per_type)):
            workers[i].type_ID=get_type(num_per_type=num_per_type,ID=i)
            workers[i].num_requesters=req_size
            workers[i].type_selcect_time=type_select_time[workers[i].type_ID]
            if mode_cold_start:
                workers_test[i].type_ID=get_type(num_per_type=num_per_type,ID=i)
                workers_test[i].num_requesters=req_size
                workers_test[i].type_selcect_time=type_select_time[workers_test[i].type_ID]
        for i in range(req_size):
            if mode_cold_start:
                requesters_test.append(Requester(ID=i,budget=x[i],workers=workers_test,num_per_type=num_per_type,num_requester=req_size,data=workers[0].data,batch_size=batch_size,mode=mode_flag))
            requesters.append(Requester(ID=i,budget=x[i],workers=workers,num_per_type=num_per_type,num_requester=req_size,data=workers[0].data,batch_size=batch_size,mode=mode_flag))   
        req_set=Request_Set(workers=workers,requesters=requesters,num_per_type=num_per_type,budget=budget_all)
        if mode_cold_start:
            req_test_set=Request_Set(workers=workers_test,requesters=requesters_test,num_per_type=num_per_type,budget=budget_all)
        #the reputation evaluationset fix across all num of requester
        if flag and not(mode_cold_start):
            for i in range(15):
                req_set.run(mode='get_rep',size_of_selection=(int)(sum(num_per_type)/10),mode_flag=mode_flag)
            rep_set=req_set.rep
            flag=False
        for i in range(10):
            # #for cold start
            if mode_cold_start:
                req_set.run(mode='get_rep',size_of_selection=(int)(sum(num_per_type)/10),mode_flag=mode_flag)
                req_test_set.rep=req_set.rep
                for worker_test,worker in zip(workers_test,workers):
                    worker_test.bid_test=worker.bid_test
                for mod in mode:
                    req_test_set.reset_for_ALG(i)
                    print(mod,end='\n\n')
                    print(f'round{i}')
                    req_test_set.run(mode=mod)    
            else:
                req_set.rep=rep_set
                for mod in mode:
                    req_set.reset_for_ALG(i)
                    print(mod,end='\n\n')
                    print(f'round{i}')
                    req_set.run(mode=mod) 
        #get avg accuracy for all rounds
        if mode_cold_start:
            ac=req_test_set.accuracy
            rep_per_round=req_test_set.rep_per_round
            del req_test_set
            del requesters_test
        else:
            ac=req_set.accuracy
            rep_per_round=req_set.rep_per_round    
        del req_set
        del requesters
        final_data={}
        for k,v in ac.items():
            if len(v):
                final_data[k]=(sum(v)/len(v),sum(rep_per_round[k])/(len(rep_per_round[k]))) 
        reqs[len(x)]=(final_data)
        print(f'reqs { reqs[len(x)]}')     
        print(reqs)    
        df=pd.DataFrame.from_dict(reqs)
        if mode_flag=='MNIST':
            df.to_csv("num_req.csv")
        elif mode_flag=="FASHIONMNIST":
            df.to_csv("fashion_num_req.csv")
        else:
            df.to_csv("cifar_num_req.csv")
        middle_data={}
        for k,v in ac.items():
            if(len(v)):
                middle_data[k]=[(x,y) for x,y in zip(ac[k],rep_per_round[k])]
        df=pd.DataFrame.from_dict(middle_data)
        if mode_flag=='MNIST':
            df.to_csv(f"num_req_{len(x)}.csv")
        elif mode_flag=="FASHIONMNIST":
            df.to_csv(f"fashion_num_req_{len(x)}.csv")
        else:
            df.to_csv(f"cifar_num_req_{len(x)}.csv")
     
    # '''(num of worker)/(num of requster) rate is constant change num of worker'''    
    # #no coldstart_test for this mode
    # budget_each=[[30 for j in range(i) ] for i in range(4,11,2)]
    # num_per_type=[[5 for j in range(i) ] for i in range(4,11,2)] 
    # # for i in range(len(budget_each)):
    # #     print(budget_each[i],num_per_type[i])
    # groups={}
    # for index in range(len(num_per_type)):
    #     print(f'num_per_type{num_per_type[index]},budget_each{budget_each[index]}')
    #     flag=True
    #     rep_set=[]
    #     accuracy=[]
    #     high=[]
    #     low=[]
    #     #same dataset for workers
    #     for i in range(sum(num_per_type[index])):
    #         if non_iid_type==False:
    #             accuracy.append(random.randint(1,10)/10)
    #             data[i]['train']=change_labels(data[i]['train'],1-accuracy[-1])
    #             high.append(3+(int)(accuracy[i]*10/3))
    #             low.append(1+(int)(accuracy[i]*10/3)) 
    #         else:
    #             accuracy.append(1)
    #             # if get_num_types(data[i])==10:
    #             #     #get_max_label_percent(data[i]) in [0.1,0.5+]
    #             #     high.append(3+(int)((6-(int)(get_max_label_percent(data[i])*10))*2/3))
    #             #     low.append(3+(int)((6-(int)(get_max_label_percent(data[i])*10))*2/3))
    #             # else:
    #             high.append(3+(int)(get_num_types(data[i])/3))
    #             low.append(1+(int)(get_num_types(data[i])/3))
    #     # print(accuracy,len(accuracy))   
    #     mode=['BMW_FL_s','BMW_FL_g','Ran_Pri_seperate','Ran_Pri_overall','RRAFL_seperate','RRAFL_overall']  
    #     budget_all=sum(budget_each[index])
    #     req_size=len(budget_each[index])
    #     workers=[]
    #     requester=[]
    #     for i in range(sum(num_per_type[index])):
    #         workers.append(Worker(learning_rate=learning_rate,accuracy=accuracy[i],data=data[i],ID=i,type_ID=get_type(num_per_type=num_per_type[index],ID=i),\
    #             range_of_bid={"high":high[i],"low":low[i]},batch_size=batch_size,num_requesters=req_size,mode=mode_flag,epochs=epochs))
    #     for i in range(req_size):
    #         requester.append(Requester(ID=i,budget=budget_each[index][i],workers=workers,num_per_type=num_per_type[index],num_requester=req_size,data=workers[0].data,batch_size=batch_size,mode=mode_flag))   
    #     req_set=Request_Set(workers=workers,requesters=requester,num_per_type=num_per_type[index],budget=budget_all)
    #     #the reputation evaluationset fix across all divison of group
    #     if flag:
    #         for i in range(15):
    #             req_set.run(mode='get_rep',size_of_selection=(int)(sum(num_per_type[index])/10),mode_flag=mode_flag)
    #         rep_set=req_set.rep
    #         req_set.reset_for_ALG(i)
    #         flag=False    
    #     req_set.rep=rep_set
    #     for mod in mode:
    #         print(mod,end='\n\n') 
    #         for i in range(10):
    #             print(f'round{i}')
    #             req_set.run(mode=mod)
    #             req_set.reset_for_ALG(i)
    
    #     #get avg accuracy for all rounds 
    #     ac=req_set.accuracy
    #     rep_per_round=req_set.rep_per_round
    #     final_data={}
    #     middle_data={}
    #     for k,v in ac.items():
    #         if len(v):
    #             final_data[k]=(sum(v)/len(v),sum(rep_per_round[k])/(len(rep_per_round[k])))
    #     for k,v in ac.items():
    #         if len(v):
    #             middle_data[k]=[(x,y) for x,y in zip(ac[k],rep_per_round[k])]          
    #     groups[len(num_per_type[index])]=(final_data)
    #     print(f'num_workers { groups[len(num_per_type[index])]}')   
    #     print(f'middle data{middle_data}')    
    #     df=pd.DataFrame.from_dict(groups)
    #     if mode_flag=='MNIST':
    #         df.to_csv("rate_div.csv")
    #     elif mode_flag=="FASHIONMNIST":
    #         df.to_csv("fashion_rate_div.csv")
    #     else:
    #         df.to_csv("cifar_rate_div.csv")
    #     df=pd.DataFrame.from_dict(middle_data)
    #     if mode_flag=='MNIST':
    #         df.to_csv(f"rate_div_with_group_size_{len(num_per_type[index])}.csv")
    #     elif mode_flag=="FASHIONMNIST":
    #         df.to_csv(f"fashion_rate_div_with_group_size_{len(num_per_type[index])}.csv")
    #     else:
    #         df.to_csv(f"cifar_rate_div_with_group_size_{len(num_per_type[index])}.csv")
        
       
    