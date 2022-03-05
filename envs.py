import numpy as np
import math
from math import exp
import gym
from gym import spaces
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

class WLFU(gym.Env):
        def __init__(self,total_matrix):

            
                self.pnter = 48 # 计步器
                self.total_matrix = total_matrix
                self.fnum = self.total_matrix.shape[0]
                self.tnum = 48  # revised
                self.serve_ratio = 0.5*1e-1 # revised

#                 self.n_actions = 1   
                self.action_space = spaces.MultiDiscrete([ 31, 101]) # revised
                self.observation_space = spaces.Box(low=0,high = float("inf"),
                                                    shape = (self.fnum, self.tnum + 2), dtype=np.float32)
                self.reset()
                
                
            
        def reset(self):
                # 重置环境
                # Pnter
                self.pnter = 48
                
                self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]
#                 self.req_status = self.req_matrix[:, -1]
                self.req_status = np.zeros(self.fnum)
#                 self.norm_req_status = self.req_status / np.max(self.req_status)
                self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1)], axis = 1) 
                
                self.cache_idx = []
                self.cache_time = np.zeros(self.fnum)
                self.state = np.concatenate([self.state, self.cache_time.reshape(-1,1)], axis = 1) 
                
                # DONE
                self.done = 0

                return self.state
        
        def decay_price(self, T):
            para = 0.999888
            a = 0.017
            bias = 0.01
            unit_p = a * pow(para,T) + bias 
            return unit_p

        def step(self,action): 
            capacity = action[0]
            alpha = action[1] * 0.01
            cache_enabled = np.where(self.req_status!=0)[0]
            capacity = min(capacity, len(cache_enabled))


            
            self.pnter += 1
            self.req_status = self.req_matrix[:,-1]  + alpha * self.req_status  # 没有用到t+1信息决策
            self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]
            
            
            tmp = np.argsort(self.req_status) # revised: 和上一段调换位置
            tmp = np.flip(tmp)
            index = tmp[:capacity]
            
  
            old_cost = 0
            new_cost = 0
            # reward computation
            real_req = self.req_matrix[:, -1]


            for k in index:
                if k in self.cache_idx:
                    old_cost += self.decay_price(self.cache_time[k])
                    self.cache_time[k] += 1
                else:
                    new_cost += self.decay_price(0)
                    self.cache_time[k] = 1
            for r in self.cache_idx:
                if r not in index:
                    self.cache_time[r] = 0
            self.cache_idx = index
            gain = np.sum(real_req[index])
            gain = gain * self.serve_ratio 
            cost = new_cost + old_cost

            reward = gain - cost
            
            # DONE
            if self.pnter == self.total_matrix.shape[1] - self.tnum:
                self.done = 1

#             self.norm_req_status = self.req_status / np.max(self.req_status)
            self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1)], axis = 1)
            self.state = np.concatenate([self.state, self.cache_time.reshape(-1,1)], axis = 1) 

            return self.state, reward , self.done, {}
        
class Test_WLFU(gym.Env):
        def __init__(self,total_matrix):

                self.tnum = 48

                self.total_matrix = total_matrix
                self.fnum = self.total_matrix.shape[0]
                self.pnter = 0
                self.serve_ratio = 0.5*1e-1

                self.n_actions = 1   
                self.action_space = spaces.MultiDiscrete([ 31, 101]) 
                self.observation_space = spaces.Box(low=0,high = float("inf"),
                                                    shape = (self.fnum, self.tnum + 2), dtype=np.float32)
                self.reset()             
            
        def reset(self):
                # 重置环境
                # Pnter
                self.pnter = 0
                
                self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]
#                 self.req_status = self.req_matrix[:, -1]
                self.req_status = np.zeros(self.fnum)
#                 self.norm_req_status = self.req_status / np.max(self.req_status)
                self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1)], axis = 1) 
                
                self.cache_idx = []
                self.cache_time = np.zeros(self.fnum)
                self.state = np.concatenate([self.state, self.cache_time.reshape(-1,1)], axis = 1) 
                
                # DONE
                self.done = 0

                return self.state
        
        
        def decay_price(self, T):
            para = 0.999888
            a = 0.017
            bias = 0.01
            unit_p = a * pow(para,T) + bias 
            return unit_p

        def step(self,action): 
            capacity = action[0] 
            alpha = action[1] * 0.01
            cache_enabled = np.where(self.req_status!=0)[0]
            capacity = min(capacity, len(cache_enabled))
                           
            self.pnter += 1
            self.req_status = self.req_matrix[:,-1]  + alpha * self.req_status  # 没有用到t+1信息决策
            self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]
            
            
            tmp = np.argsort(self.req_status) # revised: 和上一段调换位置
            tmp = np.flip(tmp)
            index = tmp[:capacity]

  
            old_cost = 0
            new_cost = 0
            # reward computation
            real_req = self.req_matrix[:, -1]


            for k in index:
                if k in self.cache_idx:
                    old_cost += self.decay_price(self.cache_time[k])
                    self.cache_time[k] += 1
                else:
                    new_cost += self.decay_price(0)
                    self.cache_time[k] = 1
            for r in self.cache_idx:
                if r not in index:
                    self.cache_time[r] = 0
            self.cache_idx = index

            gain = np.sum(real_req[index])
            gain = gain * self.serve_ratio 
            cost = new_cost + old_cost

            reward = gain - cost


            # DONE
            if self.pnter == 48:
                self.done = 1
            self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1)], axis = 1)
            self.state = np.concatenate([self.state, self.cache_time.reshape(-1,1)], axis = 1) 

            return self.state, reward , self.done, {}
        
        
class WLFU_S1(gym.Env):
        def __init__(self,total_matrix):
            self.pnter = 48 # 计步器
            self.total_matrix = total_matrix
            self.fnum = self.total_matrix.shape[0]
            self.tnum = 48  # revised
            self.serve_ratio = 0.186329 * 0.01 # revised
            self.action_space = spaces.MultiDiscrete([ 31, 101]) # revised
            self.observation_space = spaces.Box(low=0,high = float("inf"),
                                                shape = (self.fnum, self.tnum + 2), dtype=np.float32)
            self.reset()         
            
        def reset(self):
            # 重置环境
            self.pnter = 48
            self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]
            self.req_status = np.sum(self.total_matrix[:,self.pnter:self.pnter + self.tnum],axis=1)
            self.index_01 = np.zeros(self.fnum)
            self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1), self.index_01.reshape(-1,1)], axis = 1) 

            # DONE
            self.done = 0
            return self.state
        

        def step(self,action): 
            # 存了哪些内容
            capacity = action[0] * 50
            alpha = action[1] * 0.01
            if self.pnter == 48:
                cache_enabled = np.where(self.req_status!=0)[0]
                real_capacity = min(capacity, len(cache_enabled))
                tmp = np.argsort(self.req_status)
                tmp = np.flip(tmp)
                self.index = tmp[:real_capacity] 
            else:
                batch1_len = len(np.where(self.req_status!=0)[0])
                batch1 = np.argsort(self.req_status)
                batch1 = np.flip(batch1)
                batch1 = batch1[:batch1_len]
                batch1_set = set(batch1)
                batch2 = np.array([x for x in self.index if x not in batch1_set])
                batch = np.concatenate([batch1,batch2]).astype('int64')
                real_capacity = min(capacity, len(batch))
                self.index = batch[:real_capacity]             
            
#             print('榜单:',np.where(self.req_status!=0)[0])
#             print('缓存列表:',self.index)
#             print('capacity:',capacity)
#             print('real_capacity:',real_capacity)

            # 更新下一时刻状态
            self.pnter += 1
            
            if self.pnter == self.total_matrix.shape[1] - self.tnum:
                self.done = 1
                return self.state, 0 , self.done, {}
            
            else: 
                self.index_01 = np.zeros(self.fnum)
                self.index_01[self.index] = 1
                self.req_status = self.total_matrix[:,self.pnter + self.tnum]  + alpha * self.req_status  # 没有用到t+1信息决策
                self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]                      
                       
  
                # reward computation
                real_req = self.total_matrix[:,self.pnter + self.tnum]
                self.index = self.index.astype('int64')
                gain = np.sum(real_req[self.index])
                gain = gain * self.serve_ratio 
                cost = capacity * 1.5085234931887e-5
                reward = gain - cost
    
                self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1), self.index_01.reshape(-1,1)], axis = 1) 
                return self.state, reward , self.done, {}

class Test_WLFU_S1(gym.Env):
        def __init__(self,total_matrix):
            self.pnter = 0 # 计步器
            self.total_matrix = total_matrix
            self.fnum = self.total_matrix.shape[0]
            self.tnum = 48  # revised
            self.serve_ratio = 0.186329 * 0.01 # revised
            self.action_space = spaces.MultiDiscrete([ 31, 101]) # revised
            self.observation_space = spaces.Box(low=0,high = float("inf"),
                                                shape = (self.fnum, self.tnum + 2), dtype=np.float32)
            self.reset()         
            
        def reset(self):
            # 重置环境
            self.pnter = 0
            self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]
            self.req_status = np.sum(self.total_matrix[:,self.pnter:self.pnter + self.tnum],axis=1)
            self.index_01 = np.zeros(self.fnum)
            self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1), self.index_01.reshape(-1,1)], axis = 1) 

            # DONE
            self.done = 0
            return self.state
        

        def step(self,action): 
            # 存了哪些内容
            capacity = action[0] * 50
            alpha = action[1] * 0.01
            if self.pnter == 0:
                cache_enabled = np.where(self.req_status!=0)[0]
                real_capacity = min(capacity, len(cache_enabled))
                tmp = np.argsort(self.req_status)
                tmp = np.flip(tmp)
                self.index = tmp[:real_capacity] 
            else:
                batch1_len = len(np.where(self.req_status!=0)[0])
                batch1 = np.argsort(self.req_status)
                batch1 = np.flip(batch1)
                batch1 = batch1[:batch1_len]
                batch1_set = set(batch1)
                batch2 = np.array([x for x in self.index if x not in batch1_set])
                batch = np.concatenate([batch1,batch2]).astype('int64')
                real_capacity = min(capacity, len(batch))
                self.index = batch[:real_capacity]             
            
#             print('榜单:',np.where(self.req_status!=0)[0])
#             print('缓存列表:',self.index)
#             print('capacity:',capacity)
#             print('real_capacity:',real_capacity)

            # 更新下一时刻状态
            self.pnter += 1
            
            if self.pnter == 48:
                self.done = 1
                return self.state, 0 , self.done, {}
            
            else: 
                self.index_01 = np.zeros(self.fnum)
                self.index_01[self.index] = 1
                self.req_status = self.total_matrix[:,self.pnter + self.tnum]  + alpha * self.req_status  # 没有用到t+1信息决策
                self.req_matrix = self.total_matrix[ :, self.pnter:self.pnter + self.tnum]                      
                       
  
                # reward computation
                real_req = self.total_matrix[:,self.pnter + self.tnum]
                self.index = self.index.astype('int64')
                gain = np.sum(real_req[self.index])
                gain = gain * self.serve_ratio 
                cost = capacity * 1.5085234931887e-5
                reward = gain - cost
    
                self.state = np.concatenate([self.req_matrix, self.req_status.reshape(-1,1), self.index_01.reshape(-1,1)], axis = 1) 
                return self.state, reward , self.done, {}
    
    
    
