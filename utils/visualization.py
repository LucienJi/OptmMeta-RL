import torch.nn as nn
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.autograd as autograd

def jacobian_bz(input, output,create_graph = False):
    B, N = output.shape
    y = output
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = autograd.grad(y,input,
                       grad_outputs=v,
                       retain_graph=True,
                       create_graph=create_graph,
                       allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)

    jacobian = torch.stack(jacobian, dim=1)
    if create_graph:
        jacobian.requires_grad_()
    return jacobian

def jacobian_norm(input:torch.Tensor,output,input_name = 'Test'):
    # input.shape = (bz,in_dim), out.shape = (bz,out_dim)
    if not input.requires_grad:
        input.requires_grad_()
    jacob = jacobian_bz(input,output,create_graph=False) # shape = (bz,out_dim,in_dim)
    jacob_norm = torch.norm(jacob,dim = -1) # shape (bz,out_dim)
    jacob_norm = jacob_norm.detach().cpu().numpy()
    jacob_index = np.arange(0,jacob_norm.shape[-1]).reshape(1,-1)
    jacob_index = np.tile(jacob_index,reps=(jacob_norm.shape[0],1))

    jacob_norm = jacob_norm.reshape(-1)
    jacob_index = jacob_index.reshape(-1)
    df = np.stack([jacob_norm,jacob_index],axis = 1)

    df = pd.DataFrame(df,columns=['Norm','Output'])
    df['Input'] = input_name
    return df 

def policy_jacob(state:torch.Tensor,env_feature:torch.Tensor,actions:torch.Tensor,Policy_name = "test"):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    df_state = jacobian_norm(state,actions,'State')
    df_envfeature = jacobian_norm(env_feature,actions,'Embedding')
    df = pd.concat([df_state,df_envfeature],ignore_index=True)
    sns.boxplot(data=df,x = 'Output',y = 'Norm' ,hue = 'Input')
    ax.set_title(f"{Policy_name} Gradient Norm wrt Input")
    return fig



def emb_similarity(embs_sample,embs_mean, change_inds = None ):
    # embs_sample.shape = (n_step,dim)
    # embs_mean.shape = (n_env,dim)
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    emb_sim = np.matmul(embs_sample,embs_mean.T) # (n_step,n_env)
    norm_1 = (embs_sample * embs_sample).sum(-1)
    norm_1 = np.expand_dims(norm_1, axis=-1)
    norm_2 = (embs_mean * embs_mean).sum(-1)
    norm_2 = np.expand_dims(norm_2, axis=-1)
    norm = np.matmul(norm_1,norm_2.T)
    norm = np.sqrt(norm)
    emb_sim = emb_sim / norm
    step = np.arange(emb_sim.shape[0])
    for i in range(embs_mean.shape[0]):
        sns.scatterplot(x = step,y = emb_sim[:,i],ax = ax ,label=f'id: {i}')
    if change_inds is not None:
        for ind in change_inds:
            ax.plot([ind, ind], [-1.1, 1.1], 'k--', alpha=0.2)
    ax.set_title("Embedding Similarity")
    return fig 

def embedding_variation(embs):
    # embs.shape (n_step,dim)
    # plt.cla()
    n_sample = embs.shape[0]
    n_dim = embs.shape[-1]
    steps = np.arange(n_sample)

    total_steps = np.tile(steps,n_dim)
    dims = []
    total_embs = []
    for i in range(n_dim):
        dims += [f'{i}' for _ in range(n_sample)]
        total_embs.append(embs[:,i])
    total_embs = np.concatenate(total_embs,axis = 0)
    df = {
        'embs':total_embs,
        'dim':dims,
        'step':total_steps
    }
    df = pd.DataFrame(df)
    g = sns.FacetGrid(df, row="dim",
                  height=1.7, aspect=4)
    g.map(sns.scatterplot, "step",'embs')
    g.set_xlabels('Time step')
    g.set_ylabels("value")
    return g.fig


def pca_analysis(embs,ids,need_pca = True,need_tsne = False):
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    plt.cla()
    if type(embs) == list:
        n_env = len(embs)
        n_sample = embs[0].shape[0]
        env_id  = np.ones((n_env,n_sample))
        for i,id in enumerate(ids):
            env_id[i] = env_id[i] * id
        env_id = env_id.reshape(-1)
        ids = env_id
        embs = np.concatenate(embs,axis=0)
        embs = embs.reshape((n_env * n_sample,-1))
    if need_pca:
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        embs = scaler.fit_transform(embs)
        principalComponents = pca.fit_transform(embs) 
    elif need_tsne:
        tsne = TSNE(n_components=2,perplexity=30,n_iter=300)
        principalComponents = tsne.fit_transform(embs)
    else:
        principalComponents = embs 
    sns.scatterplot(x= principalComponents[:,0],
                    y=principalComponents[:,1],
                    hue=ids.reshape(-1),
                    palette="deep",
                    ax = ax)
    ax.legend(loc = 'upper right',title = 'Task Id')
    ax.set_title(f"Embedding Analysis")
    return fig


    
    

def times_series_value(values,change_inds = None,name = 'Prediction Error'):
    # values 
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    # plt.cla()
    if type(values)==list:
        values = np.array(values) 
    values = values.reshape(-1)
    step = np.arange(values.shape[0])

    sns.scatterplot(x = step,y=values,ax = ax,label = name)
    max_value = np.max(values)
    
    if change_inds is not None:
        for ind in change_inds:
            ax.plot([ind,ind],[-1.1,1.1],'k--',alpha = 0.2)
    # plt.show()
    return fig

def action_discrepancy(embs,real_param,action_discrepancy,change_inds):
    fig = plt.figure()
    plt.cla()
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    embs = scaler.fit_transform(embs)
    principalComponents = pca.fit_transform(embs) 
    plt.plot(principalComponents[:, 0], label='x')
    plt.plot(principalComponents[:, 1], label='y')
    if real_param is not None:
        plt.plot(real_param[:, -1], label='real')
    if action_discrepancy is not None:
        action_discrepancy = np.array(action_discrepancy)
        abs_res = np.abs(action_discrepancy[:, 0]) / 3 + np.abs(action_discrepancy[:, 1]) / 3
        plt.plot(np.arange(action_discrepancy.shape[0]), abs_res, '-*', label='diff')
        plt.title('mean discrepancy: {:.3f}'.format(np.mean(abs_res)))
    else:
        plt.title('Embedding variation')
    for ind in change_inds:
        plt.plot([ind, ind], [-1.1, 1.1], 'k--', alpha=0.2)
    plt.legend()
    return fig

    
def embedding_with_parameters(embs,ids,paras,need_pca = True):
    raise NotImplementedError






