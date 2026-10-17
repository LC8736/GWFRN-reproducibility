"""Author-source neural benchmarks; proposed b/alpha grids are never read.

GNNWR uses the unmodified author's model class and SWNN, with NumPy OLS
as an equivalent backend and irrelevant eager plotting imports omitted.
SANNWR and FNN are disclosed ports, not claimed to be author executables.
Only the three neural benchmarks used in the seven-method experiment are dispatched.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import datetime
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import types
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import functional_integrals, pairwise_euclidean
from native_linear import features
from settings import CONFIG

ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def author_gnnwr():
    name = "author_gnnwr"
    if name+".models" in sys.modules:
        return sys.modules[name+".models"], sys.modules[name+".datasets"]
    folder = ROOT / "upstream/gnnwr/src/gnnwr"
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    loaded = {}
    for sub in ("networks","datasets"):
        spec = importlib.util.spec_from_file_location(name+"."+sub,folder/(sub+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[sub] = module

    class OLS:
        def __init__(self, dataset, xName, yName):
            design = np.column_stack((dataset[xName].values,np.ones(len(dataset))))
            # Same unpenalised OLS, intercept last, as author's sklearn OLS.
            self.params = np.linalg.lstsq(design,dataset[yName[0]].values,rcond=None)[0].tolist()

    namespace = dict(__name__=name+".models",datetime=datetime,os=os,pd=pd,np=np,
                     torch=torch,nn=nn,optim=torch.optim,warnings=warnings,
                     SummaryWriter=SummaryWriter,tqdm=tqdm,OrderedDict=OrderedDict,
                     logging=logging,OLS=OLS,**{k:getattr(loaded["networks"],k)
                     for k in ("SWNN","STPNN","STNN_SPNN")},
                     baseDataset=loaded["datasets"].baseDataset,
                     predictDataset=loaded["datasets"].predictDataset)
    # The official DIAGNOSIS and GNNWR classes are evaluated unchanged. This
    # does not import or alter any Windows-blocked DLL or OS security setting.
    import math
    namespace["math"] = math
    for filename,cls in (("utils.py","DIAGNOSIS"),("models.py","GNNWR")):
        source = folder/filename
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
    module = types.ModuleType(name+".models")
    module.__dict__.update(namespace)
    sys.modules[module.__name__] = module
    return module,loaded["datasets"]


def fit_gnnwr(data,seed,output):
    seed_all(seed)
    models,datasets = author_gnnwr()
    raw = np.column_stack((data.bspline_coef,data.z))
    columns = [f"x{k}" for k in range(raw.shape[1])]
    frame = pd.DataFrame(raw,columns=columns)
    frame["y"] = data.y.copy()
    frame.loc[data.test_idx,"y"] = 0.0 # true test labels never reach the package
    frame["u"],frame["v"] = data.coords.T
    frame["id"] = np.arange(len(frame))

    def build(fit_idx,val_idx,name):
        train,val,test = datasets.init_dataset_split(frame.iloc[fit_idx].copy(),
            frame.iloc[val_idx].copy(),frame.iloc[data.test_idx].copy(),
            columns,["y"],spatial_column=["u","v"],id_column=["id"])
        model = models.GNNWR(train,val,test,use_gpu=False,
            model_name=name,model_save_path=str(Path(output,name,"models")),
            write_path=str(Path(output,name,"tensorboard")),
            log_path=str(Path(output,name,"logs")))
        return model,train,val,test

    fi,vi = data.cv_folds[0]
    model,_,_,_ = build(fi,vi,"selection")
    model.run(CONFIG["baseline_max_epochs"],early_stop=CONFIG["baseline_patience"])
    best_epoch = int(np.argmin(model._validLossList))+1
    del model
    seed_all(seed+1)
    model,_,_,target = build(data.model_idx,data.model_idx,"refit")
    model.run(best_epoch,model_selection="last")
    target.shuffle = False
    target.dataloader = torch.utils.data.DataLoader(target,batch_size=len(target),shuffle=False)
    predicted = model.predict(target)
    prediction = predicted.set_index("id").loc[data.test_idx,"denormalized_pred_result"].to_numpy()
    model._writer.close()
    return dict(test_prediction=prediction,selected=dict(epochs=best_epoch,
        selection="native best validation R2 (equivalent to minimum MSE on this fixed fold)"),
        implementation="author gnnwr GNNWR class + SWNN, native automatic architecture/optimiser; NumPy OLS backend",
        protocol=dict(proposed_candidate_grid_used=False,optimizer="Adagrad",learning_rate=.1,
                      dropout=.2,batch_norm=True,weight_decay=.001,
                      selection_max_epochs=CONFIG["baseline_max_epochs"],
                      validation_patience=CONFIG["baseline_patience"]))


def fit_fnn(data,seed,output):
    seed_all(seed)
    raw = np.column_stack((functional_integrals(data.x),data.z))
    fi,vi = data.cv_folds[0]
    mean,scale = raw[fi].mean(0),raw[fi].std(0,ddof=1)
    scale[scale<1e-12] = 1.0
    x = torch.tensor((raw-mean)/scale,dtype=torch.float32)
    y = torch.tensor(data.y[fi],dtype=torch.float32)
    yv = torch.tensor(data.y[vi],dtype=torch.float32)
    model = nn.Sequential(nn.Linear(raw.shape[1],64),nn.Sigmoid(),nn.Linear(64,64),nn.Linear(64,1))
    for layer in model:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
    generator = np.random.default_rng(seed)
    best,wait,history = float("inf"),0,[]
    for epoch in range(100): # FuncNN::fnn.fit default, not the proposed budget
        model.train()
        order = generator.permutation(len(fi))
        for start in range(0,len(order),32):
            subset = torch.tensor(order[start:start+32])
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[fi[subset]]).ravel()-y[subset]).square().mean()
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = float((model(x[vi]).ravel()-yv).square().mean())
        history.append(vl)
        if vl<best:
            best,wait = vl,0
        else:
            wait+=1
        if wait>=15:
            break
    # The native callback has restore_best_weights=False: its stopping epoch
    # is retained. As for all other benchmarks, refit all outer-development
    # rows for the selected number of epochs, without inspecting test labels.
    stopping_epoch = epoch+1
    seed_all(seed+1)
    fi = data.model_idx
    mean,scale = raw[fi].mean(0),raw[fi].std(0,ddof=1)
    scale[scale<1e-12] = 1.0
    x = torch.tensor((raw-mean)/scale,dtype=torch.float32)
    y = torch.tensor(data.y[fi],dtype=torch.float32)
    model = nn.Sequential(nn.Linear(raw.shape[1],64),nn.Sigmoid(),nn.Linear(64,64),nn.Linear(64,1))
    for layer in model:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
    generator = np.random.default_rng(seed+1)
    for epoch in range(stopping_epoch):
        order = generator.permutation(len(fi))
        for start in range(0,len(order),32):
            subset = torch.tensor(order[start:start+32])
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[fi[subset]]).ravel()-y[subset]).square().mean()
            loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x[data.test_idx]).ravel().numpy()
    torch.save(model.state_dict(),Path(output,"fitted_model.pt"))
    return dict(test_prediction=prediction,selected=dict(epochs=epoch+1,best_validation_mse=best),
        implementation="disclosed PyTorch port of public FuncNN::fnn.fit defaults; common M=15 B-spline integrals; full-development refit",
        protocol=dict(proposed_candidate_grid_used=False,widths=[64,64],
                      activations=["sigmoid","linear"],learning_rate=.001,
                      max_epochs=100,patience=15,restore_best_weights=False,
                      spatial_validation_replaces_native_random_validation=True,
                      n_final_fit=len(fi)))


def fit_sannwr(data,seed,output):
    # No public author executable was located for SANNWR. Keep the existing
    # documented paper architecture and the user's approved 3000/early-stop
    # budget. This method has no bandwidth or alpha grid in the first place.
    import neural_methods as methods
    config = methods.NeuralConfig(epochs=CONFIG["baseline_max_epochs"],
        patience=CONFIG["baseline_patience"],batch_size=16,learning_rate=.02)
    result = methods.fit_sannwr(data,config,seed)
    result.pop("train_prediction",None)
    result["implementation"] = "SANNWR published SAPDNN 2-3-1 + SWNN 50-20 architecture; disclosed PyTorch reproduction"
    result["protocol"] = dict(proposed_candidate_grid_used=False,
        public_author_code_found=False,approved_max_epochs=3000,validation_patience=100)
    return result


def fit_native(method,data,seed,output):
    return {"FNN":fit_fnn,"GNNWR":fit_gnnwr,"SANNWR":fit_sannwr}[method](data,seed,output)
