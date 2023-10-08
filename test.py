
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def DTWdist(x,y):
    l1 = len(x)
    l2 = len(y)
    M=np.zeros([l1, l2])
    D = np.zeros([l1, l2, 4])  
    x = x.expand(l2,-1,-1).permute(1,0,2)
    y = y.expand(l1,-1,-1)
    Mgpu = torch.sqrt(torch.sum((x - y).pow(2), dim=2))+1e-8
    M = Mgpu.cpu().numpy()

    D[0][0] = [M[0][0], -1, -1, 1]
    for i in range(1, l1):
        D[i][0][0] = M[i][0] + D[i - 1][0][0]
        D[i][0][1] = i - 1
        D[i][0][2] = 0
        D[i][0][3] = 1 + D[i - 1][0][3]
    for j in range(1, l2):
        D[0][j][0] = M[0][j] + D[0][j - 1][0]
        D[0][j][1] = 0
        D[0][j][2] = j - 1
        D[0][j][3] = 1 + D[0][j - 1][3]
    for i in range(1, l1):
        for j in range(1, l2):
            cand1 = D[i - 1][j][0] / D[i - 1][j][3]
            cand2 = D[i][j - 1][0] / D[i][j - 1][3]
            cand3 = D[i - 1][j - 1][0] / D[i - 1][j - 1][3]
            minValue = min(cand1, cand2, cand3)
            if minValue == cand1:
                D[i][j][0] = M[i][j] + D[i - 1][j][0]
                D[i][j][1] = i - 1
                D[i][j][2] = j
                D[i][j][3] = 1 + D[i - 1][j][3]
            elif minValue == cand2:
                D[i][j][0] = M[i][j] + D[i][j - 1][0]
                D[i][j][1] = i
                D[i][j][2] = j - 1
                D[i][j][3] = 1 + D[i][j - 1][3]
            elif minValue == cand3:
                D[i][j][0] = M[i][j] + D[i - 1][j - 1][0]
                D[i][j][1] = i - 1
                D[i][j][2] = j - 1
                D[i][j][3] = 1 + D[i - 1][j - 1][3]
    align={}
    ii=l1-1
    jj=l2-1
    while(ii!=-1):
        align.setdefault(ii, []).append(jj)  # if Key "ii" in dictionary, append "jj" to this list; or else new "ii: []" in dict and append "jj" to list
        ii,jj=(int(D[ii][jj][1]),int(D[ii][jj][2]))
    return align

def writeData(data,fileName):
    fp1 = open(fileName, 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            fp1.write(str(data[i][j])+'\t')
        fp1.write('\n')
    fp1.close()

def rerank(predictions,queries_features_a,database_features_a):
    pred2 = []
    dist2 = []
    for query_index, pred in enumerate(predictions):
        query_features_a = queries_features_a[query_index]
        positives_features_a = database_features_a[pred]
        query_features_a = torch.Tensor(query_features_a).cuda()
        positives_features_a = torch.Tensor(positives_features_a).cuda()
        Dist = []
        n = len(positives_features_a)
        for ii in range(n):
            img1, img2 = positives_features_a[ii], query_features_a
            xlen, ylen = img1.shape[0], img1.shape[1]

            img1x = img1.flatten(start_dim=1)
            img2x = img2.flatten(start_dim=1)
            img1y = img1.permute(1,0,2).flatten(start_dim=1)
            img2y = img2.permute(1,0,2).flatten(start_dim=1)
            xalign = DTWdist(img1x, img2x)
            yalign = DTWdist(img1y, img2y)

            img1 = img1.expand(xlen, ylen,-1,-1,-1).permute(2,3,0,1,4)
            img2 = img2.expand(xlen, ylen,-1,-1,-1)
            distm = torch.sqrt(torch.sum((img1 - img2).pow(2),dim=-1)).cpu().numpy()
            dn = 0.
            n = 0
            for i in range(xlen):
                for j in range(ylen):
                    for i2 in xalign[i]:
                        for j2 in yalign[j]:
                            dn += distm[i][j][i2][j2]
                            n += 1                       
            Dist.append([pred[ii],dn/n])
        Dist = np.array(sorted(Dist,key=(lambda x:x[1])))
        pred2.append(list(map(int,Dist[:,0])))
        dist2.append(list(Dist[:,1]))
    return pred2,dist2

def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")
            all_features_a = np.empty((len(eval_ds), 8,8,384), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features_a, features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            features_a = features_a.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
            all_features_a[indices.numpy(), :] = features_a
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features_a, features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            features_a = features_a.cpu().numpy()

            if pca != None:
                features = pca.transform(features)
            
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx   = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
                all_features_a[indices.numpy(), :] = features_a
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    queries_features_a = all_features_a[eval_ds.database_num:]
    database_features_a = all_features_a[:eval_ds.database_num]

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))


    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                #print(query_index,i)
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str =", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    logging.info(f"Global retrieval recalls: {recalls_str}")

    predictions,distances = rerank(predictions,queries_features_a,database_features_a)
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                #print(query_index,i)
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str