import torch
import torch.nn.functional as F

class tripletLossDTW(torch.nn.Module):

    def __init__(self):
        super(tripletLossDTW,self).__init__()
        return

    def forward(self, feature_data):
        anchor, positive, negative = feature_data[0], feature_data[1], feature_data[2]
        PalignX = self.dtw_distance(anchor, positive,"x")
        PalignY = self.dtw_distance(anchor, positive,"y")

        NalignX = self.dtw_distance(anchor, negative,"x")
        NalignY = self.dtw_distance(anchor, negative,"y")
        lossa = torch.zeros(1).cuda()
        for i in range(anchor.shape[0]):
            distP = self.dist(anchor[i],positive[i],PalignX[i],PalignY[i])
            distN = self.dist(anchor[i],negative[i],NalignX[i],NalignY[i])
            lossa += torch.clamp(distP-distN+0.1, min=0.)
        return lossa

    def dist(self,img1,img2,xalign,yalign):
        img1_align = torch.zeros_like(img1[0][0]).unsqueeze(0).cuda()
        img2_align = torch.zeros_like(img1[0][0]).unsqueeze(0).cuda()
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                for i2 in xalign[i]:
                    for j2 in yalign[j]:
                        img1_align = torch.cat((img1_align, img1[i][j].unsqueeze(0)),dim=0)
                        img2_align = torch.cat((img2_align, img2[i2][j2].unsqueeze(0)),dim=0)                        
        ImgDistgpu = torch.sqrt(torch.sum((img1_align - img2_align).pow(2), dim=-1))#
        ImgDist = torch.sum(ImgDistgpu)/(ImgDistgpu.shape[0]-1)
        return ImgDist

    def dtw_distance(self,feature1,feature2,diretFlag):
        if diretFlag == "x":
            seqlen = feature1.shape[1]
            f1 = feature1.permute(1,0,2,3).expand(seqlen,-1,-1,-1,-1).permute(1,0,2,3,4)
            f2 = feature2.permute(1,0,2,3).expand(seqlen,-1,-1,-1,-1)    
            tmp = torch.sqrt(torch.sum((f1 - f2).pow(2), dim=[3, 4])/8.)+1e-8

        elif diretFlag == "y":
            seqlen = feature1.shape[2]
            f1 = feature1.permute(2,0,1,3).expand(seqlen,-1,-1,-1,-1).permute(1,0,2,3,4)
            f2 = feature2.permute(2,0,1,3).expand(seqlen,-1,-1,-1,-1)            
            tmp = torch.sqrt(torch.sum((f1 - f2).pow(2), dim=[3, 4]))+1e-8
        M = tmp.permute(2,0,1)
        D = torch.ones(M.shape[0],M.shape[1],M.shape[2],4).cuda()
        alignl = []
        for k in range(len(D)):
            D[k,0,0,0] = M[k,0,0]
            D[k,0,0,1] = -1
            D[k,0,0,2] = -1
            D[k,0,0,3] = 1
        for i in range(1, seqlen):
            D[:,i,0,0] = M[:,i,0] + D[:,i-1,0,0]
            D[:,i,0,1] = i - 1
            D[:,i,0,2] = 0
            D[:,i,0,3] = 1 + D[:,i-1,0,3]
        for j in range(1, seqlen):
            D[:,0,j,0] = M[:,0,j] + D[:,0,j-1,0]
            D[:,0,j,1] = 0
            D[:,0,j,2] = j - 1
            D[:,0,j,3] = 1 + D[:,0,j-1,3]
        for i in range(1, seqlen):
            for j in range(1, seqlen):
                cand1 = (D[:,i - 1,j,0] / D[:,i - 1,j,3]).unsqueeze(0)
                cand2 = (D[:,i,j - 1,0] / D[:,i,j - 1,3]).unsqueeze(0)
                cand3 = (D[:,i - 1,j - 1,0] / D[:,i - 1,j - 1,3]).unsqueeze(0)
                cands = torch.cat([cand1,cand2,cand3],dim=0)
                _, indices = torch.min(cands,dim=0)
                for k in range(len(D)):
                    if(indices[k]==0):
                        D[k,i,j,0] = M[k,i,j] + D[k,i - 1,j,0]
                        D[k,i,j,1] = i - 1
                        D[k,i,j,2] = j
                        D[k,i,j,3] = 1 + D[k,i - 1,j,3]
                    elif(indices[k]==1):
                        D[k,i,j,0] = M[k,i,j] + D[k,i,j - 1,0]
                        D[k,i,j,1] = i
                        D[k,i,j,2] = j - 1
                        D[k,i,j,3] = 1 + D[k,i,j - 1,3]
                    elif(indices[k]==2):
                        D[k,i,j,0] = M[k,i,j] + D[k,i - 1,j - 1,0]
                        D[k,i,j,1] = i - 1
                        D[k,i,j,2] = j - 1
                        D[k,i,j,3] = 1 + D[k,i - 1,j - 1,3]
        for k in range(len(D)):
            align={}
            ii=seqlen-1
            jj=seqlen-1
            while(ii!=-1):
                align.setdefault(ii, []).append(jj)  # if Key "ii" in dictionary, append "jj" to this list; or else new "ii: []" in dict and append "jj" to list
                ii,jj=(int(D[k][ii][jj][1]),int(D[k][ii][jj][2]))
            alignl.append(align)
        return alignl