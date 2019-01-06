import scipy.io.wavfile
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from rastaplp import rastaplp
from sklearn.svm import SVR,SVC,NuSVC,LinearSVC,l1_min_c
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from scipy import stats
import pickle

class CaptchaAnalysis:

    def __init__(self):
        self.X={}
        self.iii=0
    def take_hdf5_item_structure(self,g, offset='    ') :

        if isinstance(g,h5py.Group) :
            self.X[self.iii]=g.name 
            self.iii+=1 
        if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
            for key,val in dict(g).iteritems() :
                subg = val            
                self.take_hdf5_item_structure(subg, offset + '    ')

    def loadtrainparam(self,traineddatafile=""):
        if traineddatafile=="":
            pickle_in = open("last_train_filename","rb")
            traineddatafile= pickle.load(pickle_in)        
            pickle_in.close()
        pickle_in = open(traineddatafile,"rb")
        self.pca = pickle.load(pickle_in)        
        self.el = pickle.load(pickle_in)        
        self.clf = pickle.load(pickle_in)
        pickle_in.close()

    

    def loaddata(self,matfile):
        file = h5py.File(matfile, 'r')  
        self.take_hdf5_item_structure(file)
        Ceps={}
        Spec={}
        Label={}
        for i in range(2,len(self.X)):
            Ceps[i-2]=np.array(file[self.X[i]+'/ceps'])
            Spec[i-2]=np.array(file[self.X[i]+'/spec'])
            Label[i-2]=file[self.X[i]+'/label'][0]
        file.close()
        print len(Ceps), ' number of data read from given dataset'
        m,n=Ceps[0].shape
        self.Ftrain=np.zeros((len(Ceps),m*n))
        self.FtrLabel=np.zeros(len(Ceps))

        for i in range(0,len(Ceps)):
              self.Ftrain[i,:]=Ceps[i].flatten()
              self.FtrLabel[i]=Label[i]+1


    def crossval(self,cost=1,pcavar=0.95):
                
        cres=0
        for k in range(0,4):
            I=np.zeros(len(self.FtrLabel))
            I[k:-1:4]=1
            train=self.Ftrain[I==0,:]
            trLabel=self.FtrLabel[I==0]

            zstrain=stats.zscore(train)

            pca = PCA()  #n_components=m*n
            # dimensional redundancy step        
            x_new=pca.fit_transform(zstrain) 
            csv=np.cumsum(pca.explained_variance_ratio_)
            el = np.where(csv>pcavar)[0][0]

            xx=pca.transform(train)
            train_pc =xx[:,0:el+1]
            #train_pc =x_new[:,0:el+1]

            mn=train.mean(0)
            vr=train.std(0)   

        
            # train step
            clf={}
            results=np.zeros((11,2))
            #trGroup=np.zeros(len(trLabel))                   
        
            clf =  SVC(C=cost) #NuSVC(nu=0.02)      
            clf.fit(train_pc, trLabel)

            testd=self.Ftrain[I==1,:]
            test_pc_pca=pca.transform(testd)
            test_pc =test_pc_pca[:,0:el+1]
            ttLabel=self.FtrLabel[I==1]

            w=clf.predict(test_pc) 
            for i in range(0,11):
                results[i,0]=np.sum(w[ttLabel==i+1]==i+1)
                results[i,1]=np.sum(ttLabel==i+1)
                cres+= results[i,0]/results[i,1]
            

        return cres/44

    def trainNB(self,matfile):
        file = h5py.File(matfile, 'r')  
        self.take_hdf5_item_structure(file)
        Ceps={}
        Spec={}
        Label={}
        for i in range(2,len(self.X)):
            Ceps[i-2]=np.array(file[self.X[i]+'/ceps'])
            Spec[i-2]=np.array(file[self.X[i]+'/spec'])
            Label[i-2]=file[self.X[i]+'/label'][0]
        file.close()
        print len(Ceps), ' number of data read from given dataset'
        m,n=Ceps[0].shape
        train=np.zeros((len(Ceps),m*n))
        trLabel=np.zeros(len(Ceps))

        for i in range(0,len(Ceps)):
              train[i,:]=Ceps[i].flatten()
              trLabel[i]=Label[i]+1
        

        zstrain=stats.zscore(train)
        
        mn=train.mean(0)
        vr=train.std(0)   
        
        train_pc=zstrain
        
        # train step
        clf={}
        results=np.zeros((11,2))
        #trGroup=np.zeros(len(trLabel))                   
        
        clf = GaussianNB()

           
        clf.fit(train_pc, trLabel)
        w=clf.predict(train_pc) 
        for i in range(0,11):
            results[i,0]=np.sum(w[trLabel==i+1]==i+1)
            results[i,1]=np.sum(trLabel==i+1)
        

        filen=matfile[:-4]+"_NB"
        pickle_out = open(filen,"wb")
        pickle.dump(mn, pickle_out)        
        pickle.dump(vr, pickle_out)
        pickle.dump(clf, pickle_out)
        pickle_out.close()

        pickle_out2 = open("last_train_filename","wb")
        pickle.dump(filen, pickle_out2)
        pickle_out2.close()

        return results


    def train(self,matfile,cost=1,pcavar=0.95):
        file = h5py.File(matfile, 'r')  
        self.take_hdf5_item_structure(file)
        Ceps={}
        Spec={}
        Label={}
        for i in range(2,len(self.X)):
            Ceps[i-2]=np.array(file[self.X[i]+'/ceps'])
            Spec[i-2]=np.array(file[self.X[i]+'/spec'])
            Label[i-2]=file[self.X[i]+'/label'][0]
        file.close()
        print len(Ceps), ' number of data read from given dataset'
        m,n=Ceps[0].shape
        train=np.zeros((len(Ceps),m*n))
        trLabel=np.zeros(len(Ceps))

        for i in range(0,len(Ceps)):
              train[i,:]=Ceps[i].flatten()
              trLabel[i]=Label[i]+1
        

        zstrain=stats.zscore(train)
        
        pca = PCA()  #n_components=m*n
        # dimensional redundancy step        
        x_new=pca.fit_transform(zstrain) 
        csv=np.cumsum(pca.explained_variance_ratio_)
        el = np.where(csv>pcavar)[0][0]

        xx=pca.transform(train)
        train_pc =xx[:,0:el+1]
        #train_pc =x_new[:,0:el+1]

        
        mn=train.mean(0)
        vr=train.std(0)   

        
        # train step
        clf={}
        results=np.zeros((11,2))
        #trGroup=np.zeros(len(trLabel))                   
        
        clf =  SVC(C=cost) #NuSVC(nu=0.02)      
        clf.fit(train_pc, trLabel)
        w=clf.predict(train_pc) 
        for i in range(0,11):
            results[i,0]=np.sum(w[trLabel==i+1]==i+1)
            results[i,1]=np.sum(trLabel==i+1)
        

        filen=matfile[:-4]+"_"+str(cost)+"_"+str(pcavar)
        pickle_out = open(filen,"wb")
        pickle.dump(pca, pickle_out)        
        pickle.dump(el, pickle_out)
        pickle.dump(clf, pickle_out)
        pickle_out.close()

        pickle_out2 = open("last_train_filename","wb")
        pickle.dump(filen, pickle_out2)
        pickle_out2.close()

        return results

    def running_mean(self,x, N):
        cumsum = np.cumsum(x) #(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 

    def test(self,testfile):

        digit_number = len(testfile)-4;        

        fs, ff = scipy.io.wavfile.read(testfile)
        length = len(ff);
        f = ff[0 : int(np.floor(length/2))]/32768.0;
        energy_f = np.abs(f)*f*f
        y=self.running_mean(energy_f, 100)
        mean_locs = (y>0.0003)
        zero_locs = (y<0.00001)
        flag = 0;

        location = {}; # np.zeros(1000)#
        cnt=0

        for i in range(0,len(y)):
            if (flag==0 and mean_locs[i]==1):
                location[cnt] = i
                cnt+=1
                flag = 1;
            elif (flag==1 and mean_locs[i]==0):
                location[cnt] = i-1
                cnt+=1
                flag = 0
        index = -1
        startpoint = 0
        endpoint = 0
        segment={}
        ceps={}
        spec={}
        svmLabel={}
        digit_count=0
        result=""
        for i in range(1,1+int(np.floor(len(location)/2))):
            if location[2*i-1]-location[2*i-2]>200:
                for j in range(int(location[2*i-2]),-1,-1 ):
                    if zero_locs[j]==1:
                        startpoint = j
                        break   
                if startpoint< np.floor((location[2*i-2]+location[2*i-1])/2)-1750:
                    startpoint =int(np.floor((location[2*i-2]+location[2*i-1])/2)-1750)
                

                if (startpoint > endpoint and startpoint +3500 < len(f)):
                    index = index+1
                    segment[index] = f[startpoint : startpoint+3501]
                    ceps[index], spec[index], p_spectrum, lpc_anal, F, M = rastaplp(segment[index], fs,0, 12)        
                    endpoint = startpoint                   

                    m,n=ceps[index].shape
          
                    testd=np.zeros((1,m*n))        
                    testd[0,:]=ceps[index].flatten()

                    #testd=(testd-mn)/ vr[np.newaxis,:]

                    test_pc_pca=self.pca.transform(testd)
                    test_pc =test_pc_pca[:,0:self.el+1]

                    #Score=np.zeros(11)
                    #for cls in range(0,11):
                    #    prelabel=clf[cls].predict(test_pc)
                    #    #Score[cls]=clf[cls].score(testd,[1])                       

                    svmLabel[index]=self.clf.predict(test_pc)                    
                                       
                    #svmLabel[index]=np.argmax(Score)

                    if svmLabel[index]>=1 and svmLabel[index]<=10:
                        endpoint=startpoint+3500
                        digit_count+=1
                        result+=str(int(svmLabel[index][0])-1)
                        #result+=str(int(svmLabel[index]))
                    else:
                        endpoint=startpoint                       

            if digit_count==digit_number:
                break
        return result            
           
    def testNB(self,testfile):

        digit_number = len(testfile)-4;        

        fs, ff = scipy.io.wavfile.read(testfile)
        length = len(ff);
        f = ff[0 : int(np.floor(length/2))]/32768.0;
        energy_f = np.abs(f)*f*f
        y=self.running_mean(energy_f, 100)
        mean_locs = (y>0.0003)
        zero_locs = (y<0.00001)
        flag = 0;

        location = {}; # np.zeros(1000)#
        cnt=0

        for i in range(0,len(y)):
            if (flag==0 and mean_locs[i]==1):
                location[cnt] = i
                cnt+=1
                flag = 1;
            elif (flag==1 and mean_locs[i]==0):
                location[cnt] = i-1
                cnt+=1
                flag = 0
        index = -1
        startpoint = 0
        endpoint = 0
        segment={}
        ceps={}
        spec={}
        svmLabel={}
        digit_count=0
        result=""
        for i in range(1,1+int(np.floor(len(location)/2))):
            if location[2*i-1]-location[2*i-2]>200:
                for j in range(int(location[2*i-2]),-1,-1 ):
                    if zero_locs[j]==1:
                        startpoint = j
                        break   
                if startpoint< np.floor((location[2*i-2]+location[2*i-1])/2)-1750:
                    startpoint =int(np.floor((location[2*i-2]+location[2*i-1])/2)-1750)
                

                if (startpoint > endpoint and startpoint +3500 < len(f)):
                    index = index+1
                    segment[index] = f[startpoint : startpoint+3501]
                    ceps[index], spec[index], p_spectrum, lpc_anal, F, M = rastaplp(segment[index], fs,0, 12)        
                    endpoint = startpoint                   

                    m,n=ceps[index].shape
          
                    testd=np.zeros((1,m*n))        
                    testd[0,:]=ceps[index].flatten()

                    test_pc=(testd-self.pca)/ self.el[np.newaxis,:]

                                    

                    svmLabel[index]=self.clf.predict(test_pc)                    
                                       
                    #svmLabel[index]=np.argmax(Score)

                    if svmLabel[index]>=1 and svmLabel[index]<=10:
                        endpoint=startpoint+3500
                        digit_count+=1
                        result+=str(int(svmLabel[index][0])-1)
                        #result+=str(int(svmLabel[index]))
                    else:
                        endpoint=startpoint                       

            if digit_count==digit_number:
                break
        return result           
           
    
if __name__ == "__main__":    
    ca=CaptchaAnalysis()     
    ca.loadtrainparam('dataset/train/train_features_50.0_0.9')

    result=ca.test('dataset/test/04648.wav')
    print (result)
