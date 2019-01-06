from CaptchaAnalysis import CaptchaAnalysis
import sys




print 'Using instruction......'
print 'python captcha_train.py train_features.mat costvalue pcavar'
print 'default train data name dataset/train/train_features.mat, default costvalue=100.0 default pcavar=0.95'
print '---------------------'

if len(sys.argv)<2:
    cst=100
    pcavar=0.95
    file='dataset/train/train_features.mat' 
elif len(sys.argv)<3:
    file=str(sys.argv[1])
    cst=100
    pcavar=0.95
elif len(sys.argv)<4:
    file=str(sys.argv[1])
    cst=float(sys.argv[2])
    pcavar=0.95
else:
    file=str(sys.argv[1])
    cst=float(sys.argv[2])
    pcavar=float(sys.argv[3])



ca=CaptchaAnalysis() 
result=ca.train(file,cost=cst,pcavar=pcavar)
#result=ca.trainNB(file)

print 'Success rate of ', file , ' costvalue=', cst , ' and pcavar=', pcavar
print ''

for i in range(0,11):
    if i<10:
        print i, "success rate:", result[i,0]/result[i,1]
    else:
        print "non numbered parts success rate:", result[i,0]/result[i,1]


