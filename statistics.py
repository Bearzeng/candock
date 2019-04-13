import numpy as np
import matplotlib.pyplot as plt
import util

def stage(stages):	
    #N3->0  N2->1  N1->2  REM->3  W->4
    stage_cnt=np.array([0,0,0,0,0])
    for i in range(len(stages)):
        stage_cnt[stages[i]] += 1
    stage_cnt_per = stage_cnt/len(stages) 
    util.writelog('statistics of dataset [S3 S2 S1 R W]: '+str(stage_cnt))
    print('statistics of dataset [S3 S2 S1 R W]:\n',stage_cnt,'\n',stage_cnt_per)
    return stage_cnt,stage_cnt_per

def result(mat):
    wide=mat.shape[0]
    sub_acc = np.zeros(wide)
    sub_recall = np.zeros(wide)
    err = 0
    for i in range(wide):
        if np.sum(mat[i]) == 0 :
            sub_recall[i] = 0
        else:
            sub_recall[i]=mat[i,i]/np.sum(mat[i])
        err += mat[i,i]
        sub_acc[i] = (np.sum(mat)-((np.sum(mat[i])+np.sum(mat[:,i]))-2*mat[i,i]))/np.sum(mat)
    avg_recall = np.mean(sub_recall)
    avg_acc = np.mean(sub_acc)
    err = 1-err/np.sum(mat)
    return avg_recall,avg_acc,err

def stagefrommat(mat):
    wide=mat.shape[0]
    stage_num = np.zeros(wide,dtype='int')
    for i in range(wide):
        stage_num[i]=np.sum(mat[i])
    util.writelog('statistics of dataset [S3 S2 S1 R W]:\n'+str(stage_num),True)



def show(plot_result,epoch):
    train_recall = np.array(plot_result['train'])
    test_recall = np.array(plot_result['test'])
    plt.figure('running recall')
    plt.clf()
    train_recall_x = np.linspace(0,epoch,len(train_recall))
    test_recall_x = np.linspace(0,int(epoch),len(test_recall))
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.ylim((0,1))
    if epoch <10:
        plt.xlim((0,10))
    else:
        plt.xlim((0,epoch))
    plt.plot(train_recall_x,train_recall,label='train',linewidth = 2.0,color = 'red')
    plt.plot(test_recall_x,test_recall,label='test', linewidth = 2.0,color = 'blue')
    plt.legend(loc=4)
    plt.savefig('./running_recall.png')

    # plt.draw()
    # plt.pause(0.01)


def main():
    mat=[[37980,1322,852,2,327],[3922,8784,3545,0,2193],[1756,5136,99564,1091,991],[18,1,7932,4063,14],[1361,1680,465,0,23931]]
    mat = np.array(mat)
    avg_recall,avg_acc,err = result(mat)
    print(avg_recall,avg_acc,err)
if __name__ == '__main__':
    main()

